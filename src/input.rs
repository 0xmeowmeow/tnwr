//! Input — the queue, the key tree, and addressing as a pure state machine.
//!
//! CLAUDE.md §3, locked: *"The input queue must be independent of the render
//! loop. Buffer input separately, drain it fully, never do work in the render
//! path."*
//!
//! So this module does no I/O at all, and nothing in it waits on a node.
//!
//! - **The queue.** A keystroke arriving from the window system is pushed and
//!   nothing else. Receipt is O(1), so a slow node can delay *processing* but
//!   never *receipt*, and never drop a key.
//! - **The key tree.** Every `keys` table in the namespace, fetched off the
//!   main thread and looked up per keystroke. Resolving `k b p s` is four map
//!   lookups, not four network round-trips.
//! - **Addressing.** `apply` turns one key into state and sometimes an
//!   `Effect`. The effect goes to an I/O worker; when the node answers, the
//!   `Outcome` comes back and `settle` applies it. Both are pure, so the whole
//!   ordering discipline is tested here without a thread, a window or a node.
//!
//! The ordering discipline, stated once: **the drain stops at an effect.** Keys
//! behind it stay queued until its outcome settles. Arriving at a writable leaf
//! opens an editor, and a key typed after that — even typed before the node
//! answered — must land in the editor rather than be read as an address. That
//! only works if the key is interpreted *after* the answer, not merely after
//! it was received.

use std::collections::{HashMap, VecDeque};

/// A keystroke, after the window system's representation has been left behind
/// at the edge. Space arrives as `Char(' ')`.
#[derive(Clone, Debug, PartialEq)]
pub enum Key {
    Char(char),
    Enter,
    Escape,
    Backspace,
}

/// Something `apply` cannot decide alone, because deciding it means asking the
/// node. Performed by the I/O worker, never on the main thread.
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    /// Arrived at a leaf. The worker reads it: a `0`/`1` switch flips, a
    /// writable value opens the editor, anything else is shown.
    Arrive(String),
    /// Enter pressed in the editor: write this value to this path.
    Commit(String, String),
}

/// What the node said when an effect was performed.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// A `0`/`1` switch at `path` went `from` → `to`.
    Toggled { path: String, from: String, to: String },
    /// A writable leaf that is not a switch: open the editor on this value.
    Edit(String),
    /// A read-only leaf: show its value.
    Shown { path: String, value: String },
    /// A commit landed.
    Wrote(String),
    /// The node could not do it, or could not be reached.
    Failed { path: String, error: String },
}

impl Outcome {
    /// Whether the node's state changed, so the view should be re-fetched.
    pub fn changed_node(&self) -> bool {
        matches!(self, Outcome::Toggled { .. } | Outcome::Wrote(_))
    }
}

/// Every key table in the namespace, keyed by directory path (`""` is root).
///
/// A path with a table is a directory; a path without one is a leaf. That is
/// the whole test, so a directory with no addressable children still needs an
/// entry — an empty one.
#[derive(Clone, Debug, Default)]
pub struct KeyTree {
    tables: HashMap<String, Vec<(char, String)>>,
}

impl KeyTree {
    /// Parse a `keys` file: `<letter> <name>` per line.
    pub fn parse(text: &str) -> Vec<(char, String)> {
        text.lines()
            .filter_map(|line| {
                let mut it = line.splitn(2, ' ');
                let ch = it.next()?.chars().next()?;
                let name = it.next()?.trim();
                (!name.is_empty()).then(|| (ch, name.to_string()))
            })
            .collect()
    }

    pub fn insert(&mut self, path: &str, table: Vec<(char, String)>) {
        self.tables.insert(path.to_string(), table);
    }

    pub fn table(&self, path: &str) -> &[(char, String)] {
        self.tables.get(path).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn is_dir(&self, path: &str) -> bool {
        self.tables.contains_key(path)
    }

    /// True once any snapshot has arrived — including an offline one, which
    /// carries an empty root. Keys typed before that wait in the queue rather
    /// than being rejected against a tree that simply has not loaded yet.
    pub fn loaded(&self) -> bool {
        self.is_dir("")
    }
}

/// Where addressing is, and what has been typed to get there.
#[derive(Clone, Debug, Default)]
pub struct Addressing {
    /// Resolved namespace path. Every address starts at the root, so every
    /// address is absolute — ISPF's `=3.4` comes free.
    pub cursor: String,
    pub typed: String,
    /// `Some(buffer)` while typing a value into a writable leaf.
    pub editing: Option<String>,
    pub flash: String,
    /// Keys that reached a real node, plus effects the node carried out. Only
    /// grows. Config generations read it: "last known good" needs input that
    /// *worked*, and a letter that hits nothing does not count.
    pub resolved: u64,
}

impl Addressing {
    pub fn reset(&mut self) {
        self.cursor.clear();
        self.typed.clear();
    }

    /// Apply one key. No I/O; anything that needs the node comes back as an
    /// `Effect`.
    pub fn apply(&mut self, key: Key, tree: &KeyTree) -> Option<Effect> {
        if let Some(buf) = self.editing.as_mut() {
            // The editor takes the keyboard entirely: letters are letters.
            return match key {
                Key::Char(c) => {
                    buf.push(c);
                    None
                }
                Key::Backspace => {
                    buf.pop();
                    None
                }
                Key::Enter => {
                    let value = self.editing.take().unwrap_or_default();
                    let path = std::mem::take(&mut self.cursor);
                    self.typed.clear();
                    Some(Effect::Commit(path, value))
                }
                Key::Escape => {
                    self.editing = None;
                    self.flash = "cancelled".into();
                    self.reset();
                    None
                }
            };
        }

        match key {
            Key::Escape => {
                self.flash.clear();
                self.reset();
                None
            }
            Key::Char(c) => {
                self.flash.clear();
                let hit = tree
                    .table(&self.cursor)
                    .iter()
                    .find(|(k, _)| *k == c)
                    .map(|(_, name)| name.clone());
                let Some(name) = hit else {
                    self.flash = format!("no '{c}' here");
                    return None;
                };
                let next = format!("{}/{}", self.cursor, name);
                self.resolved += 1;
                self.typed.push(c);
                self.cursor = next.clone();
                if tree.is_dir(&next) {
                    None
                } else {
                    // A leaf: arriving is the operation.
                    Some(Effect::Arrive(next))
                }
            }
            Key::Enter | Key::Backspace => None,
        }
    }

    /// Apply what the node said. Pure — the I/O already happened elsewhere.
    pub fn settle(&mut self, outcome: Outcome) {
        if !matches!(outcome, Outcome::Failed { .. }) {
            self.resolved += 1;
        }
        match outcome {
            Outcome::Toggled { path, from, to } => {
                self.flash = format!("{path} : {from} -> {to}");
                self.reset();
            }
            Outcome::Edit(value) => {
                // The cursor stays on the leaf, so Enter knows where to write.
                self.editing = Some(value);
            }
            Outcome::Shown { path, value } => {
                let short: String = value.chars().take(60).collect();
                self.flash = format!("{path} = {short}");
                self.reset();
            }
            Outcome::Wrote(path) => {
                self.flash = format!("wrote {path}");
            }
            Outcome::Failed { path, error } => {
                self.flash = format!("{path}: {error}");
                self.editing = None;
                self.reset();
            }
        }
    }
}

/// Drain keys until one produces an effect, and stop there.
///
/// Returns how many keys were taken and the effect, if any. The caller sends
/// the effect to the I/O worker and does not drain again until its outcome has
/// been settled — the keys behind it wait, in order, in the queue.
pub fn drain_until_effect(
    queue: &mut VecDeque<Key>,
    addr: &mut Addressing,
    tree: &KeyTree,
) -> (usize, Option<Effect>) {
    let mut n = 0;
    while let Some(key) = queue.pop_front() {
        n += 1;
        if let Some(effect) = addr.apply(key, tree) {
            return (n, Some(effect));
        }
    }
    (n, None)
}

/// The four physical keys of the escape chord.
///
/// Physical, not logical: a keymap translates physical keys into logical ones,
/// so matching the physical position is what makes the chord unshadowable by
/// any remap. On a non-QWERTY board it is the same four *positions*, whatever
/// they happen to be labelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChordKey {
    T = 0,
    N = 1,
    W = 2,
    R = 3,
}

/// Recovery layer 1, from the brief: *"immutable escape chord in the core,
/// unshadowable by any keymap."* `t n w r`, all four held at once.
///
/// Held state is tracked from presses *and* releases, so typing the letters —
/// even fast, with rollover — never fires it.
///
/// The chord's own letters must not act on the way to it: pressed in the
/// wrong order they could spell a switch's address before the fourth key
/// lands, and an escape that flips something is not an escape. So a chord-key
/// press is withheld at receipt. It goes to the queue when that key is
/// released, or when any other key is about to enter the queue — in press
/// order either way, so type-ahead loses nothing and reorders nothing. If the
/// chord completes, the withheld presses are dropped. Once it has fired, chord
/// keys never reach the queue until all four are up, so a chord held into
/// auto-repeat types nothing.
#[derive(Clone, Debug, Default)]
pub struct EscapeChord {
    held: [bool; 4],
    /// Presses not yet given to the queue, in the order they went down.
    withheld: Vec<(ChordKey, Vec<Key>)>,
    /// Fired, and not every chord key is up yet.
    latched: bool,
}

impl EscapeChord {
    /// A chord key went down, translating to `keys`. True exactly when this
    /// press completes the chord.
    pub fn press(
        &mut self,
        k: ChordKey,
        keys: Vec<Key>,
        repeat: bool,
        queue: &mut VecDeque<Key>,
    ) -> bool {
        if self.latched {
            let was = self.complete();
            self.held[k as usize] = true;
            return !was && self.complete();
        }
        if repeat {
            // Down long enough to auto-repeat: this is typing, not a chord.
            self.flush(queue);
            queue.extend(keys);
            return false;
        }
        self.held[k as usize] = true;
        if self.complete() {
            self.withheld.clear();
            self.latched = true;
            return true;
        }
        self.withheld.push((k, keys));
        false
    }

    /// A chord key came up. Its withheld press goes to the queue, behind any
    /// pressed before it. True when this release lets go of a chord that
    /// fired — the last of the four coming up — which is when a generation
    /// picked while holding it is switched to.
    pub fn release(&mut self, k: ChordKey, queue: &mut VecDeque<Key>) -> bool {
        self.held[k as usize] = false;
        if self.latched {
            self.latched = self.held.iter().any(|h| *h);
            return !self.latched;
        }
        if let Some(i) = self.withheld.iter().position(|(w, _)| *w == k) {
            for (_, keys) in self.withheld.drain(..=i) {
                queue.extend(keys);
            }
        }
        false
    }

    /// The chord has fired and is still held. While it is, space picks a
    /// config generation instead of typing.
    pub fn latched(&self) -> bool {
        self.latched
    }

    /// Another key is about to enter the queue. Anything withheld was typing
    /// after all, and goes first.
    pub fn before_other(&mut self, queue: &mut VecDeque<Key>) {
        self.flush(queue);
    }

    /// Focus left. Releases for keys down now will never arrive, and a stale
    /// held set would fire the chord on one later press, so forget what is
    /// held. What was pressed was typed, and is kept.
    pub fn focus_lost(&mut self, queue: &mut VecDeque<Key>) {
        self.flush(queue);
        self.held = [false; 4];
        self.latched = false;
    }

    fn flush(&mut self, queue: &mut VecDeque<Key>) {
        for (_, keys) in self.withheld.drain(..) {
            queue.extend(keys);
        }
    }

    fn complete(&self) -> bool {
        self.held.iter().all(|h| *h)
    }
}

/// What the escape chord does: return to a known state, unconditionally.
///
/// Drops keys not yet interpreted, cancels the editor without committing, and
/// returns addressing to the root. It does not undo an effect the node already
/// performed — that is recovery layer 2, config generations. The guarantee here
/// is narrower and absolute: afterwards you are at the root, nothing half-done.
pub fn escape(queue: &mut VecDeque<Key>, addr: &mut Addressing) {
    queue.clear();
    addr.editing = None;
    addr.reset();
    addr.flash = "escape · t n w r".into();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tokeniser's namespace, as far as addressing cares.
    fn tree() -> KeyTree {
        let mut t = KeyTree::default();
        t.insert("", KeyTree::parse("p params\ni in\nk fork\n"));
        t.insert("/params", KeyTree::parse("s special\nt tokeniser\n"));
        t.insert("/in", KeyTree::parse("x text\n"));
        t.insert("/fork", KeyTree::parse("b b\n"));
        t.insert("/fork/b", KeyTree::parse("p params\n"));
        t.insert("/fork/b/params", KeyTree::parse("s special\n"));
        t
    }

    fn keys(s: &str) -> VecDeque<Key> {
        s.chars().map(Key::Char).collect()
    }

    /// Run a queue to empty the way the app does — drain to an effect, let a
    /// fake node answer, settle, repeat — and return the effects in order.
    fn run(
        q: &mut VecDeque<Key>,
        a: &mut Addressing,
        t: &KeyTree,
        mut node: impl FnMut(&Effect) -> Outcome,
    ) -> Vec<Effect> {
        let mut seen = Vec::new();
        loop {
            let (_, effect) = drain_until_effect(q, a, t);
            let Some(effect) = effect else { break };
            let outcome = node(&effect);
            seen.push(effect);
            a.settle(outcome);
        }
        seen
    }

    fn editor(e: &Effect) -> Outcome {
        match e {
            Effect::Arrive(_) => Outcome::Edit(String::new()),
            Effect::Commit(p, _) => Outcome::Wrote(p.clone()),
        }
    }

    #[test]
    fn four_keystrokes_resolve_without_io() {
        let t = tree();
        let mut a = Addressing::default();
        let (n, e) = drain_until_effect(&mut keys("kbps"), &mut a, &t);
        assert_eq!(n, 4);
        assert_eq!(e, Some(Effect::Arrive("/fork/b/params/special".into())));
        assert_eq!(a.typed, "kbps");
    }

    #[test]
    fn keys_behind_an_effect_wait_until_it_settles() {
        // The whole discipline in one test. `i x` reaches a writable leaf and
        // the drain stops; "hi" was already queued and must NOT be read as the
        // addresses h and i while the node is still answering.
        let t = tree();
        let mut a = Addressing::default();
        let mut q = keys("ixhi");

        let (n, e) = drain_until_effect(&mut q, &mut a, &t);
        assert_eq!(e, Some(Effect::Arrive("/in/text".into())));
        assert_eq!(n, 2);
        assert_eq!(q.len(), 2, "h and i wait in the queue, uninterpreted");

        a.settle(Outcome::Edit(String::new())); // the node answers
        let (_, e) = drain_until_effect(&mut q, &mut a, &t);
        assert_eq!(e, None);
        assert_eq!(a.editing.as_deref(), Some("hi"));
        assert_eq!(a.cursor, "/in/text");
    }

    #[test]
    fn enter_commits_the_value_to_the_leaf() {
        let t = tree();
        let mut a = Addressing::default();
        let mut q = keys("ixoven");
        q.push_back(Key::Enter);
        let effects = run(&mut q, &mut a, &t, editor);
        assert_eq!(
            effects,
            vec![
                Effect::Arrive("/in/text".into()),
                Effect::Commit("/in/text".into(), "oven".into()),
            ]
        );
        assert!(a.editing.is_none());
        assert_eq!(a.flash, "wrote /in/text");
    }

    #[test]
    fn a_switch_settles_as_a_toggle_and_returns_to_the_root() {
        let t = tree();
        let mut a = Addressing::default();
        let mut q = keys("kbps");
        q.extend(keys("ps")); // typed ahead of the toggle's answer
        let effects = run(&mut q, &mut a, &t, |e| match e {
            Effect::Arrive(p) => Outcome::Toggled { path: p.clone(), from: "1".into(), to: "0".into() },
            _ => unreachable!(),
        });
        // The second `p s` resolves from the root, because the toggle reset
        // addressing before those keys were interpreted.
        assert_eq!(
            effects,
            vec![
                Effect::Arrive("/fork/b/params/special".into()),
                Effect::Arrive("/params/special".into()),
            ]
        );
    }

    #[test]
    fn backspace_edits_the_buffer() {
        let t = tree();
        let mut a = Addressing {
            cursor: "/in/text".into(),
            editing: Some("ovem".into()),
            ..Default::default()
        };
        let mut q: VecDeque<Key> = vec![Key::Backspace, Key::Char('n')].into();
        drain_until_effect(&mut q, &mut a, &t);
        assert_eq!(a.editing.as_deref(), Some("oven"));
    }

    #[test]
    fn an_unknown_letter_flashes_and_does_not_move() {
        let t = tree();
        let mut a = Addressing::default();
        drain_until_effect(&mut keys("pz"), &mut a, &t);
        assert_eq!(a.cursor, "/params");
        assert_eq!(a.flash, "no 'z' here");
    }

    #[test]
    fn escape_returns_to_the_root_from_anywhere() {
        let t = tree();
        let mut a = Addressing::default();
        let mut q = keys("kbp");
        q.push_back(Key::Escape);
        drain_until_effect(&mut q, &mut a, &t);
        assert!(a.cursor.is_empty() && a.typed.is_empty());
    }

    #[test]
    fn escape_in_the_editor_cancels_without_committing() {
        let t = tree();
        let mut a = Addressing::default();
        let mut q = keys("ixhalf");
        q.push_back(Key::Escape);
        let effects = run(&mut q, &mut a, &t, editor);
        assert_eq!(effects, vec![Effect::Arrive("/in/text".into())]);
        assert!(a.editing.is_none());
        assert_eq!(a.flash, "cancelled");
    }

    #[test]
    fn a_failure_clears_the_editor_and_resets() {
        let mut a = Addressing {
            cursor: "/in/text".into(),
            editing: Some("half".into()),
            ..Default::default()
        };
        a.settle(Outcome::Failed { path: "/in/text".into(), error: "node gone".into() });
        assert!(a.editing.is_none() && a.cursor.is_empty());
        assert_eq!(a.flash, "/in/text: node gone");
    }

    #[test]
    fn a_letter_is_a_node_property_not_a_position() {
        // Reordering a key table must not change what a letter reaches.
        let mut t = tree();
        t.insert("/params", KeyTree::parse("t tokeniser\ns special\n"));
        let mut a = Addressing::default();
        let (_, e) = drain_until_effect(&mut keys("ps"), &mut a, &t);
        assert_eq!(e, Some(Effect::Arrive("/params/special".into())));
    }

    fn chord_key(ch: char) -> Option<ChordKey> {
        match ch {
            't' => Some(ChordKey::T),
            'n' => Some(ChordKey::N),
            'w' => Some(ChordKey::W),
            'r' => Some(ChordKey::R),
            _ => None,
        }
    }

    /// Feed key events through the chord the way receipt does, and return
    /// what reached the queue and how often the chord fired. `t+` is a press,
    /// `t*` an auto-repeat, `t-` a release.
    fn play(c: &mut EscapeChord, events: &str) -> (String, usize) {
        let mut q = VecDeque::new();
        let mut fired = 0;
        for ev in events.split_whitespace() {
            let mut cs = ev.chars();
            let (ch, op) = (cs.next().unwrap(), cs.next().unwrap());
            let keys = vec![Key::Char(ch)];
            match (chord_key(ch), op) {
                (Some(k), '-') => {
                    c.release(k, &mut q);
                }
                (Some(k), _) => {
                    if c.press(k, keys, op == '*', &mut q) {
                        fired += 1;
                        q.clear(); // what escape() does
                    }
                }
                (None, '-') => {}
                (None, _) => {
                    c.before_other(&mut q);
                    q.extend(keys);
                }
            }
        }
        let typed = q.iter().map(|k| match k { Key::Char(c) => *c, _ => '?' }).collect();
        (typed, fired)
    }

    #[test]
    fn the_chord_fires_once_when_all_four_are_held() {
        let mut c = EscapeChord::default();
        assert_eq!(play(&mut c, "w+ t+ r+ n+"), (String::new(), 1), "the presses are the chord, not letters");
        assert_eq!(play(&mut c, "n* n* n*"), (String::new(), 0), "held into auto-repeat, it types nothing");
        assert_eq!(play(&mut c, "t- t+"), (String::new(), 1), "release and re-press fires again");
        assert_eq!(play(&mut c, "w- r- n- t- t+ t-"), ("t".into(), 0), "all four up, t is a letter again");
    }

    #[test]
    fn typing_the_letters_with_rollover_never_fires_and_keeps_order() {
        // Each key pressed before the last is released. In "turn" three chord
        // letters are down at once; in "wrtn" all four go down, never together.
        let mut c = EscapeChord::default();
        for (word, events) in [
            ("return", "r+ e+ r- t+ e- u+ t- r+ u- n+ r- n-"),
            ("turn", "t+ u+ r+ n+ t- u- r- n-"),
            ("north", "n+ o+ n- r+ o- t+ r- h+ t- h-"),
            ("wrtn", "w+ r+ t+ w- r- n+ t- n-"),
        ] {
            assert_eq!(play(&mut c, events), (word.into(), 0), "{events}");
        }
    }

    #[test]
    fn letting_go_is_reported_once_all_four_are_up() {
        let mut c = EscapeChord::default();
        let mut q = VecDeque::new();
        for ch in ['t', 'n', 'w', 'r'] {
            c.press(chord_key(ch).unwrap(), vec![Key::Char(ch)], false, &mut q);
        }
        assert!(c.latched());
        assert!(!c.release(ChordKey::T, &mut q));
        assert!(!c.release(ChordKey::N, &mut q));
        assert!(!c.release(ChordKey::W, &mut q));
        assert!(c.release(ChordKey::R, &mut q), "the last one up lets go");
        assert!(!c.latched());
        assert!(!c.release(ChordKey::R, &mut q), "and only once");
    }

    #[test]
    fn only_input_that_reaches_something_counts_as_resolved() {
        let t = tree();
        let mut a = Addressing::default();
        drain_until_effect(&mut keys("zq"), &mut a, &t);
        assert_eq!(a.resolved, 0, "letters that hit nothing");
        drain_until_effect(&mut keys("p"), &mut a, &t);
        assert_eq!(a.resolved, 1);
        a.settle(Outcome::Failed { path: "/params".into(), error: "gone".into() });
        assert_eq!(a.resolved, 1, "a failure is not success");
        a.settle(Outcome::Wrote("/in/text".into()));
        assert_eq!(a.resolved, 2);
    }

    #[test]
    fn a_held_letter_that_repeats_is_typing() {
        let mut c = EscapeChord::default();
        assert_eq!(play(&mut c, "t+ t* t* t-"), ("ttt".into(), 0));
    }

    #[test]
    fn the_chord_letters_do_nothing_on_the_way_to_it() {
        // A node that happened to give `w r` to a switch.
        let mut t = KeyTree::default();
        t.insert("", KeyTree::parse("w wifi\n"));
        t.insert("/wifi", KeyTree::parse("r radio\n"));
        let (_, e) = drain_until_effect(&mut keys("wr"), &mut Addressing::default(), &t);
        assert_eq!(e, Some(Effect::Arrive("/wifi/radio".into())), "typed, those letters flip it");

        // Held as the chord in that order, draining after every event the way
        // the app does, they must not.
        let mut a = Addressing::default();
        let mut c = EscapeChord::default();
        let mut q = VecDeque::new();
        let mut effects = Vec::new();
        for ch in ['w', 'r', 't', 'n'] {
            if c.press(chord_key(ch).unwrap(), vec![Key::Char(ch)], false, &mut q) {
                escape(&mut q, &mut a);
            }
            effects.extend(drain_until_effect(&mut q, &mut a, &t).1);
        }
        assert_eq!(effects, vec![]);
        assert_eq!(a.flash, "escape · t n w r");
    }

    #[test]
    fn losing_focus_keeps_what_was_typed_and_forgets_what_is_held() {
        let mut c = EscapeChord::default();
        let mut q = VecDeque::new();
        for ch in ['t', 'n', 'w'] {
            c.press(chord_key(ch).unwrap(), vec![Key::Char(ch)], false, &mut q);
        }
        c.focus_lost(&mut q); // those releases will never arrive
        assert_eq!(q.len(), 3, "presses that reached us were typed");
        assert!(
            !c.press(ChordKey::R, vec![Key::Char('r')], false, &mut q),
            "one press after refocus cannot complete a stale chord"
        );
    }

    #[test]
    fn escape_leaves_nothing_half_done() {
        let mut a = Addressing {
            cursor: "/in/text".into(),
            typed: "ix".into(),
            editing: Some("half-typed".into()),
            ..Default::default()
        };
        let mut q = keys("queued");
        q.push_back(Key::Enter); // would have committed
        escape(&mut q, &mut a);
        assert!(q.is_empty(), "uninterpreted keys are gone, so that Enter never commits");
        assert!(a.editing.is_none() && a.cursor.is_empty() && a.typed.is_empty());
    }

    #[test]
    fn nothing_is_interpreted_before_the_tree_loads() {
        assert!(!KeyTree::default().loaded());
        let mut offline = KeyTree::default();
        offline.insert("", Vec::new());
        assert!(offline.loaded(), "an offline snapshot still counts as loaded");
    }
}
