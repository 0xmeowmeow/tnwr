//! Config generations — recovery layer 2.
//!
//! Brief v2: *"numbered config generations, NixOS-style, selected by a
//! core-held chord."* And: *"Define 'last known good' mechanically: the last
//! generation that survived N seconds of successful input, auto-marked.
//! Otherwise it is a guess with a confident name."*
//!
//! A node serves its config at `/config`: every leaf it flags as config, and
//! every directory that was created (a fork). Whenever that differs from the
//! current generation, the core writes it to disk as the next number.
//! Generations are never modified and never deleted. Rolling back writes an
//! old one back to the node, and it becomes current again.
//!
//! They live in the core, on disk, not in a node — recovery must not depend on
//! the thing being recovered. Kept per node address for now, because the core
//! attaches to one node; when it attaches to a graph, a generation becomes the
//! whole graph's config.
//!
//! Split like `input.rs`. `Generations` is the policy, pure, tested with no
//! clock and no disk. `Store` is the disk. The worker joining them to the node
//! is in `main.rs`.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// How long a generation must keep taking input before it is marked good.
pub const SURVIVE: Duration = Duration::from_secs(30);

/// What an observed config means.
#[derive(Debug, PartialEq, Eq)]
pub enum Seen {
    /// Same as the current generation. Nothing to record.
    Same,
    /// The generation being switched to has taken: it is current again.
    Became(u32),
    /// Differs from the current generation: record it as the next number.
    New,
}

#[derive(Debug)]
pub struct Generations {
    /// Every generation, ascending, with its body.
    known: Vec<(u32, String)>,
    current: Option<u32>,
    good: Option<u32>,
    /// When `current` was first seen holding in this run, or when the escape
    /// chord last restarted the clock. `None` until the node has been seen.
    since: Option<Instant>,
    /// Written back to the node, not yet seen there.
    switching: Option<u32>,
    survive: Duration,
}

impl Generations {
    pub fn new(
        known: Vec<(u32, String)>,
        current: Option<u32>,
        good: Option<u32>,
        survive: Duration,
    ) -> Self {
        Self { known, current, good, since: None, switching: None, survive }
    }

    pub fn body(&self, n: u32) -> Option<&str> {
        self.known.iter().find(|(k, _)| *k == n).map(|(_, b)| b.as_str())
    }

    pub fn current(&self) -> Option<u32> {
        self.current
    }

    /// The node's config, as just read.
    pub fn observe(&mut self, body: &str, now: Instant) -> Seen {
        if let Some(n) = self.switching.take() {
            if self.body(n) == Some(body) {
                self.make_current(n, now);
                return Seen::Became(n);
            }
        }
        if self.current.and_then(|c| self.body(c)) == Some(body) {
            // First sight this run: survival counts from here.
            self.since.get_or_insert(now);
            return Seen::Same;
        }
        Seen::New
    }

    /// `observe` said `New`, and the store gave it this number.
    pub fn recorded(&mut self, n: u32, body: String, now: Instant) {
        self.known.push((n, body));
        self.make_current(n, now);
    }

    /// A generation has been written back to the node. It becomes current
    /// when the node is seen holding it, not before.
    pub fn switching(&mut self, n: u32) {
        self.switching = Some(n);
    }

    /// The escape chord fired. Needing it is not surviving, so the clock
    /// starts again.
    pub fn escaped(&mut self, now: Instant) {
        if self.since.is_some() {
            self.since = Some(now);
        }
    }

    /// Input reached something at `now`. Returns the generation to mark good
    /// if this is the input that proves it survived. Time alone never marks
    /// anything: a generation that broke the keyboard can sit there for an
    /// hour and stay unmarked.
    pub fn input_ok(&mut self, now: Instant) -> Option<u32> {
        let (current, since) = (self.current?, self.since?);
        if self.good == Some(current) || now.saturating_duration_since(since) < self.survive {
            return None;
        }
        self.good = Some(current);
        Some(current)
    }

    fn make_current(&mut self, n: u32, now: Instant) {
        self.current = Some(n);
        self.since = Some(now);
    }

    pub fn view(&self, problem: Option<String>) -> View {
        View {
            numbers: self.known.iter().map(|(n, _)| *n).collect(),
            current: self.current,
            good: self.good,
            problem,
        }
    }
}

/// What the main thread needs to draw generations and pick one while the
/// chord is held. A copy, so picking never waits on the worker.
#[derive(Clone, Debug, Default)]
pub struct View {
    pub numbers: Vec<u32>,
    pub current: Option<u32>,
    pub good: Option<u32>,
    pub problem: Option<String>,
}

impl View {
    /// One tap of space while the chord is held. The first goes to last known
    /// good — or one older than current, if nothing is good yet — and each
    /// after that one generation older, stopping at the oldest.
    pub fn step_back(&self, picked: Option<u32>) -> Option<u32> {
        match picked {
            None => self.good.or_else(|| self.older(self.current?)),
            Some(n) => Some(self.older(n).unwrap_or(n)),
        }
    }

    fn older(&self, n: u32) -> Option<u32> {
        self.numbers.iter().rev().copied().find(|&k| k < n)
    }

    /// One line of state, for the hint.
    pub fn summary(&self) -> String {
        if let Some(p) = &self.problem {
            return p.clone();
        }
        match (self.current, self.good) {
            (None, _) => String::new(),
            (Some(c), Some(g)) if c == g => format!("gen {c} ✓ good"),
            (Some(c), Some(g)) => format!("gen {c} · good {g}"),
            (Some(c), None) => format!("gen {c} · none good yet"),
        }
    }

    /// What letting go would switch to.
    pub fn describe(&self, n: u32) -> String {
        let mut s = format!("→ gen {n}");
        if self.good == Some(n) {
            s.push_str(" ✓ good");
        }
        if self.current == Some(n) {
            s.push_str(" (current)");
        }
        s
    }
}

/// Generations on disk. One file per generation, named by its number, holding
/// exactly the bytes the node served — so any two compare with plain `diff`.
/// `current` and `good` are separate one-line files, so a generation's own
/// file is written once and never touched again.
pub struct Store {
    dir: PathBuf,
}

impl Store {
    /// `$TNWR_STATE`, else `$XDG_STATE_HOME/tnwr`, else `~/.local/state/tnwr`;
    /// then `generations/<node address>`.
    pub fn dir_for(node: &str) -> PathBuf {
        let base = std::env::var_os("TNWR_STATE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("XDG_STATE_HOME").map(|x| PathBuf::from(x).join("tnwr")))
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state/tnwr")))
            .unwrap_or_else(|| PathBuf::from(".tnwr-state"));
        base.join("generations").join(node.replace('/', "_"))
    }

    pub fn open(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Every generation, ascending, then the `current` and `good` marks.
    pub fn load(&self) -> io::Result<(Vec<(u32, String)>, Option<u32>, Option<u32>)> {
        let mut known = Vec::new();
        for n in self.numbers()? {
            known.push((n, fs::read_to_string(self.dir.join(n.to_string()))?));
        }
        known.sort_by_key(|(n, _)| *n);
        Ok((known, self.mark_of("current"), self.mark_of("good")))
    }

    /// Write `body` as the next number. It goes to a temporary name and is
    /// hard-linked into place, which fails rather than replaces if the number
    /// is taken — so a generation is never overwritten, even by a second core.
    pub fn record(&self, body: &str) -> io::Result<u32> {
        let tmp = self.dir.join(format!(".new-{}", std::process::id()));
        fs::write(&tmp, body)?;
        let mut n = self.numbers()?.into_iter().max().unwrap_or(0) + 1;
        let result = loop {
            match fs::hard_link(&tmp, self.dir.join(n.to_string())) {
                Ok(()) => break Ok(n),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => n += 1,
                Err(e) => break Err(e),
            }
        };
        let _ = fs::remove_file(&tmp);
        result
    }

    /// Point `current` or `good` at a generation. Written and renamed, so a
    /// crash leaves the old mark or the new one, never half of either.
    pub fn mark(&self, name: &str, n: u32) -> io::Result<()> {
        let tmp = self.dir.join(format!(".{name}-{}", std::process::id()));
        fs::write(&tmp, format!("{n}\n"))?;
        fs::rename(&tmp, self.dir.join(name))
    }

    fn mark_of(&self, name: &str) -> Option<u32> {
        fs::read_to_string(self.dir.join(name)).ok()?.trim().parse().ok()
    }

    fn numbers(&self) -> io::Result<Vec<u32>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let name = entry?.file_name();
            let Some(s) = name.to_str() else { continue };
            if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(n) = s.parse() {
                    out.push(n);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Duration = Duration::from_secs(30);

    fn at(t0: Instant, secs: u64) -> Instant {
        t0 + Duration::from_secs(secs)
    }

    #[test]
    fn a_config_is_recorded_only_when_it_changes() {
        let t0 = Instant::now();
        let mut g = Generations::new(vec![], None, None, N);
        assert_eq!(g.observe("/params/special\t1\n", t0), Seen::New);
        g.recorded(1, "/params/special\t1\n".into(), t0);
        assert_eq!(g.observe("/params/special\t1\n", t0), Seen::Same);
        assert_eq!(g.observe("/params/special\t0\n", t0), Seen::New);
    }

    #[test]
    fn going_back_to_an_old_config_by_hand_is_still_a_new_number() {
        // As in NixOS: numbers are history, not content.
        let t0 = Instant::now();
        let mut g = Generations::new(vec![(1, "a".into()), (2, "b".into())], Some(2), None, N);
        assert_eq!(g.observe("a", t0), Seen::New);
    }

    #[test]
    fn good_is_marked_by_input_that_works_n_seconds_in() {
        let t0 = Instant::now();
        let mut g = Generations::new(vec![(1, "a".into())], Some(1), None, N);
        assert_eq!(g.input_ok(at(t0, 40)), None, "not yet seen this run, so nothing has survived");
        g.observe("a", t0);
        assert_eq!(g.input_ok(at(t0, 29)), None);
        assert_eq!(g.input_ok(at(t0, 30)), Some(1));
        assert_eq!(g.input_ok(at(t0, 31)), None, "marked once");
    }

    #[test]
    fn needing_the_escape_chord_restarts_the_clock() {
        let t0 = Instant::now();
        let mut g = Generations::new(vec![(1, "a".into())], Some(1), None, N);
        g.observe("a", t0);
        g.escaped(at(t0, 25));
        assert_eq!(g.input_ok(at(t0, 40)), None);
        assert_eq!(g.input_ok(at(t0, 55)), Some(1));
    }

    #[test]
    fn a_switch_is_current_once_the_node_holds_it() {
        let t0 = Instant::now();
        let mut g = Generations::new(vec![(1, "a".into()), (2, "b".into())], Some(2), Some(1), N);
        g.switching(1);
        assert_eq!(g.observe("a", t0), Seen::Became(1));
        assert_eq!(g.current(), Some(1));
        assert_eq!(g.observe("a", t0), Seen::Same, "no new number for it");
    }

    #[test]
    fn a_switch_the_node_did_not_take_changes_nothing() {
        let t0 = Instant::now();
        let mut g = Generations::new(vec![(1, "a".into()), (2, "b".into())], Some(2), None, N);
        g.switching(1);
        assert_eq!(g.observe("b", t0), Seen::Same);
        assert_eq!(g.current(), Some(2));
    }

    #[test]
    fn space_goes_to_last_known_good_then_older() {
        let v = View { numbers: vec![1, 2, 3, 4, 5], current: Some(5), good: Some(3), problem: None };
        let mut picked = None;
        let mut seen = Vec::new();
        for _ in 0..5 {
            picked = v.step_back(picked);
            seen.push(picked.unwrap());
        }
        assert_eq!(seen, vec![3, 2, 1, 1, 1], "good, then older, stopping at the oldest");
    }

    #[test]
    fn with_nothing_good_yet_space_goes_one_older() {
        let v = View { numbers: vec![1, 2], current: Some(2), good: None, problem: None };
        assert_eq!(v.step_back(None), Some(1));
        let only = View { numbers: vec![1], current: Some(1), good: None, problem: None };
        assert_eq!(only.step_back(None), None, "nothing to go back to");
    }

    #[test]
    fn the_store_numbers_in_order_and_never_overwrites() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("tnwr-gen-test-{}-{nanos}", std::process::id()));
        let s = Store::open(dir.clone()).unwrap();
        assert_eq!(s.record("a").unwrap(), 1);
        assert_eq!(s.record("b").unwrap(), 2);
        s.mark("current", 2).unwrap();
        s.mark("good", 1).unwrap();

        let (known, current, good) = Store::open(dir.clone()).unwrap().load().unwrap();
        assert_eq!(known, vec![(1, "a".to_string()), (2, "b".to_string())]);
        assert_eq!((current, good), (Some(2), Some(1)));
        assert_eq!(fs::read_to_string(dir.join("1")).unwrap(), "a", "generation 1 untouched");
        fs::remove_dir_all(dir).unwrap();
    }
}
