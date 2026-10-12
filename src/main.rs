//! tnwr — M0 + M1, "The Fork".
//!
//! A winit window, a wgpu surface, a text grid drawn with glyphon, and a quad
//! pass underneath it for per-token backgrounds. The token data comes from the
//! tokeniser node over 9P: this process knows about paths, not tokenisers.
//!
//! Windowed backend first on purpose. The DRM/KMS one comes later behind the
//! same renderer, because debugging on a bare VT with no stack trace is how
//! this project dies.

mod generations;
mod input;
mod ninep;
mod quads;
mod vt;

use std::collections::VecDeque;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use quads::{Quad, QuadRenderer};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

/// Which node to attach to. `tnwr [addr]` — the core is not supposed to know
/// what kind of node is on the other end, so this is the only thing that
/// names one.
fn node_addr() -> String {
    std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5640".to_string())
}
const FONT_SIZE: f32 = 16.0;
const LINE_HEIGHT: f32 = 23.0;
const MARGIN: f32 = 18.0;

const TINT_A: [f32; 4] = [0.14, 0.17, 0.24, 1.0];
const TINT_B: [f32; 4] = [0.21, 0.26, 0.36, 1.0];
/// Divergent tokens. Warm against the cool tints so the eye finds them without
/// having to read the diff first.
const TINT_DIFF_A: [f32; 4] = [0.42, 0.24, 0.16, 1.0];
const TINT_DIFF_B: [f32; 4] = [0.52, 0.31, 0.20, 1.0];
const POLL: Duration = Duration::from_millis(200);
const FG: Color = Color::rgb(226, 231, 243);
const DIM: Color = Color::rgb(130, 141, 165);

/// A token: a byte range into the pane's text. The node's `out/spans` is
/// literally this, which is where the two halves of the project meet.
#[derive(Clone, Copy, Debug)]
struct Token {
    start: usize,
    end: usize,
}

/// One side of a fork: its own namespace path, its own text, its own tokens.
struct Pane {
    buffer: Buffer,
    head: Buffer,
    /// A node may declare `table <path> <col>...`; whitespace columns are
    /// aligned here and drawn below the body. Monospace makes this honest —
    /// the alignment the renderer computes is the alignment you see.
    table: Buffer,
    tokens: Vec<Token>,
    /// Half-open token index ranges that differ from the other side, straight
    /// from the node's `diverged` file. Not recomputed here.
    diverged: Vec<(usize, usize)>,
    /// Byte offset at which each line starts in the pane's text. cosmic-text
    /// reports glyph byte offsets relative to their own line, but the node's
    /// spans are absolute, so one has to be rebased onto the other.
    line_starts: Vec<usize>,
}

/// Read one pane by asking the node what to show.
///
/// The node serves a `view` file — `body`, optional `spans`, and any number of
/// `header` lines, each naming a path relative to the node. The core therefore
/// knows nothing about tokenisers or samplers; it renders what it is told to.
/// A node without a `view` renders nothing rather than being guessed at.
fn load_pane(
    c: &mut ninep::Client,
    path: &str,
    title: &str,
) -> std::io::Result<(String, Vec<Token>, String, String)> {
    let view = c.read_string(&format!("{path}/view"))?;

    let mut body = String::new();
    let mut tokens = Vec::new();
    let mut heads: Vec<String> = vec![title.to_string()];
    let mut table = String::new();

    for line in view.lines() {
        let mut it = line.split_whitespace();
        let (Some(role), Some(rel)) = (it.next(), it.next()) else { continue };
        let full = format!("{path}/{rel}");
        match role {
            "body" => {
                body = c
                    .read_string(&full)?
                    .trim_end_matches('\n')
                    .to_string();
            }
            "spans" => {
                if let Ok(sp) = c.read_string(&full) {
                    tokens = sp
                        .lines()
                        .filter_map(|l| {
                            let mut n = l.split_whitespace();
                            Some(Token {
                                start: n.next()?.parse().ok()?,
                                end: n.next()?.parse().ok()?,
                            })
                        })
                        .collect();
                    // NB: zero-width tokens are KEPT. They draw no quad, but
                    // dropping them would shift every later index and the
                    // node's `diverged` ranges would land one token late.
                }
            }
            "table" => {
                let cols: Vec<&str> = it.collect();
                if let Ok(raw) = c.read_string(&full) {
                    table = align_table(&raw, &cols);
                }
            }
            "header" => {
                let label = rel.rsplit('/').next().unwrap_or(rel);
                if let Ok(v) = c.read_string(&full) {
                    heads.push(format!("{label}={}", v.trim()));
                }
            }
            _ => {}
        }
    }
    Ok((body, tokens, heads.join("   "), table))
}

/// Align a table, selecting the columns the view asked for.
///
/// The file's first line names its own columns, so a view can ask for a subset
/// by name — the node stays the authority on what its data contains, and the
/// view decides what is worth the screen width. Columns are matched by name;
/// anything unrecognised is skipped rather than silently misaligned.
fn align_table(raw: &str, cols: &[&str]) -> String {
    if cols.is_empty() {
        return raw.to_string();
    }
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        return String::new();
    };
    let file_cols: Vec<&str> = header.split_whitespace().collect();
    let n_file = file_cols.len();

    // which file column each requested name lives in
    let idx: Vec<usize> = cols
        .iter()
        .filter_map(|want| file_cols.iter().position(|c| c == want))
        .collect();
    if idx.is_empty() {
        return String::new();
    }
    let cols: Vec<&str> = idx.iter().map(|&i| file_cols[i]).collect();

    let rows: Vec<Vec<String>> = lines
        .map(|l| {
            // splitn on the FILE's column count so the last file column keeps
            // its spaces — a token may legitimately be " the".
            let mut parts: Vec<String> = l
                .splitn(n_file, char::is_whitespace)
                .map(|x| x.to_string())
                .collect();
            while parts.len() < n_file {
                parts.push(String::new());
            }
            idx.iter().map(|&i| parts[i].clone()).collect()
        })
        .collect();
    let n = cols.len();

    let _ = n;
    let mut width: Vec<usize> = cols.iter().map(|c| c.len()).collect();
    for r in &rows {
        for (i, cell) in r.iter().enumerate() {
            width[i] = width[i].max(cell.chars().count());
        }
    }

    let mut out = String::new();
    for (i, c) in cols.iter().enumerate() {
        out.push_str(&format!("{:<w$} ", c, w = width[i]));
    }
    out.push('\n');
    for r in rows.iter().take(24) {
        for (i, cell) in r.iter().enumerate() {
            out.push_str(&format!("{:<w$} ", cell, w = width[i]));
        }
        out.push('\n');
    }
    out
}

/// Fetch `path/keys` and recurse into every child that has one. A child with
/// no `keys` file is a leaf. Depth-limited, because forks nest.
fn walk_keys(c: &mut ninep::Client, path: &str, tree: &mut input::KeyTree, depth: usize) {
    let Ok(text) = c.read_string(&format!("{path}/keys")) else {
        return;
    };
    let table = input::KeyTree::parse(&text);
    let children: Vec<String> = table.iter().map(|(_, n)| n.clone()).collect();
    tree.insert(path, table);
    if depth >= 6 {
        return;
    }
    for name in children {
        walk_keys(c, &format!("{path}/{name}"), tree, depth + 1);
    }
}

struct Gpu {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
    quads: QuadRenderer,
    panes: Vec<Pane>,
    status: Buffer,
    /// Addressing, fed by a queue. Receipt pushes onto `keyq` and does
    /// nothing else; `drain_input` runs once winit has delivered everything
    /// pending. Keystrokes resolve against `keytree`, a cache of every `keys`
    /// table, so the input path never talks to the node. See `input.rs`.
    addr: input::Addressing,
    keytree: input::KeyTree,
    keyq: VecDeque<input::Key>,
    hint: Buffer,
    /// Effects go to their own I/O worker. While one is out, `pending` holds
    /// the drain: keys behind it wait, in order, for its outcome.
    effects: mpsc::Sender<input::Effect>,
    pending: bool,
    /// Recovery layer 1. Checked at receipt, against physical keys, before any
    /// translation, addressing or node — so nothing can shadow it.
    chord: input::EscapeChord,
    /// Outcomes still owed by effects that were out when the chord fired.
    /// Each is dropped on arrival, so it cannot settle into the cleared state.
    /// A count, not a flag: two escapes can each leave an effect in flight.
    discard_outcomes: u32,
    /// Receipt changed what the hint says; shape it in about_to_wait, not there.
    hint_stale: bool,
    /// Recovery layer 2. A copy of the generations worker's state, so picking
    /// one while the chord is held never waits on anything.
    gens: generations::View,
    gen_tx: mpsc::Sender<GenMsg>,
    /// What space has picked while the chord is held.
    picked: Option<u32>,
    /// `addr.resolved` as last reported to the generations worker.
    resolved_sent: u64,
}

impl Gpu {
    async fn new(
        window: Arc<Window>,
        effects: mpsc::Sender<input::Effect>,
        gen_tx: mpsc::Sender<GenMsg>,
    ) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("no adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("no device");

        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface unsupported by adapter");
        let format = config.format;
        surface.configure(&device, &config);

        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut atlas = TextAtlas::new(&device, &queue, &cache, format);
        let text_renderer =
            TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);
        let quads = QuadRenderer::new(&device, format);

        // The window appears before the node has said anything. Panes and the
        // key tree arrive as a snapshot from the fetch worker; until then keys
        // queue rather than being read against a tree that has not loaded.
        let panes: Vec<Pane> = Vec::new();
        let status_text = format!("connecting to {}…", node_addr());
        let mut status = Buffer::new(&mut font_system, Metrics::new(FONT_SIZE, LINE_HEIGHT));
        status.set_text(
            &status_text,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );

        let mut hint = Buffer::new(&mut font_system, Metrics::new(FONT_SIZE, LINE_HEIGHT));
        hint.set_text("", &Attrs::new().family(Family::Monospace), Shaping::Advanced, None);

        let mut me = Self {
            window,
            device,
            queue,
            surface,
            config,
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            quads,
            panes,
            status,
            addr: input::Addressing::default(),
            keytree: input::KeyTree::default(),
            keyq: VecDeque::new(),
            hint,
            effects,
            pending: false,
            chord: input::EscapeChord::default(),
            discard_outcomes: 0,
            hint_stale: false,
            gens: generations::View::default(),
            gen_tx,
            picked: None,
            resolved_sent: 0,
        };
        me.render_hint();
        me.relayout();
        me
    }

    // ---------------------------------------------------------- addressing

    fn render_hint(&mut self) {
        let a = &self.addr;
        let here = if a.cursor.is_empty() { "/" } else { a.cursor.as_str() };
        let typed = if a.typed.is_empty() {
            "—".to_string()
        } else {
            a.typed.chars().map(|c| format!("{c} ")).collect()
        };
        let opts: Vec<String> = self
            .keytree
            .table(&a.cursor)
            .iter()
            .map(|(k, n)| format!("[{k}] {n}"))
            .collect();
        let body = if !self.keytree.loaded() {
            format!("{here}\n\n(waiting for the node…)")
        } else if opts.is_empty() {
            format!("{here}\n\n(leaf — enter toggles, esc resets)")
        } else {
            format!("{here}\n\n{}", opts.join("   "))
        };
        let text = match &a.editing {
            Some(buf) => format!(
                "editing {}\n\n> {buf}_\n\nenter writes   esc cancels",
                a.cursor
            ),
            None => format!("typed: {typed}   {}\n{body}\n{}", self.gens.summary(), a.flash),
        };
        self.hint.set_text(
            &text,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        self.hint
            .set_size(Some(self.config.width as f32 - MARGIN * 2.0), Some(200.0));
        self.hint.shape_until_scroll(&mut self.font_system, false);
        self.window.request_redraw();
    }

    /// The escape chord fired. Receipt-safe: no I/O and no shaping.
    ///
    /// Clearing `pending` immediately is the point. If an effect is out and the
    /// node has hung, waiting for its outcome would leave input gated forever —
    /// the escape chord leaving you stuck, which is the one thing it must never
    /// do. Its outcome is counted and dropped when it eventually arrives.
    fn escape(&mut self) {
        input::escape(&mut self.keyq, &mut self.addr);
        if self.pending {
            self.discard_outcomes += 1;
            self.pending = false;
        }
        self.picked = None;
        // Needing the chord is not surviving: the generation's clock restarts.
        let _ = self.gen_tx.send(GenMsg::Escaped(Instant::now()));
        self.hint_stale = true;
    }

    /// Space while the chord is held: one generation further back. Receipt-
    /// safe — the view is a local copy.
    fn pick_older(&mut self) {
        self.picked = self.gens.step_back(self.picked);
        self.addr.flash = match self.picked {
            Some(n) => format!("escape · t n w r   {}   let go to switch", self.gens.describe(n)),
            None => "escape · t n w r   no earlier generation".into(),
        };
        self.hint_stale = true;
    }

    /// The last chord key came up. If space picked a generation, switch to it;
    /// if not, the chord was only an escape.
    fn let_go(&mut self) {
        if let Some(n) = self.picked.take() {
            self.addr.flash = match self.gen_tx.send(GenMsg::Switch(n)) {
                Ok(()) => format!("switching to gen {n}…"),
                Err(_) => "generations worker gone".into(),
            };
            self.hint_stale = true;
        }
    }

    /// Tell the generations worker that input worked, if any has since the
    /// last report. One channel send; the worker decides what it proves.
    fn report_input(&mut self) {
        if self.addr.resolved != self.resolved_sent {
            self.resolved_sent = self.addr.resolved;
            let _ = self.gen_tx.send(GenMsg::InputOk(Instant::now()));
        }
    }

    /// Interpret queued keys up to the first effect, and stop there.
    ///
    /// Nothing is interpreted before the key tree has arrived, and nothing
    /// while an effect is out — those keys wait, in order, until its outcome
    /// settles. No I/O happens here: the effect goes to its worker.
    fn drain_input(&mut self) {
        if self.pending || !self.keytree.loaded() || self.keyq.is_empty() {
            return;
        }
        let (taken, effect) =
            input::drain_until_effect(&mut self.keyq, &mut self.addr, &self.keytree);
        // Opt-in, and in the drain rather than on receipt: counting what the
        // app actually interpreted is the only way to tell a dropped key from
        // one the window system never delivered.
        if taken > 0 && std::env::var_os("TNWR_INPUT_LOG").is_some() {
            eprintln!("input: drained {taken}");
        }
        if let Some(effect) = effect {
            if self.effects.send(effect).is_ok() {
                self.pending = true;
            } else {
                let path = self.addr.cursor.clone();
                self.addr.settle(input::Outcome::Failed {
                    path,
                    error: "i/o worker gone".into(),
                });
            }
        }
        self.report_input();
        self.render_hint();
    }

    /// Shape fetched pane data into buffers. Main thread only, because shaping
    /// needs the FontSystem — and it does no I/O, so it is quick.
    fn build(font_system: &mut FontSystem, data: Vec<PaneData>) -> Vec<Pane> {
        let attrs = Attrs::new().family(Family::Monospace);
        data.into_iter()
            .map(|d| {
                let mut buffer = Buffer::new(font_system, Metrics::new(FONT_SIZE, LINE_HEIGHT));
                buffer.set_text(&d.text, &attrs, Shaping::Advanced, None);
                let mut head = Buffer::new(font_system, Metrics::new(FONT_SIZE, LINE_HEIGHT));
                head.set_text(&d.detail, &attrs, Shaping::Advanced, None);
                let mut table = Buffer::new(font_system, Metrics::new(FONT_SIZE, LINE_HEIGHT));
                table.set_text(&d.table, &attrs, Shaping::Advanced, None);
                let mut line_starts = vec![0usize];
                for (i, b) in d.text.bytes().enumerate() {
                    if b == b'\n' {
                        line_starts.push(i + 1);
                    }
                }
                Pane {
                    buffer,
                    head,
                    table,
                    tokens: d.tokens,
                    diverged: d.diverged,
                    line_starts,
                }
            })
            .collect()
    }

    fn pane_width(&self) -> f32 {
        let n = self.panes.len().max(1) as f32;
        ((self.config.width as f32) - MARGIN * (n + 1.0)) / n
    }

    fn relayout(&mut self) {
        let w = self.pane_width();
        let h = self.config.height as f32;
        for pane in &mut self.panes {
            pane.buffer.set_size(Some(w), Some(h));
            pane.buffer.shape_until_scroll(&mut self.font_system, false);
            pane.head.set_size(Some(w), Some(h));
            pane.head.shape_until_scroll(&mut self.font_system, false);
            // No wrap width: a table that wraps is unreadable, so it is left
            // at full width and clipped to its pane by TextBounds instead.
            pane.table.set_size(None, Some(h));
            pane.table.shape_until_scroll(&mut self.font_system, false);
        }
        self.status
            .set_size(Some(self.config.width as f32 - MARGIN * 2.0), Some(h));
        self.status.shape_until_scroll(&mut self.font_system, false);
    }

    fn pane_origin(&self, i: usize) -> (f32, f32) {
        // Headers wrap when panes are narrow, so the body starts below the
        // TALLEST header across all panes — measured, and shared so the
        // bodies stay aligned with each other for comparison.
        let head_lines = self
            .panes
            .iter()
            .map(|p| p.head.layout_runs().count())
            .max()
            .unwrap_or(1)
            .max(1) as f32;
        (
            MARGIN + i as f32 * (self.pane_width() + MARGIN),
            MARGIN + (head_lines + 1.0) * LINE_HEIGHT,
        )
    }

    /// Rectangles for each token, taken from where the shaper actually put the
    /// glyphs rather than an assumed character width. Alternating tints make
    /// the cuts legible without drawing separators.
    fn token_quads(&self) -> Vec<Quad> {
        let mut out = Vec::new();
        for (pi, pane) in self.panes.iter().enumerate() {
            let (ox, oy) = self.pane_origin(pi);
            for (i, tok) in pane.tokens.iter().enumerate() {
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                let (mut top, mut height) = (0.0f32, 0.0f32);
                for run in pane.buffer.layout_runs() {
                    let base = pane.line_starts.get(run.line_i).copied().unwrap_or(0);
                    for g in run.glyphs {
                        let (gs, ge) = (base + g.start, base + g.end);
                        if gs >= tok.start && ge <= tok.end {
                            lo = lo.min(g.x);
                            hi = hi.max(g.x + g.w);
                            top = run.line_top;
                            height = run.line_height;
                        }
                    }
                }
                if hi > lo {
                    let odd = i % 2 == 1;
                    let differs = pane.diverged.iter().any(|(a, b)| i >= *a && i < *b);
                    let tint = match (differs, odd) {
                        (true, false) => TINT_DIFF_A,
                        (true, true) => TINT_DIFF_B,
                        (false, false) => TINT_A,
                        (false, true) => TINT_B,
                    };
                    out.push(Quad::new(ox + lo, oy + top, hi - lo, height, tint));
                }
            }
        }
        out
    }

    /// Apply a snapshot from the fetch worker: shape it, swap in the key tree,
    /// redraw. The fetching already happened on another thread.
    fn apply_snapshot(&mut self, snap: Snapshot) {
        self.panes = Self::build(&mut self.font_system, snap.panes);
        self.keytree = snap.keytree;
        self.status.set_text(
            &snap.status,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        self.relayout();
        self.render_hint();
        self.window.request_redraw();
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.relayout();
        self.window.request_redraw();
    }

    fn render(&mut self) {
        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.config.width,
                height: self.config.height,
            },
        );

        let quads = self.token_quads();
        self.quads.prepare(
            &self.device,
            &self.queue,
            (self.config.width, self.config.height),
            &quads,
        );

        let full = TextBounds {
            left: 0,
            top: 0,
            right: self.config.width as i32,
            bottom: self.config.height as i32,
        };
        let mut areas: Vec<TextArea> = Vec::new();
        // Layout flows rather than sitting at fixed fractions of the window:
        // a table's height depends on how many steps a node generated, so
        // anything positioned by percentage collides as soon as content grows.
        let mut content_bottom = 0.0f32;
        for (i, pane) in self.panes.iter().enumerate() {
            let (ox, oy) = self.pane_origin(i);
            let body_lines = pane.buffer.layout_runs().count().max(1) as f32;
            let table_top = oy + (body_lines + 1.0) * LINE_HEIGHT;
            let table_lines = pane.table.layout_runs().count() as f32;
            content_bottom = content_bottom.max(table_top + table_lines * LINE_HEIGHT);

            let col = TextBounds {
                left: ox as i32,
                top: 0,
                right: (ox + self.pane_width()) as i32,
                bottom: self.config.height as i32,
            };
            areas.push(TextArea {
                buffer: &pane.head,
                left: ox,
                top: MARGIN,
                scale: 1.0,
                bounds: col,
                default_color: DIM,
                custom_glyphs: &[],
            });
            areas.push(TextArea {
                buffer: &pane.buffer,
                left: ox,
                top: oy,
                scale: 1.0,
                bounds: col,
                default_color: FG,
                custom_glyphs: &[],
            });
            if table_lines > 0.0 {
                areas.push(TextArea {
                    buffer: &pane.table,
                    left: ox,
                    top: table_top,
                    scale: 1.0,
                    bounds: col,
                    default_color: DIM,
                    custom_glyphs: &[],
                });
            }
        }

        let hint_top = content_bottom + LINE_HEIGHT;
        let hint_lines = self.hint.layout_runs().count() as f32;
        areas.push(TextArea {
            buffer: &self.hint,
            left: MARGIN,
            top: hint_top,
            scale: 1.0,
            bounds: full,
            default_color: FG,
            custom_glyphs: &[],
        });
        areas.push(TextArea {
            buffer: &self.status,
            left: MARGIN,
            top: hint_top + (hint_lines + 1.0) * LINE_HEIGHT,
            scale: 1.0,
            bounds: full,
            default_color: DIM,
            custom_glyphs: &[],
        });

        self.text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                areas,
                &mut self.swash_cache,
            )
            .expect("prepare");

        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match self.surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => f,
            Cst::Timeout | Cst::Occluded => return,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tnwr"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.043,
                            g: 0.047,
                            b: 0.063,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.quads.render(&mut pass);
            self.text_renderer
                .render(&self.atlas, &self.viewport, &mut pass)
                .expect("render");
        }
        self.queue.submit(Some(encoder.finish()));
        // wgpu 30 presents through the queue rather than the texture.
        self.queue.present(frame);
        self.atlas.trim();
    }
}

/// Raw data for one pane, fetched off the main thread. No buffers: shaping
/// needs the FontSystem, which lives with the renderer.
struct PaneData {
    detail: String,
    text: String,
    tokens: Vec<Token>,
    table: String,
    diverged: Vec<(usize, usize)>,
}

/// Everything the window shows, gathered in one pass by the fetch worker.
struct Snapshot {
    panes: Vec<PaneData>,
    status: String,
    keytree: input::KeyTree,
}

/// What the workers send back to the event loop.
enum Wake {
    Snapshot(Snapshot),
    Outcome(input::Outcome),
    /// The generations worker's state changed.
    Generations(generations::View),
    /// A switch to a generation finished: its number, or why not.
    Switched(Result<u32, String>),
}

/// What the generations worker is told.
enum GenMsg {
    /// The fetch worker read the node's `/config`.
    Observed(String),
    /// Input reached something, at this moment.
    InputOk(Instant),
    /// The escape chord fired, at this moment.
    Escaped(Instant),
    /// Write this generation back to the node.
    Switch(u32),
}

/// Read everything the window shows. Runs on the fetch worker, never the main
/// thread — through a slow node this is well over a hundred round trips, and
/// measured at nine seconds of blocked input when it ran on the main thread.
fn fetch(client: &mut Option<ninep::Client>, first: bool, gens: &mpsc::Sender<GenMsg>) -> Snapshot {
    if client.is_none() {
        *client = ninep::Client::connect(&node_addr()).ok();
    }
    let Some(c) = client.as_mut() else {
        let msg = format!(
            "node unreachable at {}\nstart it: python3 nodes/tok_node.py",
            node_addr()
        );
        let mut keytree = input::KeyTree::default();
        keytree.insert("", Vec::new()); // offline still counts as loaded
        return Snapshot {
            panes: vec![PaneData {
                detail: "offline".into(),
                text: msg.clone(),
                tokens: Vec::new(),
                table: String::new(),
                diverged: Vec::new(),
            }],
            status: msg,
            keytree,
        };
    };

    // A fork is a directory. Made once — a refresh must not stamp on edits
    // made from outside, or the window stops being a view and becomes an
    // opinion. Diverge it from the keyboard: k b p <letter>.
    if first {
        let _ = c.mkdir("/fork/b");
    }

    // Every fork the node has, not a hardcoded pair.
    let mut targets: Vec<(String, String)> = vec![(String::new(), "/".into())];
    if let Ok(list) = c.read_string("/fork/keys") {
        for line in list.lines() {
            if let Some(name) = line.split_whitespace().nth(1) {
                targets.push((format!("/fork/{name}"), format!("/fork/{name}")));
            }
        }
    }

    let mut root_ok = true;
    let mut panes = Vec::new();
    for (i, (path, title)) in targets.iter().enumerate() {
        let mut pd = match load_pane(c, path, title) {
            Ok((text, tokens, detail, table)) => PaneData {
                detail,
                text,
                tokens,
                table,
                diverged: Vec::new(),
            },
            Err(e) => {
                if i == 0 {
                    root_ok = false;
                }
                PaneData {
                    detail: format!("{title}: {e}"),
                    text: String::new(),
                    tokens: Vec::new(),
                    table: String::new(),
                    diverged: Vec::new(),
                }
            }
        };
        // Each fork is tinted against its own comparison.
        if i > 0 {
            if let Ok(d) = c.read_string(&format!("{path}/diverged")) {
                for line in d.lines() {
                    let n: Vec<usize> =
                        line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                    if n.len() == 4 {
                        pd.diverged.push((n[2], n[3]));
                    }
                }
            }
        }
        panes.push(pd);
    }

    let mut status = String::new();
    for (path, title) in targets.iter().skip(1) {
        if let Ok(d) = c.read_string(&format!("{path}/diff")) {
            status.push_str(&format!("cat {title}/diff\n{}\n", d.trim_end()));
        }
    }

    let mut keytree = input::KeyTree::default();
    walk_keys(c, "", &mut keytree, 0);
    if !keytree.loaded() {
        keytree.insert("", Vec::new());
    }
    // The config, for generations. Read here because a fetch runs exactly when
    // something changed; deciding whether it is a new generation is the
    // generations worker's job.
    if root_ok {
        if let Ok(config) = c.read_string("/config") {
            let _ = gens.send(GenMsg::Observed(config));
        }
    }
    if !root_ok {
        *client = None; // reconnect next time rather than reuse a dead socket
    }
    Snapshot { panes, status, keytree }
}

/// Do one effect against the node. Runs on the effect worker.
fn perform(client: &mut Option<ninep::Client>, effect: input::Effect) -> input::Outcome {
    use input::{Effect, Outcome};
    if client.is_none() {
        *client = ninep::Client::connect(&node_addr()).ok();
    }
    let target = match &effect {
        Effect::Arrive(p) | Effect::Commit(p, _) => p.clone(),
    };
    let Some(c) = client.as_mut() else {
        return Outcome::Failed { path: target, error: "node unreachable".into() };
    };
    let outcome = match effect {
        Effect::Arrive(path) => match c.read_string(&path) {
            Ok(v) => {
                let v = v.trim().to_string();
                if v == "0" || v == "1" {
                    // A switch: arriving at it flips it.
                    let to = if v == "1" { "0" } else { "1" };
                    match c.write(&path, to.as_bytes()) {
                        Ok(()) => Outcome::Toggled { path, from: v, to: to.into() },
                        Err(e) => Outcome::Failed { path, error: e.to_string() },
                    }
                } else if c.writable(&path).unwrap_or(false) {
                    Outcome::Edit(v)
                } else {
                    Outcome::Shown { path, value: v }
                }
            }
            Err(e) => Outcome::Failed { path, error: e.to_string() },
        },
        Effect::Commit(path, value) => match c.write(&path, value.as_bytes()) {
            Ok(()) => Outcome::Wrote(path),
            Err(e) => Outcome::Failed { path, error: e.to_string() },
        },
    };
    if matches!(outcome, Outcome::Failed { .. }) {
        *client = None;
    }
    outcome
}

/// The fetch worker. Its channel carries only fetch requests, so any that
/// queue while one runs are all satisfied by the next.
fn spawn_fetcher(proxy: EventLoopProxy<Wake>, gens: mpsc::Sender<GenMsg>) -> mpsc::Sender<bool> {
    let (tx, rx) = mpsc::channel::<bool>();
    std::thread::spawn(move || {
        let mut client: Option<ninep::Client> = None;
        while let Ok(mut first) = rx.recv() {
            while let Ok(more) = rx.try_recv() {
                first |= more;
            }
            let snap = fetch(&mut client, first, &gens);
            if proxy.send_event(Wake::Snapshot(snap)).is_err() {
                return; // event loop gone
            }
        }
    });
    tx
}

/// The effect worker — separate from the fetcher on purpose. On one shared
/// thread a toggle would queue behind a background refresh, which through a
/// slow node is seconds. Effects run in order among themselves, and each one
/// that changes the node asks the fetcher for a fresh view.
fn spawn_effector(proxy: EventLoopProxy<Wake>, fetches: mpsc::Sender<bool>) -> mpsc::Sender<input::Effect> {
    let (tx, rx) = mpsc::channel::<input::Effect>();
    std::thread::spawn(move || {
        let mut client: Option<ninep::Client> = None;
        while let Ok(effect) = rx.recv() {
            let outcome = perform(&mut client, effect);
            let changed = outcome.changed_node();
            if proxy.send_event(Wake::Outcome(outcome)).is_err() {
                return;
            }
            if changed {
                let _ = fetches.send(false);
            }
        }
    });
    tx
}

/// Watch the node's content hashes and ask for a fresh view when they move.
/// A thread rather than a timer in `about_to_wait`: nothing that waits on a
/// socket belongs in the loop that also receives keys.
fn spawn_poll(fetches: mpsc::Sender<bool>) {
    std::thread::spawn(move || {
        let mut client: Option<ninep::Client> = None;
        let mut last = String::new();
        loop {
            std::thread::sleep(POLL);
            if client.is_none() {
                client = ninep::Client::connect(&node_addr()).ok();
            }
            let Some(c) = client.as_mut() else { continue };
            let Ok(root) = c.read_string("/out/hash") else {
                client = None; // node restarted or went away; reconnect next tick
                continue;
            };
            // Fold in every fork's hash, and the fork list itself, so creating
            // or deleting a fork also counts as a change.
            let forks = c.read_string("/fork/keys").unwrap_or_default();
            let mut now = format!("{root}{forks}");
            for line in forks.lines() {
                if let Some(name) = line.split_whitespace().nth(1) {
                    if let Ok(h) = c.read_string(&format!("/fork/{name}/out/hash")) {
                        now.push_str(&h);
                    }
                }
            }
            if now != last {
                if !last.is_empty() && fetches.send(false).is_err() {
                    return; // fetch worker gone
                }
                last = now;
            }
        }
    });
}

/// The generations worker: the store, the policy, and the node's `/config`.
/// Its own thread, because recording touches the disk and switching talks to
/// the node, and neither belongs anywhere near receipt.
fn spawn_generations(proxy: EventLoopProxy<Wake>) -> mpsc::Sender<GenMsg> {
    let (tx, rx) = mpsc::channel::<GenMsg>();
    std::thread::spawn(move || {
        let survive = std::env::var("TNWR_GOOD_AFTER")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(Duration::from_secs_f64)
            .unwrap_or(generations::SURVIVE);
        let opened = generations::Store::open(generations::Store::dir_for(&node_addr()))
            .and_then(|s| s.load().map(|loaded| (s, loaded)));
        let (store, mut gens, mut problem) = match opened {
            Ok((s, (known, current, good))) => (
                Some(s),
                generations::Generations::new(known, current, good, survive),
                None,
            ),
            Err(e) => (
                None,
                generations::Generations::new(Vec::new(), None, None, survive),
                Some(format!("generations off: {e}")),
            ),
        };
        let mut client: Option<ninep::Client> = None;
        let _ = proxy.send_event(Wake::Generations(gens.view(problem.clone())));

        while let Ok(msg) = rx.recv() {
            let Some(store) = store.as_ref() else { continue };
            let changed = match msg {
                GenMsg::Observed(body) => match observe(&mut gens, store, &body) {
                    Ok(changed) => changed,
                    Err(e) => {
                        problem = Some(format!("generations: {e}"));
                        true
                    }
                },
                GenMsg::InputOk(at) => match gens.input_ok(at) {
                    Some(n) => {
                        if let Err(e) = store.mark("good", n) {
                            problem = Some(format!("generations: {e}"));
                        }
                        true
                    }
                    None => false,
                },
                GenMsg::Escaped(at) => {
                    gens.escaped(at);
                    false
                }
                GenMsg::Switch(n) => {
                    let result = switch(&mut gens, store, &mut client, n);
                    if proxy.send_event(Wake::Switched(result)).is_err() {
                        return;
                    }
                    true
                }
            };
            if changed && proxy.send_event(Wake::Generations(gens.view(problem.clone()))).is_err() {
                return; // event loop gone
            }
        }
    });
    tx
}

/// Record what the node holds, if it is not the current generation. True if
/// anything the window shows changed.
fn observe(
    gens: &mut generations::Generations,
    store: &generations::Store,
    body: &str,
) -> std::io::Result<bool> {
    match gens.observe(body, Instant::now()) {
        generations::Seen::Same => Ok(false),
        generations::Seen::Became(n) => store.mark("current", n).map(|()| true),
        generations::Seen::New => {
            let n = store.record(body)?;
            gens.recorded(n, body.to_string(), Instant::now());
            store.mark("current", n).map(|()| true)
        }
    }
}

/// Write generation `n` back to the node, then read `/config` straight back.
/// It is current only if the node now holds exactly it.
fn switch(
    gens: &mut generations::Generations,
    store: &generations::Store,
    client: &mut Option<ninep::Client>,
    n: u32,
) -> Result<u32, String> {
    let body = gens.body(n).ok_or(format!("no generation {n}"))?.to_string();
    if client.is_none() {
        *client = ninep::Client::connect(&node_addr()).ok();
    }
    let c = client.as_mut().ok_or("node unreachable")?;
    let held = match c.write("/config", body.as_bytes()).and_then(|()| c.read_string("/config")) {
        Ok(held) => held,
        Err(e) => {
            *client = None;
            return Err(format!("gen {n}: {e}"));
        }
    };
    gens.switching(n);
    observe(gens, store, &held).map_err(|e| format!("gen {n}: {e}"))?;
    match gens.current() {
        Some(c) if c == n => Ok(n),
        other => Err(format!(
            "gen {n}: the node holds something else, recorded as gen {}",
            other.map_or("?".into(), |c| c.to_string())
        )),
    }
}

struct App {
    gpu: Option<Gpu>,
    proxy: EventLoopProxy<Wake>,
    polling: bool,
}

impl ApplicationHandler<Wake> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("tnwr / the fork")
            .with_inner_size(LogicalSize::new(1200.0, 620.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        // Workers before the window's first frame, so the window never waits
        // on the node to appear.
        let gens = spawn_generations(self.proxy.clone());
        let fetches = spawn_fetcher(self.proxy.clone(), gens.clone());
        let effects = spawn_effector(self.proxy.clone(), fetches.clone());
        self.gpu = Some(pollster::block_on(Gpu::new(window, effects, gens)));
        let _ = fetches.send(true); // the first fetch creates /fork/b
        if !self.polling {
            self.polling = true;
            spawn_poll(fetches);
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: Wake) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        match event {
            Wake::Snapshot(snap) => gpu.apply_snapshot(snap),
            Wake::Generations(view) => {
                gpu.gens = view;
                gpu.render_hint();
            }
            Wake::Switched(result) => {
                gpu.addr.flash = match result {
                    Ok(n) => format!("gen {n} is current · nothing was deleted"),
                    Err(e) => e,
                };
                gpu.render_hint();
            }
            Wake::Outcome(outcome) => {
                if gpu.discard_outcomes > 0 {
                    // Owed by an effect that was out when the chord fired. Its
                    // node-side change still shows: the effect worker asks for
                    // a fresh view regardless.
                    gpu.discard_outcomes -= 1;
                    return;
                }
                gpu.addr.settle(outcome);
                gpu.pending = false;
                gpu.report_input();
                // Keys that waited behind the effect are interpreted now, in
                // the state its outcome created.
                if gpu.keyq.is_empty() {
                    gpu.render_hint();
                } else {
                    gpu.drain_input();
                }
            }
        }
    }

    /// winit calls this after it has delivered every pending event, which is
    /// exactly when a drain should run: the queue holds everything typed so
    /// far, and a slow frame in between cannot have dropped any of it.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = self.gpu.as_mut() {
            if gpu.hint_stale {
                gpu.hint_stale = false;
                gpu.render_hint();
            }
            gpu.drain_input();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        logical_key,
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                // Receipt only: translate at the edge, queue, return. O(1).
                //
                // The escape chord is matched on physical keys, before any
                // keymap has a say. Its four keys go through the chord, which
                // withholds a press until it is known to be typing — see
                // input::EscapeChord.
                // While the chord is held, space picks a config generation.
                // Physical like the chord, and never typed.
                if physical_key == PhysicalKey::Code(KeyCode::Space) && gpu.chord.latched() {
                    if state == ElementState::Pressed && !repeat {
                        gpu.pick_older();
                    }
                    return;
                }
                let chord_key = match physical_key {
                    PhysicalKey::Code(KeyCode::KeyT) => Some(input::ChordKey::T),
                    PhysicalKey::Code(KeyCode::KeyN) => Some(input::ChordKey::N),
                    PhysicalKey::Code(KeyCode::KeyW) => Some(input::ChordKey::W),
                    PhysicalKey::Code(KeyCode::KeyR) => Some(input::ChordKey::R),
                    _ => None,
                };
                if state == ElementState::Released {
                    if let Some(k) = chord_key {
                        if gpu.chord.release(k, &mut gpu.keyq) {
                            gpu.let_go();
                        }
                    }
                    return;
                }
                let keys: Vec<input::Key> = match logical_key {
                    Key::Named(NamedKey::Escape) => vec![input::Key::Escape],
                    Key::Named(NamedKey::Enter) => vec![input::Key::Enter],
                    Key::Named(NamedKey::Backspace) => vec![input::Key::Backspace],
                    Key::Named(NamedKey::Space) => vec![input::Key::Char(' ')],
                    // every char, not just the first — composed input can
                    // arrive as more than one
                    Key::Character(sc) => sc.chars().map(input::Key::Char).collect(),
                    _ => Vec::new(),
                };
                match chord_key {
                    Some(k) => {
                        if gpu.chord.press(k, keys, repeat, &mut gpu.keyq) {
                            gpu.escape();
                        }
                    }
                    None if !keys.is_empty() => {
                        gpu.chord.before_other(&mut gpu.keyq);
                        gpu.keyq.extend(keys);
                    }
                    None => {}
                }
            }
            // Releases for keys held at the moment focus leaves never arrive.
            // A pick is only switched to by letting go; losing focus is not that.
            WindowEvent::Focused(false) => {
                gpu.picked = None;
                gpu.chord.focus_lost(&mut gpu.keyq);
            }
            WindowEvent::RedrawRequested => gpu.render(),
            _ => {}
        }
    }
}

fn main() {
    // Subcommands for the VT backend. Kept in front of the event loop so
    // enumeration never has to create a window.
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("vt") => {
            let card = args.get(2).cloned().unwrap_or("/dev/dri/card0".into());
            if let Err(e) = vt::enumerate(&card) {
                eprintln!("vt: {e}");
                std::process::exit(1);
            }
            return;
        }
        Some("vt-paint") => {
            let card = args.get(2).cloned().unwrap_or("/dev/dri/card0".into());
            let secs = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
            if let Err(e) = vt::paint(&card, secs) {
                eprintln!("vt: {e}");
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }

    let event_loop = EventLoop::<Wake>::with_user_event()
        .build()
        .expect("event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let proxy = event_loop.create_proxy();
    let mut app = App {
        gpu: None,
        proxy,
        polling: false,
    };
    event_loop.run_app(&mut app).expect("run");
}
