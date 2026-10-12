//! The VT backend — taking the display directly, no compositor underneath.
//!
//! Brief v2, locked: *"Direct DRM/KMS. Take the VT, become DRM master,
//! mode-set, render on the GPU."* This is the first half of that: open the
//! card, find a connected display, become master, set a mode, and put pixels
//! on it through a dumb buffer.
//!
//! Dumb buffers are CPU-mapped and slow, and that is deliberate for a first
//! step — it proves the VT takeover, the mode-set and the scanout without any
//! of wgpu's surface machinery in the way. When this works, the GPU path
//! replaces only the buffer, not the takeover.
//!
//! Two subcommands, because they have very different risk:
//!
//!   tnwr vt          enumerate. Read-only, safe from anywhere.
//!   tnwr vt-paint    become master and drive the display. Needs a free VT.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::io::{AsFd, BorrowedFd};

use drm::buffer::Buffer as _;
use drm::control::{connector, crtc, Device as ControlDevice};
use drm::Device as BasicDevice;

pub struct Card(File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
impl BasicDevice for Card {}
impl ControlDevice for Card {}

impl Card {
    pub fn open(path: &str) -> io::Result<Self> {
        Ok(Card(
            OpenOptions::new().read(true).write(true).open(path)?,
        ))
    }
}

/// What we found on a card: the connected output, its preferred mode, and the
/// crtc that can drive it.
struct Output {
    connector: connector::Handle,
    crtc: crtc::Handle,
    mode: drm::control::Mode,
    name: String,
}

fn find_output(card: &Card) -> Result<Output, String> {
    let res = card
        .resource_handles()
        .map_err(|e| format!("resource_handles: {e}"))?;

    for &conn in res.connectors() {
        let info = card
            .get_connector(conn, false)
            .map_err(|e| format!("get_connector: {e}"))?;
        if info.state() != connector::State::Connected {
            continue;
        }
        let Some(&mode) = info.modes().first() else {
            continue;
        };
        let name = format!("{:?}-{}", info.interface(), info.interface_id());

        // An encoder links a connector to the crtcs that can drive it.
        let enc = info
            .current_encoder()
            .or_else(|| info.encoders().first().copied());
        let crtc = enc
            .and_then(|e| card.get_encoder(e).ok())
            .and_then(|e| e.crtc())
            .or_else(|| res.crtcs().first().copied());

        if let Some(crtc) = crtc {
            return Ok(Output {
                connector: conn,
                crtc,
                mode,
                name,
            });
        }
    }
    Err("no connected output with a usable crtc".into())
}

/// Read-only survey. Safe to run from inside a compositor session.
pub fn enumerate(path: &str) -> Result<(), String> {
    let card = Card::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let res = card
        .resource_handles()
        .map_err(|e| format!("resource_handles: {e}"))?;

    println!("card          {path}");
    println!(
        "resources     {} connectors, {} crtcs, {} encoders",
        res.connectors().len(),
        res.crtcs().len(),
        res.encoders().len()
    );

    for &conn in res.connectors() {
        let Ok(info) = card.get_connector(conn, false) else {
            continue;
        };
        let state = match info.state() {
            connector::State::Connected => "connected",
            connector::State::Disconnected => "disconnected",
            _ => "unknown",
        };
        let modes = info.modes();
        print!(
            "  {:?}-{:<3} {:<13}",
            info.interface(),
            info.interface_id(),
            state
        );
        match modes.first() {
            Some(m) => {
                let (w, h) = m.size();
                println!(" preferred {}x{}@{}  ({} modes)", w, h, m.vrefresh(), modes.len())
            }
            None => println!(" no modes"),
        }
    }

    match find_output(&card) {
        Ok(o) => {
            let (w, h) = o.mode.size();
            println!("chosen        {} {}x{} on crtc {:?}", o.name, w, h, o.crtc);
        }
        Err(e) => println!("chosen        none — {e}"),
    }
    Ok(())
}

/// Take the display. Requires a free VT: whoever holds DRM master keeps it,
/// so this fails cleanly rather than fighting a running compositor.
pub fn paint(path: &str, seconds: u64) -> Result<(), String> {
    let card = Card::open(path).map_err(|e| format!("open {path}: {e}"))?;

    card.acquire_master_lock()
        .map_err(|e| format!("acquire DRM master: {e} — is a compositor holding this card?"))?;
    println!("master        acquired on {path}");

    let out = find_output(&card)?;
    let (w, h) = out.mode.size();
    println!("output        {} {}x{}", out.name, w, h);

    let mut db = card
        .create_dumb_buffer((w as u32, h as u32), drm::buffer::DrmFourcc::Xrgb8888, 32)
        .map_err(|e| format!("create_dumb_buffer: {e}"))?;
    let fb = card
        .add_framebuffer(&db, 24, 32)
        .map_err(|e| format!("add_framebuffer: {e}"))?;

    let pitch = db.pitch() as usize; // read before the mapping borrows it
    {
        let mut map = card
            .map_dumb_buffer(&mut db)
            .map_err(|e| format!("map_dumb_buffer: {e}"))?;
        let buf = map.as_mut();
        // The same background the windowed backend clears to, plus a ramp, so
        // it is obvious at a glance that this is tnwr and not a blank console.
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = y * pitch + x * 4;
                if i + 3 >= buf.len() {
                    continue;
                }
                let t = x as f32 / w as f32;
                buf[i] = (16.0 + t * 60.0) as u8; // B
                buf[i + 1] = (12.0 + t * 40.0) as u8; // G
                buf[i + 2] = (11.0 + t * 20.0) as u8; // R
                buf[i + 3] = 0;
            }
        }
        // a bright band across the middle, so a failed scanout is unmistakable
        for y in (h as usize / 2 - 6)..(h as usize / 2 + 6) {
            for x in 0..w as usize {
                let i = y * pitch + x * 4;
                if i + 3 < buf.len() {
                    buf[i] = 0xba;
                    buf[i + 1] = 0xf2;
                    buf[i + 2] = 0x5d;
                }
            }
        }
    }

    card.set_crtc(out.crtc, Some(fb), (0, 0), &[out.connector], Some(out.mode))
        .map_err(|e| format!("set_crtc: {e}"))?;
    println!("modeset       done — holding for {seconds}s");

    std::thread::sleep(std::time::Duration::from_secs(seconds));

    // Tidy up. The kernel restores the console when master is dropped, but
    // removing what we added first keeps the failure modes boring.
    let _ = card.destroy_framebuffer(fb);
    let _ = card.destroy_dumb_buffer(db);
    let _ = card.release_master_lock();
    println!("released      master dropped, console restored");
    Ok(())
}
