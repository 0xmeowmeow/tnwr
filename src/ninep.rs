//! A minimal 9P2000 client — the core's half of the protocol boundary.
//!
//! Deliberately small: enough to walk to a file, read it, and write it. The
//! node is a separate process, so the core learns nothing about tokenisers
//! here, only about paths. That is the whole point of the boundary being a
//! socket rather than a function call.

use std::io::{self, Read, Write};
use std::net::TcpStream;

const TVERSION: u8 = 100;
const TATTACH: u8 = 104;
const RERROR: u8 = 107;
const TWALK: u8 = 110;
const TOPEN: u8 = 112;
const TCREATE: u8 = 114;
const TREAD: u8 = 116;
const TWRITE: u8 = 118;
const TCLUNK: u8 = 120;
const TSTAT: u8 = 124;

const MSIZE: u32 = 65536;
const NOFID: u32 = 0xFFFF_FFFF;

pub struct Client {
    stream: TcpStream,
    tag: u16,
    next_fid: u32,
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u16).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

impl Client {
    pub fn connect(addr: &str) -> io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        let mut c = Self {
            stream,
            tag: 1,
            next_fid: 1,
        };

        let mut body = Vec::new();
        body.extend_from_slice(&MSIZE.to_le_bytes());
        put_str(&mut body, "9P2000");
        c.rpc(TVERSION, &body)?;

        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes()); // fid 0 = root
        body.extend_from_slice(&NOFID.to_le_bytes());
        put_str(&mut body, "tnwr");
        put_str(&mut body, "");
        c.rpc(TATTACH, &body)?;
        Ok(c)
    }

    fn rpc(&mut self, mtype: u8, body: &[u8]) -> io::Result<Vec<u8>> {
        let tag = self.tag;
        self.tag = self.tag.wrapping_add(1).max(1);

        let mut msg = Vec::with_capacity(7 + body.len());
        msg.extend_from_slice(&((7 + body.len()) as u32).to_le_bytes());
        msg.push(mtype);
        msg.extend_from_slice(&tag.to_le_bytes());
        msg.extend_from_slice(body);
        self.stream.write_all(&msg)?;

        let mut hdr = [0u8; 4];
        self.stream.read_exact(&mut hdr)?;
        let size = u32::from_le_bytes(hdr) as usize;
        let mut rest = vec![0u8; size - 4];
        self.stream.read_exact(&mut rest)?;

        if rest[0] == RERROR {
            let n = u16::from_le_bytes([rest[3], rest[4]]) as usize;
            let msg = String::from_utf8_lossy(&rest[5..5 + n]).into_owned();
            return Err(io::Error::new(io::ErrorKind::Other, msg));
        }
        Ok(rest[3..].to_vec())
    }

    fn walk(&mut self, path: &str) -> io::Result<u32> {
        let fid = {
            self.next_fid += 1;
            self.next_fid
        };
        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes()); // from root
        body.extend_from_slice(&fid.to_le_bytes());
        body.extend_from_slice(&(parts.len() as u16).to_le_bytes());
        for p in &parts {
            put_str(&mut body, p);
        }
        let r = self.rpc(TWALK, &body)?;
        // 9P answers a partial walk with fewer qids than names, and does NOT
        // set the new fid. Treating that as success is how a missing file
        // turns into a confusing "unknown fid" from the next operation.
        let got = u16::from_le_bytes([r[0], r[1]]) as usize;
        if got != parts.len() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no such path: {path}"),
            ));
        }
        Ok(fid)
    }

    fn clunk(&mut self, fid: u32) {
        let _ = self.rpc(TCLUNK, &fid.to_le_bytes());
    }

    pub fn read(&mut self, path: &str) -> io::Result<Vec<u8>> {
        let fid = self.walk(path)?;
        let mut body = Vec::new();
        body.extend_from_slice(&fid.to_le_bytes());
        body.push(0); // OREAD
        self.rpc(TOPEN, &body)?;

        let mut out = Vec::new();
        let mut offset: u64 = 0;
        loop {
            let mut body = Vec::new();
            body.extend_from_slice(&fid.to_le_bytes());
            body.extend_from_slice(&offset.to_le_bytes());
            body.extend_from_slice(&(MSIZE - 24).to_le_bytes());
            let r = self.rpc(TREAD, &body)?;
            let n = u32::from_le_bytes([r[0], r[1], r[2], r[3]]) as usize;
            if n == 0 {
                break;
            }
            out.extend_from_slice(&r[4..4 + n]);
            offset += n as u64;
        }
        self.clunk(fid);
        Ok(out)
    }

    pub fn read_string(&mut self, path: &str) -> io::Result<String> {
        Ok(String::from_utf8_lossy(&self.read(path)?).into_owned())
    }

    pub fn write(&mut self, path: &str, data: &[u8]) -> io::Result<()> {
        // One Twrite, always. A node applies each write whole — `/config` is
        // applied on write — so splitting one across messages would apply
        // halves. Refuse instead of doing that quietly.
        if data.len() > (MSIZE - 24) as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} bytes is more than one 9P write carries", data.len()),
            ));
        }
        let fid = self.walk(path)?;
        let mut body = Vec::new();
        body.extend_from_slice(&fid.to_le_bytes());
        body.push(1); // OWRITE
        self.rpc(TOPEN, &body)?;

        let mut body = Vec::new();
        body.extend_from_slice(&fid.to_le_bytes());
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&(data.len() as u32).to_le_bytes());
        body.extend_from_slice(data);
        self.rpc(TWRITE, &body)?;
        self.clunk(fid);
        Ok(())
    }

    /// Ask the node whether a path may be written. 9P carries this in the
    /// stat mode bits, so the renderer never has to keep its own list of which
    /// paths happen to be editable.
    pub fn writable(&mut self, path: &str) -> io::Result<bool> {
        let fid = self.walk(path)?;
        let r = self.rpc(TSTAT, &fid.to_le_bytes())?;
        self.clunk(fid);
        // stat[n]: n[2], then size[2] type[2] dev[4] qid[13] mode[4]
        if r.len() < 27 {
            return Ok(false);
        }
        let mode = u32::from_le_bytes([r[23], r[24], r[25], r[26]]);
        Ok(mode & 0o200 != 0)
    }

    /// Create a directory — this is how a fork is made.
    pub fn mkdir(&mut self, path: &str) -> io::Result<()> {
        let (parent, name) = path.rsplit_once('/').unwrap_or(("", path));
        let fid = self.walk(if parent.is_empty() { "/" } else { parent })?;
        let mut body = Vec::new();
        body.extend_from_slice(&fid.to_le_bytes());
        put_str(&mut body, name);
        body.extend_from_slice(&(0x8000_0000u32 | 0o755).to_le_bytes());
        body.push(0);
        self.rpc(TCREATE, &body)?;
        self.clunk(fid);
        Ok(())
    }
}
