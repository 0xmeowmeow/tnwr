"""sukkal9p — a minimal 9P2000 server.

The protocol boundary for tnwr nodes. Every node is a process serving a
namespace over this; the core, a shell with `cat`, and another machine all
reach a node the same way. That is the whole point of choosing 9P, so this
file is the piece most worth getting right.

Implements the nine messages M1 needs:

    Tversion Tattach Twalk Topen Tread Twrite Tclunk Tstat Tcreate

plus Tremove (fork deletion) and a stub Tauth that declines authentication.
No permissions, no groups, no Twstat. Everything is little-endian.

Serve a tree of Entry objects. A file's contents can be static bytes or a
callable evaluated at read time — that is how out/ids stays lazy.
"""

from __future__ import annotations

import hashlib
import pathlib
import socket
import struct
import sys
import threading
import time
from typing import Callable, Optional

# ------------------------------------------------------------------ protocol

Tversion, Rversion = 100, 101
Tauth, Rauth = 102, 103
Tattach, Rattach = 104, 105
Rerror = 107
Tflush, Rflush = 108, 109
Twalk, Rwalk = 110, 111
Topen, Ropen = 112, 113
Tcreate, Rcreate = 114, 115
Tread, Rread = 116, 117
Twrite, Rwrite = 118, 119
Tclunk, Rclunk = 120, 121
Tremove, Rremove = 122, 123
Tstat, Rstat = 124, 125

QTDIR, QTFILE = 0x80, 0x00
DMDIR = 0x80000000
NOFID = 0xFFFFFFFF

MSIZE = 65536


def _s(b: bytes) -> bytes:
    """9P string: len[2] followed by the bytes."""
    return struct.pack("<H", len(b)) + b


def _rs(buf: bytes, off: int) -> tuple[bytes, int]:
    (n,) = struct.unpack_from("<H", buf, off)
    off += 2
    return buf[off:off + n], off + n


class NineError(Exception):
    pass


# ------------------------------------------------------ content addressing

def shape_of(e: "Entry") -> str:
    """A node's structure, with names deliberately left out.

    Names are labels hanging off a node, not part of what it is — so two nodes
    that differ only in what they are called have the same shape, and renaming
    something never invalidates anything downstream. What counts is the tree:
    which children are directories, which are readable, which are writable.
    """
    def walk(n: "Entry") -> str:
        kind = "d" if n.is_dir else ("w" if n.writer else "r")
        kids = "".join(walk(c) for c in n.children.values() if c.name != "trace")
        return f"({kind}{kids})"
    return walk(e)


def identity(source: bytes, shape: str) -> str:
    """A module is named by the hash of its own structure.

    Editing produces a new hash, so it is a new module and nothing downstream
    breaks — the Unison rule, and the honest version of what "blockchain
    everywhere" was reaching for.
    """
    h = hashlib.sha256()
    h.update(shape.encode())
    h.update(b"\0")
    h.update(source)
    return h.hexdigest()


# ------------------------------------------------------------------ tracing

MSG_NAMES = {
    100: "version", 104: "attach", 110: "walk", 112: "open", 114: "create",
    116: "read", 118: "write", 120: "clunk", 122: "remove", 124: "stat",
    108: "flush", 102: "auth",
}


class Trace:
    """Every sukkal message, recorded at the protocol layer.

    Done here rather than per module, so the live execution graph is free for
    every node ever written -- including ones by people who never thought
    about tracing. Retrofitted per module it would never be universal.

    A ring buffer: tracing must never be the reason a node runs out of memory.
    """

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.rows: list[tuple] = []
        self.seq = 0
        self.t0 = time.monotonic()
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def record(self, mtype: int, path: str, nbytes: int, micros: int, err: str = "") -> None:
        name = MSG_NAMES.get(mtype, str(mtype))
        with self.lock:
            self.seq += 1
            self.counts[name] = self.counts.get(name, 0) + 1
            self.rows.append((
                self.seq,
                round((time.monotonic() - self.t0) * 1000, 1),
                name,
                path or "-",
                nbytes,
                micros,
                err or "-",
            ))
            if len(self.rows) > self.capacity:
                del self.rows[: len(self.rows) - self.capacity]

    def log(self) -> bytes:
        with self.lock:
            rows = list(self.rows)
        out = ["seq ms op path bytes us err"]
        out += [" ".join(str(c) for c in r) for r in rows[-120:]]
        return ("\n".join(out) + "\n").encode()

    def stats(self) -> bytes:
        with self.lock:
            counts = dict(self.counts)
            total = self.seq
            elapsed = time.monotonic() - self.t0
        out = ["op count"]
        out += [f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        out.append(f"TOTAL {total}")
        out.append(f"UPTIME_S {elapsed:.1f}")
        return ("\n".join(out) + "\n").encode()


# ------------------------------------------------------------------ the tree

class Entry:
    """A file or directory.

    `reader` is called with no arguments and returns bytes; `writer` is
    called with the written bytes. Either may be None. Directories carry
    `children` and optionally `on_create`, which is what makes fork/ work.
    """

    _next_path = [1]

    def __init__(self, name: str, is_dir: bool = False,
                 reader: Optional[Callable[[], bytes]] = None,
                 writer: Optional[Callable[[bytes], None]] = None,
                 on_create: Optional[Callable[[str], "Entry"]] = None,
                 key: str = "", config: bool = False):
        # `key` is the addressing letter. Assigned here, at creation, and
        # stored on the node. It is NEVER derived from sorted position --
        # derived letters move when a listing reorders and no muscle memory
        # can ever form. This is the load-bearing rule of the addressing model.
        self.key = key
        # A config leaf is what a generation captures: the core keeps numbered
        # copies of `/config` and can write one back. Input and output are not
        # config -- rolling back must never take away what was typed.
        self.config = config
        self.name = name
        self.is_dir = is_dir
        self.reader = reader
        self.writer = writer
        self.on_create = on_create
        # Called with the child's name after a child is removed. Set it
        # alongside `on_create` when the node keeps its own state per child.
        self.on_remove: Optional[Callable[[str], None]] = None
        self.children: dict[str, Entry] = {}
        self.parent: Optional[Entry] = None
        self.qid_path = Entry._next_path[0]
        Entry._next_path[0] += 1
        self.version = 0

    def path(self) -> str:
        """Absolute path of this entry, for trace lines."""
        parts, cur = [], self
        while cur is not None and cur.parent is not None:
            parts.append(cur.name)
            cur = cur.parent
        return "/" + "/".join(reversed(parts))

    def add(self, child: "Entry") -> "Entry":
        child.parent = self
        self.children[child.name] = child
        return child

    def remove(self, name: str) -> None:
        # A created child's state can live outside the tree -- a graph step is
        # also a key in `state.steps` -- and removing only the entry would
        # leave it running unseen. `on_remove` lets the node drop it too.
        if self.children.pop(name, None) is not None and self.on_remove is not None:
            self.on_remove(name)

    # -- 9P representation ------------------------------------------------

    def qid(self) -> bytes:
        t = QTDIR if self.is_dir else QTFILE
        return struct.pack("<BIQ", t, self.version, self.qid_path)

    def keys(self) -> bytes:
        """`<letter> <name>` per addressable child, one per line.

        The hint and the accelerator are the same character, so a renderer
        shows exactly what a user types. Children without a letter are
        reachable by path but not by keystroke, and are simply absent here.
        """
        return "".join(
            f"{c.key} {c.name}\n" for c in self.children.values() if c.key
        ).encode()

    def contents(self) -> bytes:
        if self.is_dir:
            return b"".join(c.stat() for c in self.children.values())
        if self.reader is None:
            return b""
        out = self.reader()
        return out.encode() if isinstance(out, str) else out

    def stat(self) -> bytes:
        # Writability is the node's knowledge, and 9P already has a field for
        # it. A renderer should ask rather than keep its own list of which
        # paths happen to be editable.
        if self.is_dir:
            mode = DMDIR | 0o555
        else:
            mode = 0o666 if self.writer is not None else 0o444
        length = 0 if self.is_dir else len(self.contents())
        name = self.name.encode()
        uid = b"meow"
        body = (
            struct.pack("<HI", 0, 0)          # type[2] dev[4]
            + self.qid()                       # qid[13]
            + struct.pack("<III", mode, 0, 0)  # mode atime mtime
            + struct.pack("<Q", length)        # length[8]
            + _s(name) + _s(uid) + _s(uid) + _s(uid)
        )
        return struct.pack("<H", len(body)) + body


# ------------------------------------------------------------------ config

def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def _unesc(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"n": "\n", "t": "\t"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _find(root: Entry, path: str) -> Optional[Entry]:
    cur = root
    for part in path.split("/"):
        if not part:
            continue
        if not cur.is_dir or part not in cur.children:
            return None
        cur = cur.children[part]
    return cur


def _value(e: Entry) -> str:
    v = e.contents().decode("utf-8", "replace")
    return v[:-1] if v.endswith("\n") else v


def config_of(root: Entry) -> bytes:
    """Everything a generation captures, one line each, sorted.

    `path<TAB>value` for every config leaf, and `path/` for every directory
    that was *created* -- a child of a directory with `on_create`, which is
    what a fork is. Sorted, so the same config is the same bytes whatever
    order it was built in, and a parent always comes before its children.
    Backslash, tab and newline are escaped in paths and values alike.
    """
    lines = []

    def walk(e: Entry) -> None:
        for c in e.children.values():
            if c.is_dir:
                if e.on_create is not None:
                    lines.append(_esc(c.path()) + "/")
                walk(c)
            elif c.config and c.writer is not None:
                lines.append(_esc(c.path()) + "\t" + _esc(_value(c)))

    walk(root)
    return "".join(line + "\n" for line in sorted(lines)).encode()


def apply_config(root: Entry, data: bytes) -> None:
    """Make the tree match a config: create, write, prune.

    The whole config is parsed before anything changes, so an unreadable one
    changes nothing. Then the directories it names are created, parents
    first; values that differ are written; and created directories it does
    not name are removed. Values only ever go to leaves flagged config -- a
    generation cannot reach into input or output. Lines naming something this
    node cannot hold are skipped and reported, after the rest is applied.
    """
    dirs, values = [], []
    for raw in data.decode("utf-8").split("\n"):
        if not raw:
            continue
        if "\t" in raw:
            p, v = raw.split("\t", 1)
            values.append((_unesc(p), _unesc(v)))
        elif raw.endswith("/"):
            dirs.append(_unesc(raw[:-1]))
        else:
            raise NineError(f"config: unreadable line {raw[:60]!r}")

    problems = []
    for d in sorted(dirs):
        parent_path, _, name = d.rpartition("/")
        parent = _find(root, parent_path)
        if parent is None or parent.on_create is None:
            problems.append(f"cannot create {d}")
        elif name not in parent.children:
            parent.on_create(name)

    for p, v in values:
        e = _find(root, p)
        if e is None or e.is_dir or not e.config or e.writer is None:
            problems.append(f"not a config leaf: {p}")
        elif _value(e) != v:
            e.writer(v.encode())
            e.version += 1

    keep = set(dirs)

    def prune(e: Entry) -> None:
        for c in list(e.children.values()):
            if not c.is_dir:
                continue
            if e.on_create is not None and c.path() not in keep:
                e.remove(c.name)
            else:
                prune(c)

    prune(root)
    if problems:
        raise NineError("config: " + "; ".join(problems[:4]))


# ------------------------------------------------------------------ server

class Server:
    def __init__(self, root: Entry, names: str = ""):
        self.root = root
        # Labels live in a sidecar `<module>.names`, deliberately NOT in the
        # source: identity is the hash of the module, so if names were in the
        # source then renaming a node would change what it is. Any number of
        # them, any script.
        self.names = names or self._read_names() or "ascii  node\n"
        self.trace = Trace()
        self._last_path = ""
        # Grafted on by the server, not by the node. This is the whole point
        # of doing it at the protocol layer: neither tok_node.py nor
        # sampler_node.py mentions tracing, and both have it.
        if "trace" not in root.children:
            t = root.add(Entry("trace", is_dir=True, key="t"))
            t.add(Entry("keys", reader=t.keys))
            t.add(Entry("log", reader=self.trace.log, key="l"))
            t.add(Entry("stats", reader=self.trace.stats, key="s"))

        # Identity is the hash; names are labels hanging off it. Grafted by
        # the server so a node gets it without asking, same as tracing.
        if "id" not in root.children:
            src = b""
            try:
                main = sys.modules.get("__main__")
                if main is not None and getattr(main, "__file__", None):
                    src = pathlib.Path(main.__file__).read_bytes()
            except OSError:
                pass
            root.add(Entry("id", key="h", reader=lambda: (
                identity(src, shape_of(root)) + "\n").encode()))
            root.add(Entry("shape", reader=lambda: (shape_of(root) + "\n").encode()))
            if "names" not in root.children:
                root.add(Entry("names", key="n", reader=lambda: self.names.encode()))

        # Config generations. The core keeps numbered copies of this file and
        # writes one back to roll back. A node takes part by flagging leaves
        # `config=True`, and never learns what a generation is.
        if "config" not in root.children:
            root.add(Entry("config", reader=lambda: config_of(root),
                           writer=lambda data: apply_config(root, data)))

    @staticmethod
    def _read_names() -> str:
        main = sys.modules.get("__main__")
        f = getattr(main, "__file__", None)
        if not f:
            return ""
        side = pathlib.Path(f).with_suffix(".names")
        try:
            return side.read_text()
        except OSError:
            return ""

    def serve(self, host: str = "127.0.0.1", port: int = 5640) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(8)
        print(f"sukkal9p: serving on {host}:{port}", flush=True)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self._session, args=(conn,), daemon=True).start()

    # -- one client -------------------------------------------------------

    def _session(self, conn: socket.socket) -> None:
        fids: dict[int, Entry] = {}
        # Reads are served from a snapshot taken at open. Directories needed
        # this already; files need it too, because a file whose contents are
        # computed can change between the chunks of a single read and the
        # offsets then no longer line up. /trace/log is the sharp case --
        # reading it appends to it -- but any live file has the same problem.
        dirbuf: dict[int, bytes] = {}
        try:
            while True:
                hdr = self._recv(conn, 4)
                if not hdr:
                    return
                (size,) = struct.unpack("<I", hdr)
                body = self._recv(conn, size - 4)
                if body is None:
                    return
                mtype, tag = struct.unpack_from("<BH", body, 0)
                started = time.monotonic()
                err = ""
                try:
                    reply = self._dispatch(mtype, body[3:], fids, dirbuf)
                except NineError as e:
                    reply = (Rerror, _s(str(e).encode()))
                    err = str(e)
                except Exception as e:  # never drop a client on a bug
                    reply = (Rerror, _s(f"{type(e).__name__}: {e}".encode()))
                    err = f"{type(e).__name__}"
                micros = int((time.monotonic() - started) * 1_000_000)
                touched = self._last_path
                # Reading the trace is itself sukkal traffic. Recording it
                # would make the file grow every time it is looked at, so
                # those reads are marked rather than dropped -- self-
                # observation should be visible, not silently filtered.
                self.trace.record(mtype, touched, len(reply[1]), micros, err)
                rtype, rbody = reply
                out = struct.pack("<IBH", 7 + len(rbody), rtype, tag) + rbody
                conn.sendall(out)
        finally:
            conn.close()

    @staticmethod
    def _recv(conn: socket.socket, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _dispatch(self, mtype, b, fids, dirbuf):
        # Cleared per message: version, attach, auth and flush name no path,
        # and carrying the previous one forward makes the trace read as though
        # they touched a file they never saw.
        self._last_path = ""
        if mtype == Tversion:
            msize, off = struct.unpack_from("<I", b, 0)[0], 4
            ver, off = _rs(b, off)
            if not ver.startswith(b"9P2000"):
                return Rerror, _s(b"unsupported version")
            return Rversion, struct.pack("<I", min(msize, MSIZE)) + _s(b"9P2000")

        if mtype == Tauth:
            raise NineError("no authentication required")

        if mtype == Tattach:
            fid, afid = struct.unpack_from("<II", b, 0)
            fids[fid] = self.root
            return Rattach, self.root.qid()

        if mtype == Twalk:
            fid, newfid, nwname = struct.unpack_from("<IIH", b, 0)
            off = 10
            names = []
            for _ in range(nwname):
                nm, off = _rs(b, off)
                names.append(nm.decode())
            cur = fids.get(fid)
            if cur is None:
                raise NineError("unknown fid")
            self._last_path = "/" + "/".join(names) if names else "/"
            qids = []
            for nm in names:
                if nm == "..":
                    cur = cur.parent or self.root
                elif nm == ".":
                    pass
                else:
                    if not cur.is_dir or nm not in cur.children:
                        if not qids:
                            raise NineError("no such file")
                        break
                    cur = cur.children[nm]
                qids.append(cur.qid())
            if len(qids) == nwname:
                fids[newfid] = cur
            return Rwalk, struct.pack("<H", len(qids)) + b"".join(qids)

        if mtype == Topen:
            fid, mode = struct.unpack_from("<IB", b, 0)
            e = fids.get(fid)
            self._last_path = e.path() if e else ""
            if e is None:
                raise NineError("unknown fid")
            dirbuf[fid] = e.contents()
            return Ropen, e.qid() + struct.pack("<I", MSIZE - 24)

        if mtype == Tcreate:
            fid, off = struct.unpack_from("<I", b, 0)[0], 4
            name, off = _rs(b, off)
            perm, mode = struct.unpack_from("<IB", b, off)
            e = fids.get(fid)
            if e is None or not e.is_dir:
                raise NineError("cannot create here")
            if e.on_create is None:
                raise NineError("directory does not accept creation")
            # Tcreate on a name that exists is an error in 9P. Silently
            # replacing it loses whatever the operator had diverged, which is
            # exactly the state a fork exists to hold.
            if name.decode() in e.children:
                raise NineError(f"already exists: {name.decode()}")
            child = e.on_create(name.decode())
            fids[fid] = child
            return Rcreate, child.qid() + struct.pack("<I", MSIZE - 24)

        if mtype == Tread:
            fid, offset, count = struct.unpack_from("<IQI", b, 0)
            e = fids.get(fid)
            self._last_path = e.path() if e else ""
            if e is None:
                raise NineError("unknown fid")
            data = dirbuf.get(fid)
            if data is None:
                data = e.contents()
            chunk = data[offset:offset + count]
            return Rread, struct.pack("<I", len(chunk)) + chunk

        if mtype == Twrite:
            fid, offset, count = struct.unpack_from("<IQI", b, 0)
            data = b[16:16 + count]
            e = fids.get(fid)
            self._last_path = e.path() if e else ""
            if e is None or e.writer is None:
                raise NineError("not writable")
            e.writer(data)
            e.version += 1
            return Rwrite, struct.pack("<I", count)

        if mtype == Tclunk:
            (fid,) = struct.unpack_from("<I", b, 0)
            fids.pop(fid, None)
            dirbuf.pop(fid, None)
            return Rclunk, b""

        if mtype == Tremove:
            (fid,) = struct.unpack_from("<I", b, 0)
            e = fids.pop(fid, None)
            dirbuf.pop(fid, None)
            if e is None or e.parent is None:
                raise NineError("cannot remove")
            e.parent.remove(e.name)
            return Rremove, b""

        if mtype == Tstat:
            (fid,) = struct.unpack_from("<I", b, 0)
            e = fids.get(fid)
            if e is None:
                raise NineError("unknown fid")
            st = e.stat()
            return Rstat, struct.pack("<H", len(st)) + st

        if mtype == Tflush:
            return Rflush, b""

        raise NineError(f"unhandled message type {mtype}")
