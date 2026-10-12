#!/usr/bin/env python3
"""9p — a tiny 9P2000 client, for poking a node from the shell.

    9p.py ls   /out
    9p.py cat  /out/ids
    9p.py echo /params/special 0
    9p.py mkdir /fork/b
    9p.py tree

Exists so a node can be verified before the core can talk to it, and so the
namespace stays honest: if these commands work, `mount -t 9p` will too.
"""

from __future__ import annotations

import socket
import struct
import sys

Tversion, Tattach, Twalk = 100, 104, 110
Topen, Tcreate, Tread, Twrite = 112, 114, 116, 118
Tclunk, Tremove, Tstat = 120, 122, 124
Rerror = 107

MSIZE = 65536


def _s(b: bytes) -> bytes:
    return struct.pack("<H", len(b)) + b


class Client:
    def __init__(self, host="127.0.0.1", port=5640):
        self.sock = socket.create_connection((host, port))
        self.tag = 1
        self.fid = 1
        self._rpc(Tversion, struct.pack("<I", MSIZE) + _s(b"9P2000"))
        self._rpc(Tattach, struct.pack("<II", 0, 0xFFFFFFFF) + _s(b"meow") + _s(b""))

    def _rpc(self, mtype: int, body: bytes) -> bytes:
        tag = self.tag
        self.tag += 1
        msg = struct.pack("<IBH", 7 + len(body), mtype, tag) + body
        self.sock.sendall(msg)
        hdr = self._recv(4)
        (size,) = struct.unpack("<I", hdr)
        rest = self._recv(size - 4)
        rtype = rest[0]
        if rtype == Rerror:
            (n,) = struct.unpack_from("<H", rest, 3)
            raise RuntimeError(rest[5:5 + n].decode())
        return rest[3:]

    def _recv(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise RuntimeError("connection closed")
            buf += c
        return buf

    def walk(self, path: str, newfid: int) -> int:
        parts = [p for p in path.strip("/").split("/") if p]
        body = struct.pack("<IIH", 0, newfid, len(parts))
        body += b"".join(_s(p.encode()) for p in parts)
        self._rpc(Twalk, body)
        return newfid

    def read(self, path: str) -> bytes:
        f = self.walk(path, self._next())
        self._rpc(Topen, struct.pack("<IB", f, 0))
        out, off = b"", 0
        while True:
            r = self._rpc(Tread, struct.pack("<IQI", f, off, MSIZE - 24))
            (n,) = struct.unpack_from("<I", r, 0)
            if n == 0:
                break
            out += r[4:4 + n]
            off += n
        self._rpc(Tclunk, struct.pack("<I", f))
        return out

    def write(self, path: str, data: bytes) -> None:
        f = self.walk(path, self._next())
        self._rpc(Topen, struct.pack("<IB", f, 1))
        self._rpc(Twrite, struct.pack("<IQI", f, 0, len(data)) + data)
        self._rpc(Tclunk, struct.pack("<I", f))

    def mkdir(self, path: str) -> None:
        parent, _, name = path.rstrip("/").rpartition("/")
        f = self.walk(parent or "/", self._next())
        self._rpc(Tcreate, struct.pack("<I", f) + _s(name.encode())
                  + struct.pack("<IB", 0x80000000 | 0o755, 0))
        self._rpc(Tclunk, struct.pack("<I", f))

    def ls(self, path: str) -> list[tuple[str, bool]]:
        raw = self.read(path)
        out, off = [], 0
        while off < len(raw):
            (sz,) = struct.unpack_from("<H", raw, off)
            qtype = raw[off + 2 + 4 + 2]          # qid.type
            p = off + 2 + 2 + 4 + 13 + 4 + 4 + 4 + 8
            (n,) = struct.unpack_from("<H", raw, p)
            name = raw[p + 2:p + 2 + n].decode()
            out.append((name, bool(qtype & 0x80)))
            off += sz + 2
        return out

    def _next(self) -> int:
        self.fid += 1
        return self.fid


def tree(c: Client, path: str = "/", indent: int = 0) -> None:
    for name, isdir in c.ls(path):
        full = (path.rstrip("/") + "/" + name)
        print("  " * indent + name + ("/" if isdir else ""))
        if isdir:
            tree(c, full, indent + 1)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    c = Client()
    if cmd == "ls":
        for name, isdir in c.ls(sys.argv[2] if len(sys.argv) > 2 else "/"):
            print(name + ("/" if isdir else ""))
    elif cmd == "cat":
        sys.stdout.write(c.read(sys.argv[2]).decode("utf-8", "replace"))
    elif cmd == "echo":
        c.write(sys.argv[2], " ".join(sys.argv[3:]).encode())
    elif cmd == "mkdir":
        c.mkdir(sys.argv[2])
    elif cmd == "tree":
        tree(c)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
