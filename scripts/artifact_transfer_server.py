#!/usr/bin/env python3
"""Authenticated TLS artifact inbox. Stores bytes; never loads checkpoints."""
import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import re
import shutil
import ssl
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

CHUNK = 1024 * 1024


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            result.update(block)
    return result.hexdigest()


class Handler(BaseHTTPRequestHandler):
    server_version = 'ArtifactInbox/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(120)

    def log_message(self, fmt, *args):
        # Never log credentials or request query strings.
        print(json.dumps(dict(peer=self.client_address[0], method=self.command,
                              path=urlsplit(self.path).path)), flush=True)

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        peer = ipaddress.ip_address(self.client_address[0])
        if not any(peer in net for net in self.server.allowed):
            self.reply(403, dict(error='Peer not allowed')); return False
        expected = 'Bearer ' + self.server.token
        if not hmac.compare_digest(self.headers.get('Authorization', '').encode(), expected.encode()):
            self.reply(401, dict(error='Bearer authentication required')); return False
        return True

    def file_path(self):
        path = unquote(urlsplit(self.path).path)
        if not path.startswith('/files/'):
            raise ValueError('Expected /files/{weights|datasets}/relative/path')
        relative = path[len('/files/'):]
        parts = relative.split('/')
        if len(parts) < 2 or parts[0] not in ('weights', 'datasets'):
            raise ValueError('Use weights/ or datasets/ namespace')
        if any(not part or part in ('.', '..') or part.startswith('.') or
               any(c in part for c in ('\\', ':', '\x00', '\n', '\r')) for part in parts):
            raise ValueError('Invalid artifact path')
        candidate = self.server.root.joinpath(*parts)
        for parent in [candidate, *candidate.parents]:
            if parent == self.server.root: break
            if parent.is_symlink(): raise ValueError('Symlinks are not served')
        if not candidate.resolve().is_relative_to(self.server.root):
            raise ValueError('Path escapes artifact root')
        return candidate, str(PurePosixPath(*parts))

    def do_GET(self):
        if not self.authorized(): return
        route = urlsplit(self.path).path
        if route == '/health':
            return self.reply(200, dict(status='ok', free_bytes=shutil.disk_usage(self.server.root).free))
        if route == '/files':
            rows = []
            for namespace in ('weights', 'datasets'):
                for path in (self.server.root/namespace).rglob('*'):
                    if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.server.root):
                        rel = path.relative_to(self.server.root)
                        if not any(part.startswith('.') for part in rel.parts):
                            rows.append(dict(path=str(rel), size=path.stat().st_size))
            return self.reply(200, rows)
        self.download(False)

    def do_HEAD(self):
        if self.authorized(): self.download(True)

    def download(self, head):
        try:
            path, relative = self.file_path()
        except ValueError as error:
            return self.reply(400, dict(error=str(error)))
        if not path.is_file(): return self.reply(404, dict(error='Artifact not found'))
        sha = digest(path)
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(path.stat().st_size))
        self.send_header('X-Content-SHA256', sha)
        self.end_headers()
        if not head:
            with path.open('rb') as stream:
                shutil.copyfileobj(stream, self.wfile, CHUNK)

    def do_PUT(self):
        if not self.authorized(): return
        try:
            path, relative = self.file_path()
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('Use an explicit Content-Length')
            size = int(self.headers.get('Content-Length', '-1'))
            sha = self.headers.get('X-Content-SHA256', '').lower()
            if size < 0 or size > self.server.max_bytes:
                raise ValueError('Invalid or oversized Content-Length')
            if not re.fullmatch('[0-9a-f]{64}', sha):
                raise ValueError('X-Content-SHA256 is required')
        except ValueError as error:
            return self.reply(400, dict(error=str(error)))
        if not self.server.upload_slots.acquire(blocking=False):
            return self.reply(503, dict(error='Too many simultaneous uploads'))
        temporary = None
        try:
            if shutil.disk_usage(self.server.root).free < size + 2*1024**3:
                return self.reply(507, dict(error='Insufficient space; reserve 2 GiB'))
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix='.incoming-', dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                remaining, checksum = size, hashlib.sha256()
                while remaining:
                    block = self.rfile.read(min(CHUNK, remaining))
                    if not block: raise ValueError('Incomplete upload')
                    stream.write(block); checksum.update(block); remaining -= len(block)
                stream.flush(); os.fsync(stream.fileno())
            if checksum.hexdigest() != sha:
                return self.reply(422, dict(error='SHA256 mismatch'))
            with self.server.publish_lock:
                if path.exists():
                    if path.is_file() and path.stat().st_size == size and digest(path) == sha:
                        return self.reply(200, dict(path=relative, sha256=sha, status='already_present'))
                    return self.reply(409, dict(error='Different artifact exists; use a new versioned path'))
                os.replace(temporary, path)
                temporary = None
            self.reply(201, dict(path=relative, size=size, sha256=sha, status='stored'))
        except (OSError, ValueError) as error:
            self.reply(400, dict(error=str(error)))
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
            self.server.upload_slots.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--cert', type=Path, required=True); p.add_argument('--key', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=40115)
    p.add_argument('--allow', action='append', default=['127.0.0.0/8'])
    p.add_argument('--max-gib', type=int, default=100)
    args = p.parse_args()
    token = args.token_file.read_text().strip()
    if len(token) < 32: p.error('Use a random token of at least 32 characters')
    root = args.root.resolve()
    if not root.is_dir(): p.error('Artifact root must already exist on the intended disk')
    for namespace in ('weights', 'datasets'): (root/namespace).mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.root, server.token = root, token
    server.allowed = [ipaddress.ip_network(net) for net in args.allow]
    server.max_bytes = args.max_gib*1024**3
    server.upload_slots = threading.BoundedSemaphore(2)
    server.publish_lock = threading.Lock()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(args.cert, args.key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f'Artifact inbox listening on {args.host}:{args.port}; root={root}', flush=True)
    server.serve_forever()


if __name__ == '__main__': main()
