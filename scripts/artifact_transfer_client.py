#!/usr/bin/env python3
"""Streaming TLS client for artifact inbox; sync is source -> inbox only."""
import argparse
import hashlib
import http.client
import json
import os
import ssl
from pathlib import Path
from urllib.parse import quote, urlsplit


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): checksum.update(block)
    return checksum.hexdigest()


class Client:
    def __init__(self, url, token_file, cert):
        url = urlsplit(url)
        if url.scheme != 'https' or url.path not in ('', '/'):
            raise ValueError('Use an https://host:port base URL')
        self.host, self.port = url.hostname, url.port or 443
        self.headers = {'Authorization': 'Bearer ' + token_file.read_text().strip()}
        self.context = ssl.create_default_context(cafile=str(cert))

    def connection(self):
        return http.client.HTTPSConnection(self.host, self.port, context=self.context, timeout=120)

    def request(self, method, path, body=None, headers=None):
        connection = self.connection()
        try:
            connection.request(method, path, body=body, headers={**self.headers, **(headers or {})})
            response = connection.getresponse()
            data = response.read()
            if response.status >= 400: raise RuntimeError(f'{response.status}: {data.decode()}')
            return json.loads(data)
        finally: connection.close()

    def upload(self, source, remote):
        before = source.stat()
        checksum = digest(source)
        path = '/files/' + quote(remote, safe='/')
        connection = self.connection()
        try:
            connection.request('HEAD', path, headers=self.headers)
            response = connection.getresponse()
            remote_checksum = response.getheader('X-Content-SHA256')
            status = response.status
            response.read()
        finally: connection.close()
        if status == 200:
            if remote_checksum == checksum:
                return dict(path=remote, status='skipped_identical', sha256=checksum)
            raise RuntimeError(f'Different remote file exists: {remote}; use a versioned path')
        if status != 404: raise RuntimeError(f'HEAD failed: {status}')
        with source.open('rb') as stream:
            result = self.request('PUT', path, stream, {'Content-Length':str(before.st_size), 'X-Content-SHA256':checksum})
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed during upload: {source}; retry after writer stops')
        return result

    def download(self, remote, destination):
        if destination.exists(): raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name+'.partial')
        if partial.exists(): raise FileExistsError(partial)
        connection = self.connection()
        try:
            connection.request('GET', '/files/'+quote(remote, safe='/'), headers=self.headers)
            response = connection.getresponse()
            if response.status != 200: raise RuntimeError(f'{response.status}: {response.read().decode()}')
            checksum = hashlib.sha256()
            with partial.open('xb') as stream:
                while block := response.read(1024*1024):
                    checksum.update(block); stream.write(block)
            if checksum.hexdigest() != response.getheader('X-Content-SHA256'):
                raise RuntimeError('Download SHA256 mismatch')
            os.replace(partial, destination)
            return dict(path=remote, sha256=checksum.hexdigest(), destination=str(destination))
        finally:
            connection.close()
            partial.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='https://127.0.0.1:40115')
    p.add_argument('--token-file',type=Path,required=True); p.add_argument('--cert',type=Path,required=True)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('health'); sub.add_parser('list')
    put=sub.add_parser('upload'); put.add_argument('source',type=Path); put.add_argument('remote')
    get=sub.add_parser('download'); get.add_argument('remote'); get.add_argument('destination',type=Path)
    sync=sub.add_parser('sync'); sync.add_argument('source',type=Path); sync.add_argument('remote_prefix')
    a=p.parse_args(); client=Client(a.url,a.token_file,a.cert)
    if a.command in ('health','list'):
        print(json.dumps(client.request('GET','/health' if a.command=='health' else '/files'),ensure_ascii=False,indent=2))
    elif a.command=='upload': print(json.dumps(client.upload(a.source,a.remote)))
    elif a.command=='download': print(json.dumps(client.download(a.remote,a.destination)))
    else:
        if not a.source.is_dir(): p.error('sync source must be a directory')
        # No deletes, symlinks or automatic checkpoint loading/extraction.
        for source in sorted(a.source.rglob('*')):
            relative=source.relative_to(a.source)
            if source.is_file() and not source.is_symlink() and not any(part.startswith('.') for part in relative.parts):
                print(json.dumps(client.upload(source,a.remote_prefix.rstrip('/')+'/'+relative.as_posix())),flush=True)


if __name__=='__main__': main()
