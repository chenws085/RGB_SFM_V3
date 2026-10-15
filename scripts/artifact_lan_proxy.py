#!/usr/bin/env python3
"""Windows-side TCP relay into WSL; TLS/auth remain end-to-end at the inbox."""
import argparse
import ipaddress
import select
import socket
import socketserver
import time


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        peer = ipaddress.ip_address(self.client_address[0])
        if not any(peer in network for network in self.server.allowed):
            return
        with socket.create_connection(self.server.target, timeout=10) as upstream:
            sockets = [self.request, upstream]
            for stream in sockets: stream.settimeout(120)
            last_activity = time.monotonic()
            while time.monotonic()-last_activity < 120:
                readable, _, _ = select.select(sockets, [], [], 10)
                for stream in readable:
                    data = stream.recv(1024*1024)
                    if not data: return
                    (upstream if stream is self.request else self.request).sendall(data)
                    last_activity = time.monotonic()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='0.0.0.0'); p.add_argument('--port',type=int,default=40115)
    p.add_argument('--target',required=True); p.add_argument('--target-port',type=int,default=40115)
    p.add_argument('--allow',action='append',default=['127.0.0.0/8'])
    a=p.parse_args()
    with Server((a.host,a.port),Relay) as server:
        server.target=(a.target,a.target_port)
        server.allowed=[ipaddress.ip_network(value) for value in a.allow]
        print(f'TLS relay {a.host}:{a.port} -> {a.target}:{a.target_port}',flush=True)
        server.serve_forever()


if __name__=='__main__': main()
