"""Bounded exact-host CONNECT proxy. Never decrypt or log telemetry traffic."""
import base64
import hmac
import ipaddress
import json
import os
from pathlib import Path
import select
import socket
import socketserver
import threading
import time


class Policy:
    def __init__(self, hosts, token):
        self.hosts = frozenset(hosts)
        if not self.hosts or any(not h or h != h.lower() or '*' in h or '/' in h or ':' in h for h in self.hosts):
            raise ValueError('invalid exact domain allowlist')
        if not token or len(token) < 32:
            raise ValueError('proxy authentication required')
        self.expected = ('Basic ' + base64.b64encode(('telemetry:' + token).encode()).decode()).encode()

    def authenticate(self, value):
        return hmac.compare_digest(value.encode(), self.expected)

    def destination(self, authority):
        if authority.count(':') != 1:
            raise ValueError('destination denied')
        host, port = authority.split(':')
        if host not in self.hosts or port != '443':
            raise ValueError('destination denied')
        return host

    def resolve(self, host):
        result = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        addresses = []
        for family, socktype, proto, _, addr in result:
            ip = ipaddress.ip_address(addr[0])
            if not ip.is_global or ip.is_multicast or ip.is_reserved:
                raise ValueError('nonpublic destination denied')
            item = family, socktype, proto, addr
            if item not in addresses:
                addresses.append(item)
        if not addresses:
            raise ValueError('destination unavailable')
        return addresses


def client_hello(sock, expected_host):
    """Validate cleartext TLS SNI before forwarding; TLS payload stays encrypted."""
    wire, handshake = bytearray(), bytearray()
    deadline = time.monotonic() + 10
    def take(n):
        result = bytearray()
        while len(result) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("TLS handshake deadline")
            sock.settimeout(remaining)
            piece = sock.recv(n - len(result))
            if not piece:
                raise ValueError('incomplete TLS handshake')
            result.extend(piece)
        return bytes(result)
    while len(handshake) < 4 or len(handshake) < 4 + int.from_bytes(handshake[1:4], 'big'):
        header = take(5)
        size = int.from_bytes(header[3:5], 'big')
        if header[0] != 22 or not 0 < size <= 18432 or len(wire) + size > 65536:
            raise ValueError('TLS ClientHello required')
        body = take(size); wire.extend(header + body); handshake.extend(body)
    if handshake[0] != 1:
        raise ValueError('TLS ClientHello required')
    data = bytes(handshake[4:4 + int.from_bytes(handshake[1:4], 'big')])
    pos = 34
    def field(n):
        nonlocal pos
        value = data[pos:pos+n]
        if len(value) != n:
            raise ValueError('invalid TLS ClientHello')
        pos += n
        return value
    field(int.from_bytes(field(1), 'big'))
    field(int.from_bytes(field(2), 'big'))
    field(int.from_bytes(field(1), 'big'))
    extensions = field(int.from_bytes(field(2), 'big'))
    offset, names = 0, []
    while offset < len(extensions):
        if offset + 4 > len(extensions):
            raise ValueError('invalid TLS extensions')
        kind = int.from_bytes(extensions[offset:offset+2], 'big')
        size = int.from_bytes(extensions[offset+2:offset+4], 'big'); offset += 4
        value = extensions[offset:offset+size]; offset += size
        if len(value) != size:
            raise ValueError('invalid TLS extensions')
        if kind == 0:
            if len(value) < 5 or int.from_bytes(value[:2], 'big') != len(value)-2:
                raise ValueError('invalid TLS server name')
            entry = 2
            while entry < len(value):
                if entry+3 > len(value):
                    raise ValueError('invalid TLS server name')
                typ = value[entry]; length = int.from_bytes(value[entry+1:entry+3], 'big'); entry += 3
                name = value[entry:entry+length]; entry += length
                if len(name) != length or typ != 0:
                    raise ValueError('invalid TLS server name')
                names.append(name.decode('ascii'))
    if names != [expected_host]:
        raise ValueError('TLS server name does not match CONNECT')
    return bytes(wire)


class Handler(socketserver.BaseRequestHandler):
    def reply(self, code):
        reason = {200:'Connection Established',400:'Bad Request',403:'Forbidden',407:'Proxy Authentication Required',502:'Bad Gateway',503:'Service Unavailable'}[code]
        self.request.sendall(f'HTTP/1.1 {code} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n'.encode())

    def handle(self):
        if not self.server.slots.acquire(blocking=False):
            self.reply(503); return
        upstream = None
        try:
            self.request.settimeout(10)
            raw = bytearray()
            header_deadline = time.monotonic() + 10
            while not raw.endswith(b'\r\n\r\n'):
                remaining = header_deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.request.settimeout(remaining)
                value = self.request.recv(1)
                if not value or len(raw) >= 8192:
                    self.reply(400); return
                raw.extend(value)
            lines = raw.decode('ascii').split('\r\n')
            parts = lines[0].split(' ')
            if parts == ['GET','/health','HTTP/1.1']:
                self.reply(200); return
            if len(parts) != 3 or parts[0] != 'CONNECT' or parts[2] != 'HTTP/1.1':
                self.reply(403); return
            headers = {}
            for line in lines[1:-2]:
                if ':' not in line or line[0].isspace():
                    self.reply(400); return
                name, value = line.split(':',1); name = name.lower()
                if name in headers:
                    self.reply(400); return
                headers[name] = value.strip()
            if not self.server.policy.authenticate(headers.get('proxy-authorization','')):
                self.reply(407); return
            try:
                host = self.server.policy.destination(parts[1])
                addresses = self.server.policy.resolve(host)
            except (ValueError, OSError):
                self.reply(403); return
            for family, socktype, proto, address in addresses:
                candidate = socket.socket(family, socktype, proto)
                candidate.settimeout(5)
                try:
                    candidate.connect(address); upstream = candidate; break
                except OSError:
                    candidate.close()
            if upstream is None:
                self.reply(502); return
            self.reply(200)
            upstream.sendall(client_hello(self.request, host))
            end = time.monotonic() + 900
            sockets = (self.request, upstream)
            for sock in sockets:
                sock.settimeout(30)
            while time.monotonic() < end:
                readable, _, _ = select.select(sockets, [], [], 60)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    target = upstream if source is self.request else self.request
                    target.sendall(data)
        except (OSError, UnicodeError, ValueError):
            pass  # No raw exceptions, headers, signed URLs or credentials in logs.
        finally:
            if upstream:
                upstream.close()
            self.server.slots.release()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, policy):
        self.policy = policy
        self.slots = threading.BoundedSemaphore(64)
        super().__init__(address, Handler)


def main():
    hosts = json.loads(Path(os.getenv('ALLOWLIST_FILE','/config/hosts.json')).read_text())
    token = Path(os.getenv('PROXY_TOKEN_FILE','/secrets/token')).read_text().strip()
    with Server(('0.0.0.0', 8443), Policy(hosts, token)) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
