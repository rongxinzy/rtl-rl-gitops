"""Private-network HTTP entrypoint. Secrets and arbitrary subprocesses are absent from API."""
import hmac
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from .core import Executor

ROOT = os.environ.get('RTL_EXECUTOR_ROOT', '/root/rtl-rl')
STATE = os.environ.get('RTL_EXECUTOR_STATE', '/root/rtl-rl/research/executor/state')
TOKEN_FILE = os.environ.get('RTL_EXECUTOR_TOKEN_FILE', '/root/rtl-rl/secrets/executor-token')


class Handler(BaseHTTPRequestHandler):
    executor = None
    token = None
    def log_message(self, *args):
        pass
    def reply(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def authorized(self):
        return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + self.token)
    def do_GET(self):
        if not self.authorized():
            return self.reply(401, {'error': 'unauthorized'})
        if self.path == '/v1/catalog':
            return self.reply(200, self.executor.catalog())
        if self.path == '/health':
            return self.reply(200, {'status': 'ready'})
        if self.path.startswith('/v1/actions/'):
            self.executor.recover()
            ident = self.path.removeprefix('/v1/actions/')
            from .core import ID
            if ID.fullmatch(ident):
                path = self.executor.state / (ident + '.json')
                if path.exists():
                    return self.reply(200, json.loads(path.read_text()))
        self.reply(404, {'error': 'not_found'})
    def do_POST(self):
        if not self.authorized():
            return self.reply(401, {'error': 'unauthorized'})
        if self.path != '/v1/actions':
            return self.reply(404, {'error': 'not_found'})
        try:
            if self.headers.get('Transfer-Encoding'):
                raise ValueError()
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError()
            self.connection.settimeout(5)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError()
            self.executor.validate(body)
            if body.get('stage') == 'apply':
                from .core import digest
                record = self.executor.state / (body['action_id'] + '.json')
                if not record.exists() or body.get('plan_hash') != digest(self.executor.validate(body)):
                    raise ValueError()
                saved = json.loads(record.read_text())
                if saved['plan_hash'] != body['plan_hash']:
                    raise ValueError()
                if saved['state'] != 'staged':
                    return self.reply(200, saved)
                with (self.executor.state / 'executor.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return self.reply(409, {'action_id': body['action_id'], 'state': 'busy', 'retryable': True})
                # Worker is durable at-most-once once state becomes running. GET
                # polls the persisted result; process crash leaves explicit running.
                thread = threading.Thread(target=self.executor.submit, args=(body,), daemon=True)
                thread.start()
                return self.reply(202, {'action_id': body['action_id'], 'state': 'accepted'})
            self.reply(200, self.executor.submit(body))
        except (ValueError, TypeError):
            self.reply(400, {'error': 'invalid_request'})


def main():
    Handler.token = Path(TOKEN_FILE).read_text().strip()
    if len(Handler.token) < 32:
        raise SystemExit('Executor token must contain at least 32 characters')
    Handler.executor = Executor(ROOT, STATE)
    ThreadingHTTPServer((os.environ.get('RTL_EXECUTOR_BIND', '127.0.0.1'), 18765), Handler).serve_forever()

if __name__ == '__main__':
    main()
