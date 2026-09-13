"""Fail-closed public-tree allowlist and credential checks; never print matched text."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

MAX_BYTES = 4 * 1024 * 1024
RULES = {
    'private_key': r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----',
    'credential_url': r'[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+:[^\s/@]+@',
    'bearer_literal': r'(?i)\bBearer\s+[A-Za-z0-9_./+=-]{16,}',
    'provider_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b',
    'jwt_literal': r'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{16,}\b',
    'signed_url': r'(?i)[?&](?:X-Amz-Signature|X-Amz-Credential|X-Goog-Signature|access_token|auth_token|Signature)=[A-Za-z0-9%_+/=-]{8,}',
    'wireguard_key': r'(?im)^\s*(?:PrivateKey|PresharedKey)\s*=\s*[A-Za-z0-9+/]{43}=',
    'unquoted_credential': r'(?im)^\s*(?:export\s+)?[A-Z0-9_]*(?:PASSWORD|PASSWD|API_KEY|ACCESS_TOKEN|CLIENT_SECRET|PRIVATE_KEY)\s*[:=]\s*([A-Za-z0-9+/_.@#!=-]{8,})\s*$',
    'secret_assignment': r'''(?ix)["']?(?:password|passwd|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|secret[_-]?key|dsn)["']?\s*[:=]\s*["']([^"'\n]{8,})["']''',
    'kubernetes_secret': r'''(?im)["']?kind["']?\s*:\s*["']?Secret(?:["'\s,}]|$)''',
}
COMPILED = {name: re.compile(pattern) for name, pattern in RULES.items()}

def safe_name(name):
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and '\n' not in name

def forbidden(name):
    p = PurePosixPath(name)
    return (any(x in ('reports', 'logs', 'coordination', 'secrets', '.ssh', '.git', '__pycache__', 'staging') for x in p.parts)
            or p.suffix.lower() in ('.log', '.pem', '.key', '.p12', '.pfx', '.sqlite', '.db', '.pyc')
            or p.name.startswith('.env') or p.name in ('kubeconfig', 'credentials', 'id_rsa', 'id_ed25519'))

def content_findings(name, raw):
    if len(raw) > MAX_BYTES:
        return [{'path': name, 'line': 0, 'rule': 'oversize'}]
    try:
        text = raw.decode('utf-8')
        if '\x00' in text: raise UnicodeError()
    except UnicodeError:
        return [{'path': name, 'line': 0, 'rule': 'binary'}]
    findings = []
    for rule, pattern in COMPILED.items():
        for match in pattern.finditer(text):
            # Only explicit, visibly non-secret placeholders are exempt.
            if rule in ('secret_assignment','unquoted_credential') and re.fullmatch(r'(?:\$\{[A-Z0-9_]+\}|<[^<>]+>|REPLACE_ME|CHANGEME)', match.group(1)):
                continue
            findings.append({'path': name, 'line': text.count('\n', 0, match.start()) + 1, 'rule': rule})
    return findings

def inspect(name, raw, allowed, mode='100644'):
    findings = []
    for rule, bad in [('invalid_path', not safe_name(name)), ('not_allowlisted', name not in allowed),
                      ('forbidden_artifact', forbidden(name)), ('nonregular_file', mode not in ('100644', '100755'))]:
        if bad: findings.append({'path': name, 'line': 0, 'rule': rule})
    return findings + content_findings(name, raw)

def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

def scan(root, allowed, history=False):
    findings = []; checked = 0
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if '.git' in path.relative_to(root).parts: continue
        if path.is_symlink():
            findings.append({'path': name, 'line': 0, 'rule': 'symlink'}); continue
        if path.is_file():
            checked += 1
            with path.open('rb') as handle: raw = handle.read(MAX_BYTES + 1)
            findings.extend(inspect(name, raw, allowed))
    if history:
        seen = set()
        for commit in git(root, 'rev-list', '--all').decode().splitlines():
            for record in git(root, 'ls-tree', '-rz', commit).split(b'\0'):
                if not record: continue
                header, filename = record.split(b'\t', 1)
                mode, kind, oid = header.decode().split()
                name = filename.decode('utf-8')
                if (name, oid, mode) in seen: continue
                seen.add((name, oid, mode)); checked += 1
                if kind != 'blob':
                    rows = [{'path': name, 'line': 0, 'rule': 'non_blob'}]
                else:
                    size = int(git(root, 'cat-file', '-s', oid))
                    raw = git(root, 'cat-file', 'blob', oid) if size <= MAX_BYTES else b'x' * (MAX_BYTES + 1)
                    rows = inspect(name, raw, allowed, mode)
                findings.extend([{**row, 'commit': commit} for row in rows])
    return {'ok': not findings, 'checked': checked, 'history': history, 'findings': findings}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--allowlist', required=True)
    p.add_argument('--history', action='store_true'); a = p.parse_args()
    try:
        allowed = json.loads(Path(a.allowlist).read_text())
        if not isinstance(allowed, list) or not allowed or any(not isinstance(n, str) or not safe_name(n) for n in allowed):
            raise ValueError('invalid allowlist')
        root = Path(a.root).resolve(strict=True)
        if not root.is_dir(): raise ValueError('invalid root')
        result = scan(root, set(allowed), a.history)
    except Exception as exc:
        result = {'ok': False, 'error_type': type(exc).__name__}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result['ok'] else 1)

if __name__ == '__main__': main()
