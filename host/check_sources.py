#!/usr/bin/env python3
"""Offline read-only integrity check; does not contact or change hosts."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source-manifest.json').read_text())
for item in manifest['files']:
 path=(root/item['file']).resolve()
 if not path.is_relative_to(root/'sources') or path.is_symlink():raise ValueError('unsafe source path')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('source hash mismatch: '+item['file'])
print(json.dumps({'ok':True,'source_files':len(manifest['files']),'host_mutations':0}))
