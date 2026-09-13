#!/usr/bin/env python3
"""Build an immutable worker-compatible recipe; never change an active job."""
import argparse,pathlib,shutil,json,hashlib

def build(target):
 here=pathlib.Path(__file__).resolve().parent;target=pathlib.Path(target)
 if target.exists():raise ValueError('recipe output already exists')
 target.mkdir(parents=True)
 for name in ('model.py','provenance.py','knowledge_data.py','generate_eval.py'):
  shutil.copyfile(here.parent/'l20'/name,target/name)
 for name in ('train.py','state.py'):shutil.copyfile(here/name,target/name)
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir())}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();print(json.dumps(build(args.output),sort_keys=True))
