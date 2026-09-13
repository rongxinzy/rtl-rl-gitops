"""Upload completed L20 scalar metrics from an online host, without model access."""
import argparse
import hashlib
import json
import math
from pathlib import Path

REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'

def collect(folder):
    identity_bytes = (folder / 'job.json').read_bytes()
    identity = json.loads(identity_bytes)
    binding = hashlib.sha256(identity_bytes).hexdigest()
    final = json.loads((folder / 'training_metrics.json').read_text())
    state = json.loads((folder / 'trainer_state_final.json').read_text())
    if identity.get('model_revision') != REVISION:
        raise ValueError('official model revision required')
    if any(x.get('job_sha256') != binding for x in (final, state)):
        raise ValueError('metric identity mismatch')
    if final.get('global_step') != identity['max_steps'] or state.get('global_step') != identity['max_steps']:
        raise ValueError('only completed experiments exported')
    rows = [json.loads(x) for x in (folder / 'metrics.jsonl').read_text().splitlines() if x.strip()]
    if [r.get('step') for r in rows] != list(range(1, identity['max_steps'] + 1)):
        raise ValueError('missing or duplicate step metrics')
    scalars = []
    for row in rows:
        values = {k: row[k] for k in ('loss', 'gradient_norm', 'peak_allocated_bytes')}
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
            raise ValueError('invalid scalar metric')
        if values['gradient_norm'] <= 0:
            raise ValueError('nonzero gradient required')
        scalars.append((row['step'], values))
    # Elapsed seconds resets on resume, so it is not presented as total GPU time.
    config = {k: identity[k] for k in ('model_revision', 'model_manifest_sha256',
              'dataset_id', 'max_steps', 'max_length', 'rank', 'learning_rate', 'seed')}
    config.update(stage='l20_qlora_sft', job_sha256=binding,
                  scope='bounded pipeline experiment; loss is not RTL capability evidence')
    return config, scalars

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    config, scalars = collect(args.run)
    if not args.upload:
        print(json.dumps({'validated': True, 'config': config, 'steps': len(scalars)}))
        return
    receipt = args.run / 'swanlab-export.json'
    if receipt.exists():
        raise ValueError('export already recorded; do not duplicate a cloud run')
    from training.tracking import start_run
    import swanlab
    run = start_run('l20-' + config['job_sha256'][:16], config)
    for step, values in scalars:
        swanlab.log(values, step=step)
    swanlab.finish()
    result = {'job_sha256': config['job_sha256'], 'steps': len(scalars),
              'sdk_finish_returned': True, 'remote_independently_verified': False}
    receipt.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
