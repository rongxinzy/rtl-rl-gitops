"""Close each admitted training iteration before replacing its job identity."""
import hashlib
import json
import time
import importlib.util
from pathlib import Path
from .core import HEX


def adapter_hash(path):
    if not path.is_dir():
        raise ValueError('Adapter directory missing')
    files = [path / name for name in ('adapter_config.json', 'adapter_model.safetensors')]
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError('Invalid adapter artifact')
    rows = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()



def comparison_outcome(before, after):
    regression = any(value == 'pass' and after[key] != 'pass' for key, value in before.items())
    delta = sum(v == 'pass' for v in after.values()) - sum(v == 'pass' for v in before.values())
    return 'regressed' if regression or delta < 0 else 'improved' if delta > 0 else 'unchanged'


def runtime_valid(result, config, frozen, previous=None):
    image = result.get('metadata', {}).get('training_image_id')
    return (isinstance(image, str) and image.startswith('sha256:') and HEX.fullmatch(image[7:]) is not None
            and (previous is None or image == previous) and image == config.get('image_id')
            and result.get('judge_runner_sha256') == frozen['judge_runner_sha256'])


def verify_comparison(executor, job, report):
    root = executor.root
    job_raw = (root / 'runs' / job['job_id'] / 'job-config.json').read_bytes()
    config = json.loads(job_raw)
    if report.get('job_config_sha256') != hashlib.sha256(job_raw).hexdigest():
        return False
    recipe = config['recipe_sha256']
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if report.get('recipe_sha256') != recipe_hash or report.get('recipe_file_sha256') != recipe:
        return False
    official = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    if report.get('model_revision') != official:
        return False
    spec = importlib.util.spec_from_file_location('iteration_eval_core', root / 'research/evaluation/core.py')
    core = importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    frozen = json.loads((root / 'research/evaluation/artifacts/frozen' / (report['freeze_id'] + '.json')).read_text())
    core.verify(frozen)
    expected = {task['task_id']: task for task in frozen['tasks']}
    labels = []
    runtime = None
    for label in ('baseline', 'candidate'):
        path = Path(report[label + '_result_path']).resolve()
        if not path.is_relative_to((root / 'research/evaluation/artifacts/runs').resolve()):
            return False
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report.get(label + '_evaluation_id'):
            return False
        result = json.loads(raw)
        if label == 'candidate' and (result.get('job_id') != job['job_id'] or result.get('adapter_sha256') != report['adapter_sha256']):
            return False
        if result['freeze_id'] != report['freeze_id'] or result['model_revision'] != official:
            return False
        image = result.get('metadata', {}).get('training_image_id')
        if not runtime_valid(result, config, frozen, runtime):
            return False
        runtime = image
        rows = result['results']
        if len(rows) != len(expected) or {row['task_id'] for row in rows} != set(expected):
            return False
        for row in rows:
            task = expected[row['task_id']]
            if any(row[key] != task[key] for key in ('spec_sha256', 'reference_sha256', 'testbench_sha256')):
                return False
            if row['report'].get('image') != task['image_id']:
                return False
            if row['classification'] not in ('pass', 'fail') or core.classification(row['report']) != row['classification']:
                return False
        actual = {kind: sum(row['classification'] == kind for row in rows) for kind in ('pass', 'fail', 'infrastructure', 'unknown')}
        if result.get('counts') != actual:
            return False
        labels.append({row['task_id']: row['classification'] for row in rows})
    outcome = comparison_outcome(labels[0], labels[1])
    return report['outcome'] == outcome


def outcomes(executor, job):
    folder = executor.root / 'research/evaluation/artifacts/comparisons'
    results = []
    for path in sorted(folder.glob('*.json')):
        try:
            report = json.loads(path.read_text())
            if report.get('job_id') != job['job_id']:
                continue
            if not isinstance(report.get('freeze_id'), str) or not HEX.fullmatch(report['freeze_id']):
                continue
            if job.get('evaluation_freeze_id') and report['freeze_id'] != job['evaluation_freeze_id']:
                continue
            if job['job_id'].startswith('brain-') and not job.get('evaluation_freeze_id'):
                continue
            if job.get('baseline_evaluation_id') and report.get('baseline_evaluation_id') != job['baseline_evaluation_id']:
                continue
            if report.get('comparison_complete') is not True or report.get('infrastructure_errors') != 0:
                continue
            if report.get('adapter_sha256') != adapter_hash(executor.root / 'runs' / job['job_id'] / 'adapter'):
                continue
            if not verify_comparison(executor, job, report):
                continue
            if report.get('outcome') not in ('improved', 'unchanged', 'regressed'):
                continue
            results.append({'job_id': job['job_id'], 'freeze_id': report['freeze_id'], 'adapter_sha256': report['adapter_sha256'], 'outcome': report['outcome'], 'comparison_id': hashlib.sha256(path.read_bytes()).hexdigest()})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return results


def gate(executor, current, params):
    valid = outcomes(executor, current)
    if not valid:
        raise ValueError('Current autonomous job requires post-training comparison')
    last = valid[-1]
    if last['outcome'] in ('unchanged', 'regressed'):
        if current.get('dataset_id') == params['dataset_id']:
            raise ValueError('No improvement requires a new validated dataset')
        evidence = executor.state / 'verified' / ('dataset-' + params['dataset_id'] + '.json')
        proposed = json.loads(evidence.read_text())
        if current.get('data_sha256') and current['data_sha256'] == proposed.get('data_sha256'):
            raise ValueError('No improvement requires changed training content')
        old_id, new_id = current.get('dataset_id'), params['dataset_id']
        if isinstance(old_id, str) and HEX.fullmatch(old_id) and HEX.fullmatch(new_id):
            from research.data_factory.cli import read_dataset
            base = executor.root / 'research/data_factory/state/datasets'
            before, after = read_dataset(base / old_id), read_dataset(base / new_id)
            def semantics(manifest):
                return {task['semantic_sha256'] for task in manifest['tasks'] if task['split'] == 'train'}
            if semantics(before) == semantics(after):
                raise ValueError('No improvement requires different training semantics')
    return last


def request_evaluation(executor):
    from .core import atomic
    from .evidence import evaluations
    job = json.loads((executor.root / 'scheduling/job.json').read_text())
    verified = outcomes(executor, job)
    if verified:
        return {**verified[-1], 'status': 'comparison_reused', 'comparison_complete': True}
    folder = executor.root / 'runs' / job['job_id']
    if folder.is_symlink() or folder.resolve().parent != (executor.root / 'runs').resolve():
        raise ValueError('Invalid current job path')
    config = json.loads((folder / 'job-config.json').read_text())
    exit_state = json.loads((folder / 'last-exit.json').read_text())
    trainer = json.loads((folder / 'trainer_state_final.json').read_text())
    if config.get('job_id') != job['job_id'] or exit_state.get('returncode') != 0 or exit_state.get('step', -1) < config['max_steps'] or trainer.get('global_step', -1) < config['max_steps']:
        raise ValueError('Current training has not completed')
    identity = adapter_hash(folder / 'adapter')
    baselines = [r for r in evaluations(executor) if r['baseline_complete'] and (not job.get('evaluation_freeze_id') or r['freeze_id'] == job['evaluation_freeze_id']) and (not job.get('baseline_evaluation_id') or r['evaluation_id'] == job['baseline_evaluation_id'])]
    if not baselines:
        raise ValueError('No complete baseline for pending evaluation')
    pending = {'job_id': job['job_id'], 'adapter_sha256': identity, 'freeze_id': baselines[-1]['freeze_id'], 'baseline_evaluation_id': baselines[-1]['evaluation_id'], 'requested_at': time.time()}
    path = folder / 'evaluation-pending.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if any(existing.get(key) != pending[key] for key in ('job_id', 'adapter_sha256', 'freeze_id', 'baseline_evaluation_id')):
            raise ValueError('Pending evaluation identity mismatch')
    else:
        atomic(path, pending)
    return {'status': 'evaluation_pending', 'job_id': job['job_id'], 'evaluation_pending': True, 'comparison_complete': False}
