"""Read producer artifacts and repeat validation; never trust HTTP validation claims."""
import hashlib
import fcntl
import importlib.util
import json
from .core import atomic, HEX

MODEL = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def datasets(executor):
    from research.data_factory.cli import read_dataset
    records = []
    for path in sorted((executor.root / 'research/data_factory/state/datasets').glob('*/manifest.json')):
        try:
            value = read_dataset(path.parent)
            ident = value['dataset_id']
            if not HEX.fullmatch(ident):
                continue
            records.append({'dataset_id': ident, 'validated': True, 'data': str((path.parent / 'rl_train.jsonl').relative_to(executor.root)), 'data_sha256': value['files']['rl_train.jsonl'], 'registry': str((path.parent / 'registry_train.jsonl').relative_to(executor.root)), 'registry_sha256': value['files']['registry_train.jsonl']})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records


def evaluations(executor):
    core_path = executor.root / 'research/evaluation/core.py'
    if not core_path.exists():
        return []
    spec = importlib.util.spec_from_file_location('verified_evaluation_core', core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    base = executor.root / 'research/evaluation/artifacts'
    for path in sorted((base / 'runs').glob('qwen-base-*/result.json')):
        try:
            raw = path.read_bytes()
            report = json.loads(raw)
            if report['kind'] != 'qwen_base_baseline' or report['model_revision'] != MODEL:
                continue
            freeze_id = report['freeze_id']
            if not HEX.fullmatch(freeze_id):
                continue
            frozen = json.loads((base / 'frozen' / (freeze_id + '.json')).read_text())
            module.verify(frozen)
            results = report['results']
            if {r['task_id'] for r in results} != {t['task_id'] for t in frozen['tasks']} or len(results) != len(frozen['tasks']):
                continue
            tasks = {t['task_id']: t for t in frozen['tasks']}
            for row in results:
                task = tasks[row['task_id']]
                if any(row[key] != task[key] for key in ('spec_sha256', 'reference_sha256', 'testbench_sha256')):
                    raise ValueError('Evaluation task binding mismatch')
                if row['classification'] != module.classification(row['report']):
                    raise ValueError('Evaluation report classification mismatch')
            passed = bool(results) and all(row['classification'] == 'pass' for row in results)
            complete = bool(results) and all(row['classification'] in ('pass', 'fail') for row in results)
            records.append({'evaluation_id': hashlib.sha256(raw).hexdigest(), 'baseline_complete': complete, 'passed': passed, 'freeze_id': freeze_id, 'model_revision': MODEL, 'scope': 'internal_3val_smoke'})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records


def _refresh(executor):
    folder = executor.state / 'verified'
    folder.mkdir(exist_ok=True)
    data, evaluation = datasets(executor), evaluations(executor)
    for kind, records in [('dataset', data), ('evaluation', evaluation)]:
        valid = {record[kind + '_id'] for record in records}
        for old in folder.glob(kind + '-*.json'):
            if old.stem.removeprefix(kind + '-') not in valid:
                old.unlink()
        for record in records:
            atomic(folder / (kind + '-' + record[kind + '_id'] + '.json'), record)
    return data


def refresh(executor):
    with (executor.state / 'evidence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _refresh(executor)


def verified_freeze(executor, ident):
    if not isinstance(ident, str) or not HEX.fullmatch(ident):
        return False
    spec = importlib.util.spec_from_file_location('verified_evaluation_freeze', executor.root / 'research/evaluation/core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = json.loads((executor.root / 'research/evaluation/artifacts/frozen' / (ident + '.json')).read_text())
    return module.verify(frozen) is True


def reusable_baseline(executor):
    latest = executor.root / 'research/evaluation/artifacts/latest.json'
    try:
        freeze_id = json.loads(latest.read_text())['freeze_id']
        if not verified_freeze(executor, freeze_id):
            return None
        records = [r for r in evaluations(executor) if r['freeze_id'] == freeze_id and r['baseline_complete'] is True]
        return {**records[-1], 'status': 'baseline_reused'} if records else None
    except (OSError, ValueError, KeyError, TypeError):
        return None
