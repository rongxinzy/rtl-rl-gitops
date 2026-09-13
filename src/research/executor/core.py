"""Small capability executor: durable intent, fixed commands, bounded output."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')
HEX = re.compile(r'^[0-9a-f]{64}$')
ALIASES = {'eval.freeze': 'evaluation.freeze', 'eval.baseline': 'evaluation.baseline', 'eval.candidate': 'evaluation.candidate', 'data.build': 'data.run', 'proposal.stage': 'training.propose'}
ACTIONS = set(ALIASES) | {'knowledge.build', 'knowledge.status', 'l20.status', 'l20.admit', 'l20.compare', 'data.teacher', 'inspect', 'data.propose', 'data.compose', 'data.plan', 'data.run', 'data.status', 'evaluation.status', 'evaluation.freeze', 'evaluation.baseline', 'evaluation.candidate', 'training.propose', 'training.admit'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def project(root, keys):
    """Only explicit fields leave the executor; never return stderr/env/prompts."""
    return {key: root[key] for key in keys if key in root and isinstance(root[key], (str, int, float, bool, type(None)))}


class Executor:
    def __init__(self, root, state, runner=None):
        self.root = Path(root).resolve()
        self.state = Path(state).resolve()
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.runner = runner or self.execute
        self.recover()

    def recover(self):
        with (self.state / 'executor.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            for path in self.state.glob('*.json'):
                try:
                    record = json.loads(path.read_text())
                    if record.get('state') != 'running':
                        continue
                    pid = record.get('worker_pid')
                    if pid:
                        try:
                            os.kill(pid, 0)
                            continue
                        except ProcessLookupError:
                            pass
                    record.update(state='interrupted', reason='worker_exited', evidence_verified=False)
                    atomic(path, record)
                except (OSError, ValueError, TypeError):
                    continue

    def validate(self, body):
        if set(body) - {'action_id', 'action', 'params', 'stage', 'plan_hash'}:
            raise ValueError('Unknown request field')
        if not isinstance(body.get('action_id'), str) or not ID.fullmatch(body['action_id']):
            raise ValueError('Invalid action ID')
        action, params = body.get('action'), body.get('params', {})
        if action not in ACTIONS or not isinstance(params, dict):
            raise ValueError('Unsupported action')
        action = ALIASES.get(action, action)
        allowed = {
            'knowledge.build': set(), 'knowledge.status': set(), 'l20.status': set(), 'l20.compare': {'job_id','run_uid'}, 'data.teacher': set(), 'l20.admit': {'dataset_id','evaluation_id','max_steps'}, 'inspect': set(), 'data.propose': {'proposal'}, 'data.compose': {'dataset_ids'}, 'data.plan': {'limit'}, 'data.run': {'plan_id'}, 'data.status': set(),
            'evaluation.status': set(), 'evaluation.freeze': set(), 'evaluation.baseline': set(), 'evaluation.candidate': set(),
            'training.propose': {'dataset_id', 'evaluation_id', 'max_steps'},
            'training.admit': {'dataset_id', 'evaluation_id', 'max_steps'},
        }[action]
        if action == 'data.compose':
            ids = params.get('dataset_ids')
            if set(params) != {'dataset_ids'} or not isinstance(ids, list) or not 2 <= len(ids) <= 16 or any(not isinstance(x, str) or not HEX.fullmatch(x) for x in ids) or len(set(ids)) != len(ids):
                raise ValueError('Expected 2 to 16 distinct dataset IDs')
        if action == 'data.propose':
            if set(params) != {'proposal'}:
                raise ValueError('Exactly proposal parameter required')
            from research.data_factory.dsl import normalize
            params = {'proposal': normalize(params.get('proposal'))}
        if set(params) - allowed:
            raise ValueError('Unexpected parameter')
        if action == 'l20.compare' and params:
            if set(params) != {'job_id','run_uid'} or not re.fullmatch(r'l20-[0-9a-f]{24}',params.get('job_id','')) or not re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',params.get('run_uid','')):
                raise ValueError('Fixed Tekton job/run identity required')
        for key in ('plan_id', 'dataset_id', 'evaluation_id', 'candidate_id'):
            if key in params and (not isinstance(params[key], str) or not HEX.fullmatch(params[key])):
                raise ValueError('Invalid artifact ID')
        for key, default, minimum, maximum in [('limit', 4, 1, 8), ('max_steps', 100, 1, 1000)]:
            if key == 'max_steps' and action == 'l20.admit':
                maximum = 20
                default = 20
            if key == 'max_steps' and action == 'training.admit':
                maximum = 200
            if key in allowed:
                value = params.get(key, default)
                if type(value) is not int or not minimum <= value <= maximum:
                    raise ValueError('Out of bounds parameter')
        required = {'l20.admit': {'dataset_id', 'evaluation_id'}, 'data.run': {'plan_id'}, 'training.propose': {'dataset_id', 'evaluation_id'}, 'training.admit': {'dataset_id', 'evaluation_id'}}.get(action, set())
        if not required <= params.keys():
            raise ValueError('Missing parameter')
        if body.get('stage', 'stage') not in ('stage', 'apply'):
            raise ValueError('Invalid stage')
        return {'action': action, 'params': params}

    def submit(self, body):
        intent = self.validate(body)
        action_id = body['action_id']
        filename = self.state / (action_id + '.json')
        intent_hash = digest(intent)
        with (self.state / 'executor.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {'action_id': action_id, 'state': 'busy'}
            old = json.loads(filename.read_text()) if filename.exists() else None
            if old and old['plan_hash'] != intent_hash:
                raise ValueError('Action ID already belongs to another intent')
            if old and old['state'] != 'staged':
                # An interrupted action is never replayed automatically: caller inspects state.
                return old
            if body.get('stage', 'stage') == 'stage':
                value = old or {'action_id': action_id, 'plan_hash': intent_hash, 'state': 'staged', **intent}
                atomic(filename, value)
                return value
            if not old or body.get('plan_hash') != intent_hash:
                raise ValueError('Apply requires the exact staged plan hash')
            self.lock_fd = lock.fileno()
            value = {**old, 'state': 'running', 'started_at': time.time(), 'worker_pid': os.getpid()}
            atomic(filename, value)
            try:
                result = self.runner(intent['action'], intent['params'])
                verified = intent['action'] in ('data.run', 'data.compose', 'knowledge.build') and result.get('validated') is True
                if intent['action'] == 'evaluation.baseline':
                    verified = result.get('baseline_complete') is True
                if intent['action'] == 'evaluation.candidate':
                    verified = result.get('comparison_complete') is True
                if intent['action'] == 'l20.compare':
                    verified = result.get('comparison_complete') is True
                if intent['action'] == 'evaluation.freeze':
                    verified = result.get('freeze_verified') is True
                value.update(state='completed', result=result, evidence_verified=verified)
            except subprocess.TimeoutExpired:
                value.update(state='failed', reason='timeout')
            except Exception as exc:
                value.update(state='failed', reason=type(exc).__name__)
            value['finished_at'] = time.time()
            atomic(filename, value)
            return value

    def job_info(self):
        try:
            job = json.loads((self.root / 'scheduling/job.json').read_text())
            from .iteration import outcomes
            verified = outcomes(self, job)
            complete = (self.root / 'runs' / job['job_id'] / 'job-complete.json').is_file()
            return {**project(job, ['job_id', 'max_steps', 'dataset_id', 'evaluation_freeze_id']), 'complete': complete, 'comparison_required': complete and not verified, 'evaluation_pending': (self.root / 'runs' / job['job_id'] / 'evaluation-pending.json').exists() and not verified, 'last_outcome': verified[-1] if verified else None, 'admission_hint': 'wait_for_training' if not complete else ('evaluate_current_job' if not verified else ('new_dataset_required' if verified and verified[-1]['outcome'] in ('unchanged', 'regressed') else 'ready_for_candidate'))}
        except (OSError, ValueError, KeyError):
            return {'available': False}

    def catalog(self):
        from .evidence import refresh
        refresh(self)
        artifacts = {}
        from research.data_factory.cli import status as factory_status
        factory = factory_status()
        from .l20_branch import status as l20_status, knowledge_status
        l20 = l20_status(self)
        artifacts['plan_ids'] = factory.get('available_plans', [])
        for kind in ('dataset', 'evaluation'):
            artifacts[kind + '_ids'] = [p.stem.removeprefix(kind + '-') for p in sorted((self.state / 'verified').glob(kind + '-*.json'))]
        return {'knowledge':knowledge_status(),'l20':l20,'factory_remaining':factory.get('remaining_tasks'), 'actions': sorted(ACTIONS), 'artifacts': artifacts, 'current_job': self.job_info(), 'dataset_summary': [project(d, ['dataset_id', 'train', 'val', 'kind', 'validated']) for d in factory.get('available_datasets', [])], 'proposal_schema': json.loads((self.root / 'research/data_factory/proposal.schema.json').read_text()), 'constraints': {'knowledge.build': {}, 'knowledge.status': {}, 'l20.status': {}, 'l20.compare': {}, 'data.teacher': {}, 'l20.admit': {'dataset_id': '64 lowercase hex', 'evaluation_id': '64 lowercase hex', 'max_steps': [1, 20]}, 'inspect': {}, 'eval.freeze': {}, 'eval.baseline': {}, 'eval.candidate': {}, 'data.compose': {'dataset_ids': '2 to 16 distinct verified IDs'}, 'data.propose': {'proposal': 'proposal_schema; no code or paths'}, 'data.plan': {'limit': [1, 8]}, 'data.build': {'plan_id': '64 lowercase hex'}, 'proposal.stage': {'dataset_id': '64 lowercase hex', 'evaluation_id': '64 lowercase hex', 'max_steps': [1, 1000]}, 'training.admit': {'dataset_id': '64 lowercase hex', 'evaluation_id': '64 lowercase hex', 'max_steps': [1, 200]}}}

    def inspect(self):
        result = {'executor': 'ready'}
        for label, rel, fields in [
            ('scheduler', 'state/scheduler.json', ['phase', 'desired', 'window', 'idle_after_completion']),
            ('job', 'scheduling/job.json', ['job_id', 'max_steps']),
            ('download', 'reports/model-verification.json', ['status', 'complete', 'total_bytes']),
        ]:
            path = self.root / rel
            try:
                result[label] = project(json.loads(path.read_text()), fields)
            except (OSError, ValueError):
                result[label] = {'available': False}
        result['catalog'] = self.catalog()
        job_id = result.get('job', {}).get('job_id')
        result['current_job'] = self.job_info()
        result['current_job_complete'] = bool(job_id and (self.root / 'runs' / job_id / 'job-complete.json').is_file())
        return result

    def execute(self, action, params):
        if action == 'knowledge.status':
            from .l20_branch import knowledge_status
            return knowledge_status()
        if action.startswith('l20.'):
            from . import l20_branch
            return l20_branch.admit(self,params) if action=='l20.admit' else l20_branch.compare(self,**params) if action=='l20.compare' else l20_branch.status(self)
        if action == 'evaluation.candidate':
            from .iteration import request_evaluation
            return request_evaluation(self)
        if action == 'evaluation.baseline':
            from .evidence import reusable_baseline
            existing = reusable_baseline(self)
            if existing is not None:
                return existing
        if action == 'inspect':
            return self.inspect()
        if action == 'training.admit':
            from .admission import admit
            return admit(self, params)
        if action == 'training.propose':
            return self.propose(params)
        stdin = None
        if action.startswith('knowledge.'):
            args = [sys.executable, '-m', 'research.knowledge.factory', action.split('.')[1]]
        elif action == 'data.teacher':
            args = [sys.executable, '-m', 'research.teacher.worker', 'produce']
        elif action.startswith('data.'):
            args = [sys.executable, '-m', 'research.data_factory.cli', action.split('.')[1]]
            if action == 'data.compose':
                args += ['--stdin']
                stdin = json.dumps({'dataset_ids': params['dataset_ids']})
            if action == 'data.propose':
                args += ['--stdin']
                stdin = json.dumps(params['proposal'])
            if action == 'data.plan':
                args += ['--limit', str(params.get('limit', 4))]
            if action == 'data.run':
                args += ['--plan-id', params['plan_id']]
        elif action in ('evaluation.freeze', 'evaluation.status', 'evaluation.baseline', 'evaluation.candidate'):
            subcommand = {'evaluation.freeze': 'freeze', 'evaluation.status': 'status', 'evaluation.baseline': 'baseline-local', 'evaluation.candidate': 'candidate-local'}[action]
            args = [sys.executable, str(self.root / 'research/evaluation/cli.py'), subcommand]
        else:
            raise ValueError('Evaluation adapter not yet configured')
        # No shell, no inherited API keys, no remote/GPU mutation capabilities.
        env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(self.state), 'PYTHONPATH': str(self.root), 'PYTHONDONTWRITEBYTECODE': '1', 'RTL_EXECUTOR_LOCK_HELD': '1'}
        if action=='data.teacher':
            env.update({k:v for k,v in os.environ.items() if k in ('RTL_TEACHER_ROUTER_URL','RTL_TEACHER_ROUTER_TOKEN_FILE','RTL_TEACHER_PRIMARY_URL','RTL_TEACHER_PRIMARY_TOKEN_FILE','RTL_TEACHER_STATE')})
        process = subprocess.run(args, cwd=self.root, env=env, input=stdin, capture_output=True, text=True, pass_fds=(self.lock_fd,) if hasattr(self, 'lock_fd') else (), timeout=2100 if action in ('evaluation.baseline', 'evaluation.candidate') else 900)
        if process.returncode:
            raise RuntimeError('Fixed CPU task failed')
        if len(process.stdout) > 1024 * 1024:
            raise ValueError('Task output exceeds limit')
        output = json.loads(process.stdout)
        result = project(output, ['status', 'state', 'plan_id', 'dataset_id', 'evaluation_id', 'candidate_id', 'count', 'accepted', 'rejected', 'passed', 'reason', 'request_count', 'generation_seconds', 'completion_tokens'])
        if action == 'knowledge.build':
            from research.knowledge.factory import read_dataset
            result['validated'] = read_dataset(self.root/'research/knowledge/artifacts'/output['dataset_id'])['validated']
        if action == 'data.compose':
            from .evidence import refresh
            result['validated'] = any(record['dataset_id'] == output.get('dataset_id') for record in refresh(self))
        if action == 'data.run':
            from .evidence import refresh
            result['validated'] = any(record['dataset_id'] == params['plan_id'] for record in refresh(self))
        if action == 'evaluation.freeze':
            from .evidence import verified_freeze
            result['freeze_verified'] = verified_freeze(self, output.get('freeze_id'))
            result['freeze_id'] = output.get('freeze_id')
        if action == 'evaluation.baseline':
            from .evidence import refresh, evaluations
            refresh(self)
            evidence = evaluations(self)
            if evidence:
                matched = [e for e in evidence if e['freeze_id'] == output.get('freeze_id')]
                if matched:
                    result.update(matched[-1])
        if action == 'evaluation.candidate':
            from .iteration import outcomes
            job = json.loads((self.root / 'scheduling/job.json').read_text())
            compared = outcomes(self, job)
            result.update(comparison_complete=bool(compared))
            if compared:
                result.update(compared[-1])
        return result

    def propose(self, params):
        from .evidence import refresh
        refresh(self)
        # Gate records are written only by trusted evaluation/data adapters, not HTTP clients.
        folder = self.state / 'verified'
        data = json.loads((folder / ('dataset-' + params['dataset_id'] + '.json')).read_text())
        evaluation = json.loads((folder / ('evaluation-' + params['evaluation_id'] + '.json')).read_text())
        if data.get('dataset_id') != params['dataset_id'] or data.get('validated') is not True:
            raise ValueError('Dataset not validated')
        if evaluation.get('evaluation_id') != params['evaluation_id'] or evaluation.get('baseline_complete') is not True:
            raise ValueError('Evaluation baseline gate not passed')
        # Freeze contamination check is performed against the candidate at admission.
        if not evaluation.get('freeze_id'):
            raise ValueError('Missing frozen evaluation identity')
        proposal = {'dataset_id': params['dataset_id'], 'evaluation_id': params['evaluation_id'], 'max_steps': params.get('max_steps', 100), 'status': 'proposal_only'}
        ident = digest(proposal)
        folder = self.state / 'proposals'
        folder.mkdir(exist_ok=True)
        atomic(folder / (ident + '.json'), proposal)
        return {'proposal_id': ident, 'status': 'proposal_only', 'active_job_changed': False}
