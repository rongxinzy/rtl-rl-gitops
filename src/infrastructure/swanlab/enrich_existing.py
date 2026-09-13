"""Fill existing run headers without replacing runs or changing their state.

Uses the same name/description and additive-label endpoints as SwanLab Web UI.
SDK 0.10 resume updates config, but does not update existing run headers.
"""
import contextlib
import io
import json
import os
import swanlab
from relay import presentation, run_id, snapshot


def main():
    results = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        api = swanlab.Api(api_key=os.environ['SWANLAB_API_KEY'])
        project = os.environ.get('RTL_SWANLAB_PROJECT', 'RTL-RL')
        username = api.username
        data = snapshot()
        for job in data['jobs']:
            try:
                path = username + '/' + project + '/' + run_id(job)
                run = api.run(path)
                if not run.run_id:
                    continue
                _, description, tags = presentation(job, data.get('device_observation'))
                name = 'RTL SFT | Qwen3.8-27B | ' + job['job_id']
                first = run._put('/project/' + username + '/' + project + '/runs/' + run.run_id,
                    data={'name': name, 'description': description})
                existing = {x['name'] for x in run.labels}
                missing = [{'name': x} for x in tags if x not in existing]
                second = run._put('/project/' + username + '/' + project + '/runs',
                    data={'cuids': [run.run_id], 'createLabels': missing}) if missing else None
                fresh = api.run(path)
                verified = (first.ok and (second is None or second.ok) and fresh.name == name
                    and fresh.description == description
                    and set(tags).issubset({x['name'] for x in fresh.labels}))
                results.append({'job_id': job['job_id'], 'headers_verified': verified,
                    'tag_count': len(fresh.labels), 'state': fresh.state})
            except Exception as exc:
                results.append({'job_id': job['job_id'], 'error_type': type(exc).__name__})
    print(json.dumps(results))
    return all(x.get('headers_verified') for x in results) and bool(results)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
