"""Build a small, simulator-checked trace curriculum in the held-out JSON shape."""
import json
import pathlib
import subprocess
import tempfile

from . import curriculum, factory

FORMAT = 'trace-json-v2'
WIDTHS = (4, 6)


def messages(item):
    inputs = item['trace_inputs']
    outputs = item['trace_outputs']
    names = tuple(outputs[0])
    expected = {name: [row[name] for row in outputs] for name in names}
    prompt = (
        '按输入顺序推演下面的RTL。组合逻辑每行独立稳定后读取；时序逻辑每行对应一个上升沿，'
        '按题目说明读取更新后的状态。只返回一个JSON对象，键为输出信号名，值为按顺序排列的数值数组。\n'
        + item['sampling'] + '\n```verilog\n' + item['reference'] + '```\n输入序列：'
        + factory.canonical(inputs)
    )
    return prompt, factory.canonical(expected)


def build():
    from .curriculum import FAMILIES
    from judge.runner import run_judge

    source_rows = factory.sources()
    image = subprocess.check_output(
        ['docker', 'image', 'inspect', 'rtl-judge:local', '--format', '{{.Id}}'],
        text=True,
        timeout=15,
    ).strip()
    if not image.startswith('sha256:') or len(image) != 71:
        raise ValueError('invalid judge image identity')

    # lesson() uses this fixed allowlist; changing it only in this process
    # permits new, deterministic widths without changing the legacy builder.
    curriculum.WIDTHS = WIDTHS
    recipe = {name: factory.sha((factory.HERE / name).read_bytes())
              for name in ('factory.py', 'curriculum.py')}
    identity = {
        'version': curriculum.VERSION,
        'sources': source_rows,
        'recipe': recipe,
        'judge_image': image,
        'families': FAMILIES,
        'widths': WIDTHS,
        'target_format': FORMAT,
        'target_builder_sha256': factory.sha(pathlib.Path(__file__).read_bytes()),
    }
    ident = factory.sha(factory.canonical(identity))
    parent = factory.ROOT / 'artifacts'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / ident
    if target.exists():
        manifest = factory.read_dataset(target)
        return {'status': 'already_built', 'dataset_id': ident,
                'validated': True, 'train': manifest['train']}

    with tempfile.TemporaryDirectory(prefix='.targeted-', dir=parent) as tmp:
        folder = pathlib.Path(tmp)
        rows, lessons, evidence = [], [], {}
        for family in FAMILIES:
            for width in WIDTHS:
                item = curriculum.lesson(family, width)
                good = run_judge(item['reference'], item['testbench'],
                                 trusted_testbench=True, image=image)
                wrong = run_judge(item['mutant'], item['testbench'],
                                  trusted_testbench=True, image=image)
                if (good.get('status') != 'pass'
                        or wrong.get('compile', {}).get('status') != 'pass'
                        or wrong.get('simulation', {}).get('status') != 'fail'):
                    raise ValueError('target lesson failed independent simulation or mutant gate')
                key = item['lesson_id']
                proof = {
                    'lesson_id': key,
                    'reference': good,
                    'mutant': wrong,
                    'coverage': item['coverage'],
                    'reference_sha256': factory.sha(item['reference']),
                    'testbench_sha256': factory.sha(item['testbench']),
                }
                lessons.append(item)
                evidence[key] = proof
                prompt, answer = messages(item)
                rows.append({
                    'task_id': key + ':predict',
                    'family_id': 'knowledge_' + family,
                    'split': 'train',
                    'validation_level': 'K1-grounded',
                    'kind': 'predict',
                    'semantic_sha256': factory.sha(factory.canonical({
                        'lesson_semantics': item['semantic_sha256'],
                        'kind': 'predict', 'format': FORMAT,
                    })),
                    'knowledge_evidence_sha256': factory.sha(factory.canonical(proof)),
                    'knowledge_source_sha256': factory.sha(factory.canonical(source_rows)),
                    'verification_scope': 'Authored trace; reference independently simulated and mutant rejected; no formal proof.',
                    'messages': [
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': answer},
                    ],
                })
        train_path = folder / 'sft_train.jsonl'
        train_path.write_text(''.join(factory.canonical(row) + '\n' for row in rows))
        factory.write(folder / 'lessons.json', lessons)
        factory.write(folder / 'evidence.json', evidence)
        factory.write(folder / 'sources.json', source_rows)
        files = {p.name: factory.sha(p.read_bytes()) for p in folder.iterdir() if p.is_file()}
        manifest = {
            'dataset_id': ident,
            'kind': 'knowledge_trace_sft',
            'identity': identity,
            'train': len(rows),
            'val': 0,
            'files': files,
            'tasks': [{'task_id': row['task_id'], 'split': 'train',
                       'semantic_sha256': row['semantic_sha256']} for row in rows],
            'scope': '18 authored JSON trace predictions across nine executable concept families and two new widths.',
            'validated': True,
        }
        factory.write(folder / 'manifest.json', manifest)
        # Validate every binding at its content-addressed path before the
        # executor can see it; the executor repeats this at admission time.
        probe = parent / ident
        probe.symlink_to(folder, target_is_directory=True)
        try:
            factory.read_dataset(probe)
        finally:
            probe.unlink()
        for path in folder.iterdir():
            path.chmod(0o444)
        folder.chmod(0o555)
        folder.rename(target)
    return {'status': 'built', 'dataset_id': ident,
            'validated': True, 'train': len(rows), 'val': 0}


if __name__ == '__main__':
    print(json.dumps(build(), separators=(',', ':')))
