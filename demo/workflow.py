"""Colab experiment orchestration; model training remains in src/."""

import contextlib
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import uuid
import zipfile

VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
BASELINES = {'TopPopular', 'Random'}
ALIASES = {'ml-1m': ('ml-1m', 'ml-1m'), 'amazon_Baby': ('baby', 'amazon_baby'),
           'amazon_Toys_and_Games': ('toys', 'amazon_toys')}
FOLDERS = {'ADRec': 'ADRec/src', 'DiffuRec': 'DiffuRec/src', 'GPTRec': 'GPTRec/src',
           'SASRec': 'SASRec', 'T-DiffRec': 'DiffRec/T-DiffRec', 'TopPopular': 'TopPopular', 'Random': 'RandomRecs'}
PARAMS = {
    'ADRec': {'batch_size', 'hidden_size', 'lr', 'dropout', 'emb_dropout', 'weight_decay', 'diffusion_steps', 'noise_schedule'},
    'DiffuRec': {'batch_size', 'hidden_size', 'num_blocks', 'lr', 'dropout', 'emb_dropout', 'noise_schedule', 'diffusion_steps', 'weight_decay'},
    'SASRec': {'hidden_units', 'dropout_rate', 'num_blocks', 'num_heads', 'batch_size', 'learning_rate', 'l2_emb'},
    'T-DiffRec': {'batch_size', 'lr', 'weight_decay', 'dims', 'steps', 'sampling_steps', 'noise_scale', 'noise_min', 'noise_max', 'w_min', 'w_max'},
    'GPTRec': {'model_params.n_embd', 'model_params.n_layer', 'model_params.n_head', 'model_params.embd_pdrop',
               'model_params.attn_pdrop', 'seqrec_module.lr', 'dataloader.batch_size', 'generation_params.num_return_sequences', 'ra_temperature'},
    'TopPopular': set(), 'Random': set(),
}


def configure(model, output, python=sys.executable, root=ROOT):
    if model not in FOLDERS:
        raise ValueError(model)
    output = Path(output).resolve()
    if not output.name.endswith(('_new', '_demo')):
        raise ValueError('Use a separate output folder ending in _new or _demo')
    output.mkdir(parents=True, exist_ok=True)
    return {'model': model, 'output': output, 'python': str(python), 'root': Path(root).resolve()}


def environment(ctx):
    env = os.environ.copy()
    env.update(EXPERIMENT_OUTPUT_DIR=str(ctx['output']), EXPERIMENT_FILE_SUFFIX='_new',
               PYTHONPATH=str(ctx['root'] / 'src'), MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    env.pop('EXPERIMENT_CHECKPOINT_DIR', None)
    return env


@contextlib.contextmanager
def exclusive(ctx):
    path = ctx['output'] / '.demo_run.lock'
    try:
        handle = path.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError(f'Another notebook is running here: {path}. If its runtime was interrupted, remove this lock before resuming.') from None
    try:
        with handle:
            handle.write(json.dumps({'model': ctx['model'], 'started': datetime.now(timezone.utc).isoformat()}))
        yield
    finally:
        path.unlink()


def presets(model, dataset, maxlen):
    data = json.loads((Path(__file__).parent / 'august_presets.json').read_text(encoding='utf-8'))
    return data[model][f'{dataset}|{maxlen or 0}']


def cases(model, datasets, maxlens):
    for dataset in datasets:
        if dataset not in ALIASES:
            raise ValueError(dataset)
        for length in ([None] if model in BASELINES | {'T-DiffRec'} else maxlens):
            yield dataset, length


def source_path(ctx, dataset):
    data = ctx['root'] / 'src/data'
    if dataset == 'ml-1m':
        return data / 'info/ratings.dat'
    category = {'amazon_Baby': 'Baby', 'amazon_Toys_and_Games': 'Toys_and_Games'}[dataset]
    return data / 'amazon' / f'reviews_{category}_5.json'


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as handle:
            shutil.copyfileobj(response, handle)
        if not temporary.stat().st_size:
            raise ValueError(f'Empty download: {url}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(ctx, datasets):
    for dataset in datasets:
        path = source_path(ctx, dataset)
        if not path.exists():
            if dataset == 'ml-1m':
                archive = path.parent / 'ml-1m.zip'
                download('https://files.grouplens.org/datasets/movielens/ml-1m.zip', archive)
                with zipfile.ZipFile(archive) as z:
                    for name in ('ratings.dat', 'movies.dat', 'users.dat'):
                        (path.parent / name).write_bytes(z.read('ml-1m/' + name))
                archive.unlink()
            else:
                compressed = path.with_suffix('.json.gz')
                download('https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/' + compressed.name, compressed)
                temporary = path.with_suffix('.part')
                try:
                    with gzip.open(compressed, 'rb') as src, temporary.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                compressed.unlink()
        if not path.stat().st_size:
            raise ValueError(f'Empty source: {path}')
        if ctx['model'] == 'ADRec':
            subprocess.run([ctx['python'], 'get_data.py', '--dataset', ALIASES[dataset][0]],
                           cwd=ctx['root'] / 'src/ADRec/src', env=environment(ctx), check=True)
        elif ctx['model'] == 'T-DiffRec':
            subprocess.run([ctx['python'], 'split_load_data_dp.py', '--dataset', dataset],
                           cwd=ctx['root'] / 'src/DiffRec/T-DiffRec', env=environment(ctx), check=True)


def preflight(ctx, device):
    probe = 'import torch, pandas, scipy, sklearn, einops, yaml, openpyxl; '
    probe += "assert torch.cuda.is_available(), 'Choose a Colab GPU runtime or DEVICE=cpu'" if device == 'cuda' else 'print(torch.__version__)'
    subprocess.run([ctx['python'], '-c', probe], env=environment(ctx), check=True)
    params = presets(ctx['model'], 'ml-1m', None if ctx['model'] in BASELINES | {'T-DiffRec'} else 50)[0]['params']
    for phase in (['training'] if ctx['model'] in BASELINES else ['tuning', 'training']):
        cwd, args = command(ctx, 'ml-1m', None if ctx['model'] in BASELINES | {'T-DiffRec'} else 50, params, phase, device=device)
        if ctx['model'] == 'GPTRec':
            args += ['--cfg', 'job']
        else:
            args += ['--help']
        subprocess.run(args, cwd=cwd, env=environment(ctx), check=True, stdout=subprocess.DEVNULL)


def signature(ctx, dataset):
    digest = hashlib.sha256()
    with source_path(ctx, dataset).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    for folder in (ctx['root'] / 'src' / FOLDERS[ctx['model']], ctx['root'] / 'src/experiment_tools'):
        for file in sorted(folder.rglob('*.py')):
            digest.update(str(file.relative_to(ctx['root'])).encode())
            digest.update(file.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def command(ctx, dataset, maxlen, params, phase, max_epochs=250, patience=10, device='cuda'):
    model = ctx['model']
    if phase not in ('tuning', 'training') or device not in ('cuda', 'cpu'):
        raise ValueError((phase, device))
    if dataset not in ALIASES or (model not in BASELINES | {'T-DiffRec'} and maxlen not in (50, 100)):
        raise ValueError((dataset, maxlen))
    unknown = set(params) - PARAMS[model] - {'epochs'}
    if unknown:
        raise ValueError(f'Unsupported {model} parameters: {sorted(unknown)}')
    final = phase == 'training'
    epochs = params.get('epochs') if final else max_epochs
    if model not in BASELINES and (not isinstance(epochs, int) or epochs < 1):
        raise ValueError('A positive integer epochs is required for final training')
    if model == 'T-DiffRec' and dataset == 'ml-1m':
        fixed = dict(steps=100, sampling_steps=100, noise_scale=.1, noise_min=.0001, noise_max=.02, w_min=.1, w_max=1.)
        if any(k in params and params[k] != v for k, v in fixed.items()):
            raise ValueError('T-DiffRec hardcodes ML-1M diffusion settings; these overrides would be ignored')
    cwd = ctx['root'] / 'src' / FOLDERS[model]
    if model in BASELINES:
        if not final:
            raise ValueError('This baseline has no tuning')
        script = 'TopPopular_model.py' if model == 'TopPopular' else 'RandomRecsModel.py'
        return cwd, [ctx['python'], script, '--dataset', ALIASES[dataset][0], '--topk_list', '10', '20', '100']
    if model == 'GPTRec':
        overrides = dict(params)
        overrides.pop('epochs', None)
        temperature = overrides.pop('ra_temperature', 1.)
        overrides.update({'dataset_name': dataset, 'dataset.max_length': maxlen, 'model_params.n_positions': maxlen,
                          'final_train': final, 'final_epochs': epochs, 'trainer_params.max_epochs': epochs,
                          'trainer_params.accelerator': 'gpu' if device == 'cuda' else 'cpu', 'trainer_params.devices': 1,
                          'patience': patience, 'generation': True, 'mode': 'relevance_aggregation',
                          'test_metrics': final, 'seed': 42, 'dataloader.num_workers': 2,
                          'dataloader.validation_size': None, 'evaluator.top_k': [10, 20, 100],
                          'ra_temperature': temperature if final else 'auto', 'ra_temperature_source': None,
                          'generation_params.temperature': temperature, 'seqrec_module.filter_seen': True})
        args = [ctx['python'], 'run_train_predict.py' if final else 'tune.py', '--config-name', 'GPT_train_predict']
        return cwd, args + [f'++{k}={json.dumps(v)}' for k, v in overrides.items()]
    args = [ctx['python'], 'main.py' if final or model != 'SASRec' else 'tune.py', '--dataset',
            ALIASES[dataset][0] if model == 'ADRec' else dataset]
    def add(key, value):
        args.extend(['--' + key, str(value)])
    if model in ('ADRec', 'DiffuRec'):
        add('max_len', maxlen)
        add('epochs', epochs)
        add('random_seed', 42)
        add('device', 'cuda:0' if device == 'cuda' and model == 'ADRec' else device)
        args.extend(['--metric_ks', '10', '20', '100'])
        if model == 'ADRec':
            add('mask_seen', 'True')
            add('pretrained', 'False')
            add('freeze_emb', 'False')
        if final:
            args.append('--final' if model == 'ADRec' else '--final_train')
        else:
            add('patience', patience)
            add('eval_interval', 5)
    elif model == 'SASRec':
        add('maxlen', maxlen)
        add('num_epochs' if final else 'max_epochs', epochs)
        if not final:
            add('patience', patience)
            add('seed', 42)
    else:
        add('epochs', epochs)
        add('topN', '[10,20,100]')
        add('patience', patience)
        if final:
            args.append('--final_train')
        if device == 'cuda':
            args.append('--cuda')
    for key, value in params.items():
        if key != 'epochs':
            add(key, value)
    return cwd, args


def read_registry(ctx):
    path = ctx['output'] / 'service_files/all_experiments_new.csv'
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def tuning_rows(ctx):
    rows = []
    for path in (ctx['output'] / 'service_files/models' / ctx['model']).glob('*/tuning_files/*/demo_run.json'):
        row = json.loads(path.read_text(encoding='utf-8'))
        rows.append(row)
    return sorted(rows, key=lambda r: r['started'])


def metric_key(row):
    metrics = row['validation']
    return tuple(float(metrics.get(k, 0.)) for k in ('recall@10', 'ndcg@10', 'mrr@10', 'coverage@10'))


def choose(ctx, dataset, maxlen, mode='auto', manual=None):
    if ctx['model'] in BASELINES:
        return {'params': {}, 'source': 'baseline'}
    if mode == 'manual':
        if manual is None or 'epochs' not in manual:
            raise ValueError('Manual parameters must include epochs')
        return {'params': dict(manual), 'source': 'manual'}
    if mode != 'auto':
        raise ValueError(mode)
    fingerprint = signature(ctx, dataset)
    rows = [r for r in tuning_rows(ctx) if r['status'] == 'complete' and r['dataset'] == dataset
            and r['maxlen'] == maxlen and r['seed'] == 42 and r['signature'] == fingerprint]
    if not rows:
        raise ValueError(f'No successful matching tuning for {dataset}, maxlen={maxlen}. Run tuning or choose manual parameters.')
    best = max(rows, key=metric_key)
    return {'params': best['selected_params'], 'source': best['id'], 'validation': best['validation']}


def export_tables(ctx):
    if not read_registry(ctx):
        return
    subprocess.run([ctx['python'], str(ctx['root'] / 'src/experiment_tools/generate_result_tables.py'), '--suffix', '_new'],
                   cwd=ctx['root'], env=environment(ctx), check=True)


def run(ctx, dataset, maxlen, params, phase, max_epochs=250, patience=10, device='cuda', source='manual'):
    cwd, args = command(ctx, dataset, maxlen, params, phase, max_epochs, patience, device)
    root = ctx['output'] / 'service_files/models' / ctx['model'] / ALIASES[dataset][1]
    folder = root / ('tuning_files' if phase == 'tuning' else 'logs')
    ident = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    record_dir = folder / ('demo_' + ident)
    record = dict(id=ident, model=ctx['model'], dataset=dataset, maxlen=maxlen, seed=42, phase=phase,
                  params=params, source=source, status='running', started=datetime.now(timezone.utc).isoformat(),
                  command=args, cwd=str(cwd), signature=signature(ctx, dataset))
    with exclusive(ctx):
        record_dir.mkdir(parents=True)
        path = record_dir / 'demo_run.json'
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        before = set(folder.glob('*/validation_selection.json'))
        previous = {r['run_id'] for r in read_registry(ctx)}
        try:
            print(' '.join(args), flush=True)
            env = environment(ctx)
            env['EXPERIMENT_RUN_ID'] = ident
            env['EXPERIMENT_CHECKPOINT_DIR'] = str(record_dir / 'checkpoints')
            subprocess.run(args, cwd=cwd, env=env, check=True)
            if phase == 'tuning':
                selections = list(set(folder.glob('*/validation_selection.json')) - before)
                if len(selections) != 1:
                    raise RuntimeError(f'Expected one validation selection, found {len(selections)}')
                selection = json.loads(selections[0].read_text(encoding='utf-8'))
                metrics = {k.lower(): v for k, v in (selection.get('selected_metrics') or {}).items()}
                if 'recall@10' not in metrics or not all(math.isfinite(float(v)) for v in metrics.values()):
                    raise ValueError('Missing/non-finite validation metrics')
                epoch = int(selection['selected_epoch'])
                if not 1 <= epoch <= max_epochs:
                    raise ValueError(f'Invalid selected_epoch={epoch}')
                selected = {**params, 'epochs': epoch}
                if ctx['model'] == 'GPTRec':
                    selected['ra_temperature'] = float(metrics['ra_temperature'])
                record.update(validation=metrics, selected_params=selected, native_selection=str(selections[0]))
            else:
                new_rows = [r for r in read_registry(ctx) if r['run_id'] not in previous]
                names = {'ml-1m': 'ML-1M', 'amazon_Baby': 'Amazon Baby', 'amazon_Toys_and_Games': 'Amazon Toys'}
                matching = [r for r in new_rows if r['model'] == ctx['model'] and r['dataset'] == names[dataset]
                            and r['maxlen'] == (str(maxlen) if maxlen else '') and r['seed'] == '42']
                if len(matching) != 1:
                    raise RuntimeError('Final run did not append exactly one matching registry row')
                row = matching[0]
                for k in (10, 20, 100):
                    for metric in ('recall', 'ndcg', 'mrr', 'coverage'):
                        if not 0 <= float(row[f'{metric}@{k}']) <= 1:
                            raise ValueError(f'Invalid final metric {metric}@{k}')
                if not math.isfinite(float(row['latency_sec'])) or float(row['latency_sec']) < 0:
                    raise ValueError('Invalid final latency')
                fields = list(read_registry(ctx)[0])
                rows = read_registry(ctx)
                for item in rows:
                    if (item['model'], item['dataset'], item['maxlen'], item['seed']) == (row['model'], row['dataset'], row['maxlen'], row['seed']):
                        item['selected'] = str(item['run_id'] == row['run_id']).lower()
                registry = ctx['output'] / 'service_files/all_experiments_new.csv'
                with registry.open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                record['native_run_id'] = row['run_id']
                export_tables(ctx)
            record['status'] = 'complete'
        except BaseException as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record
