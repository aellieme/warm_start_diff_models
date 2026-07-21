import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import hashlib
import json
import random
import time
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm

from evaluate_topk_dp import compute_all_metrics
from plotting import TrainingPlotter
from utils import (build_candidate_mask, eligible_warm_start_rows,
                   filter_history_to_candidates, mask_ranking_scores)
from utils import prepare_model_history

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_tools.experiment_tracking import (
    ExperimentTracker,
    checkpoint_path,
    recommendation_popularity,
    save_torch_checkpoint,
)


TUNING_CHECKPOINT_INTERVAL = 10
TUNING_CHECKPOINT_KEEP = 2
_RESUME_SIGNATURE_FIELDS = (
    'dataset', 'random_seed', 'max_len', 'device', 'num_gpu', 'batch_size', 'hidden_size',
    'dropout', 'emb_dropout', 'hidden_act', 'num_blocks', 'decay_step',
    'gamma', 'metric_ks', 'optimizer', 'lr', 'loss_lambda',
    'weight_decay', 'momentum', 'schedule_sampler_name', 'diffusion_steps',
    'lambda_uncertainty', 'noise_schedule', 'rescale_timesteps',
    'eval_interval', 'patience', 'eval_repeats', 'amp',
)


def _resume_signature(args):
    """Return the training/selection settings that a resume must preserve."""
    return {
        field: getattr(args, field, None)
        for field in _RESUME_SIGNATURE_FIELDS
    }


def _tuning_checkpoint_prefix(args):
    signature_json = json.dumps(
        _resume_signature(args), sort_keys=True, separators=(',', ':'),
        default=str,
    )
    digest = hashlib.sha256(signature_json.encode('utf-8')).hexdigest()[:12]
    stable_path = checkpoint_path(
        'DiffuRec', args.dataset, args.max_len, args.random_seed,
    )
    directory = stable_path.parent / 'tuning'
    directory.mkdir(parents=True, exist_ok=True)
    return directory, f'{stable_path.stem}_{digest}'


def tuning_checkpoint_path(args, completed_epoch):
    directory, prefix = _tuning_checkpoint_prefix(args)
    return directory / f'{prefix}_epoch{int(completed_epoch):04d}.pt'


def list_tuning_checkpoints(args):
    directory, prefix = _tuning_checkpoint_prefix(args)
    return sorted(directory.glob(f'{prefix}_epoch*.pt'))


def _state_dict_on_cpu(model):
    if model is None:
        return None
    return {
        name: value.detach().cpu() if torch.is_tensor(value) else value
        for name, value in model.state_dict().items()
    }


def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state.get('cuda'):
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _move_optimizer_state(optimizer, device):
    for optimizer_state in optimizer.state.values():
        for name, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[name] = value.to(device)


def _prepare_resumed_epoch_loader(data_loader):
    """Preserve the pre-interruption RandomSampler stream after worker restart."""
    sampler = getattr(data_loader, 'sampler', None)
    if sampler is None or not hasattr(sampler, 'generator'):
        return
    # RandomSampler(generator=None) draws exactly one seed from the global CPU
    # RNG at the beginning of every epoch.  Reproduce that draw explicitly.
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    sampler.generator = torch.Generator().manual_seed(sampler_seed)
    # A newly constructed DataLoader also draws a worker base seed.  Keep that
    # bookkeeping off the restored training RNG stream; the dataset is static.
    if getattr(data_loader, 'generator', None) is None:
        data_loader.generator = torch.Generator().manual_seed(sampler_seed ^ 0x5DEECE66D)


def save_tuning_checkpoint(payload, args, completed_epoch, keep=TUNING_CHECKPOINT_KEEP):
    """Atomically save a rolling tuning checkpoint and retain only the newest files."""
    path = tuning_checkpoint_path(args, completed_epoch)
    save_torch_checkpoint(payload, path)
    checkpoints = list_tuning_checkpoints(args)
    for stale_path in checkpoints[:-max(1, int(keep))]:
        stale_path.unlink(missing_ok=True)
        print(f'Deleted old tuning checkpoint: {stale_path}')
    return path


def _load_tuning_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        return torch.load(path, map_location='cpu')


def _resolve_resume_checkpoint(args):
    requested = getattr(args, 'resume_checkpoint', None)
    if not requested:
        return None
    if str(requested).lower() == 'latest':
        checkpoints = list_tuning_checkpoints(args)
        if not checkpoints:
            raise FileNotFoundError(
                'No tuning checkpoint matches the current DiffuRec configuration'
            )
        return checkpoints[-1]
    path = Path(requested).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Tuning checkpoint not found: {path}')
    return path

def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def evaluation_seeds(args, repeats=None):
    """Return predeclared reverse-diffusion seeds without inspecting any metrics."""
    if repeats is None:
        repeats = getattr(args, 'eval_repeats', 1)
    repeats = max(1, int(repeats))
    first_seed = int(args.random_seed)
    return tuple(first_seed + repeat for repeat in range(repeats))


@contextmanager
def isolated_torch_rng(seed, device):
    """Run stochastic inference without advancing the training RNG streams."""
    device = torch.device(device)
    cuda_devices = []
    if device.type == 'cuda' and torch.cuda.is_available():
        cuda_devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def _collect_ranked_predictions(
    model, data_loader, args, candidate_mask, seed, use_amp=False,
):
    """Collect one reproducible stochastic ranking pass for a fixed seed."""
    device = torch.device(args.device)
    all_actual = []
    all_predicted = []
    excluded_examples = 0
    started = time.perf_counter()

    with isolated_torch_rng(seed, device), torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16,
        enabled=use_amp,
    ):
        for batch in data_loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            full_history = filter_history_to_candidates(
                batch[2] if len(batch) > 2 else batch[0], candidate_mask
            )
            batch[0] = prepare_model_history(
                full_history, candidate_mask, batch[0].shape[1]
            )
            _, rep_diffu, _, _, _, _ = model(
                batch[0], batch[1], train_flag=False
            )
            scoring_model = model.module if isinstance(model, nn.DataParallel) else model
            scores = scoring_model.diffu_rep_pre(rep_diffu)
            valid_rows = eligible_warm_start_rows(
                full_history, batch[1], candidate_mask
            )
            excluded_examples += (~valid_rows).sum().item()
            mask_ranking_scores(scores, full_history, candidate_mask)
            _, topk = torch.topk(scores, k=max(args.metric_ks), dim=-1)
            for row in valid_rows.nonzero(as_tuple=False).squeeze(-1).tolist():
                all_actual.append([batch[1][row].item()])
                all_predicted.append(topk[row].cpu().tolist())

    return {
        'actual': all_actual,
        'predicted': all_predicted,
        'excluded_examples': excluded_examples,
        'elapsed_sec': time.perf_counter() - started,
        'seed': int(seed),
    }


def evaluate_stochastic_ranking(
    model, data_loader, args, candidate_items, candidate_mask, use_amp=False,
    repeats=None,
):
    """Average metrics over fixed inference seeds without unioning recommendations."""
    was_training = model.training
    model.eval()
    runs = []
    try:
        for seed in evaluation_seeds(args, repeats=repeats):
            prediction_run = _collect_ranked_predictions(
                model, data_loader, args, candidate_mask, seed, use_amp=use_amp
            )
            if not prediction_run['actual']:
                raise ValueError("No eligible warm-start examples remain for evaluation")
            metrics = compute_all_metrics(
                prediction_run['actual'],
                prediction_run['predicted'],
                args.metric_ks,
                len(candidate_items),
                candidate_items=candidate_items,
            )
            prediction_run['metrics'] = metrics
            runs.append(prediction_run)
    finally:
        model.train(was_training)

    metric_arrays = [
        np.asarray([run['metrics'][index] for run in runs], dtype=float)
        for index in range(5)
    ]
    means = tuple(values.mean(axis=0).tolist() for values in metric_arrays)
    stds = tuple(values.std(axis=0).tolist() for values in metric_arrays)
    canonical = runs[0]
    return {
        'means': means,
        'stds': stds,
        'canonical_actual': canonical['actual'],
        'canonical_predicted': canonical['predicted'],
        'excluded_examples': canonical['excluded_examples'],
        'mean_elapsed_sec': float(np.mean([run['elapsed_sec'] for run in runs])),
        'wall_elapsed_sec': float(np.sum([run['elapsed_sec'] for run in runs])),
        'seeds': [run['seed'] for run in runs],
    }


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    candidate_mask = build_candidate_mask(
        args.coverage_candidate_items,
        args.item_num + 1,
        device,
    )
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            full_history = filter_history_to_candidates(
                test_batch[2] if len(test_batch) > 2 else test_batch[0], candidate_mask
            )
            test_batch[0] = prepare_model_history(
                full_history, candidate_mask, test_batch[0].shape[1]
            )
            
            scores_rec, rep_diffu, _, _, _, _ = model_joint(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)
            valid_rows = eligible_warm_start_rows(
                full_history, test_batch[1], candidate_mask
            )
            if not valid_rows.any():
                continue
            mask_ranking_scores(
                scores_rec_diffu, full_history, candidate_mask
            )
            metrics = hrs_and_ndcgs_k(
                scores_rec_diffu[valid_rows], test_batch[1][valid_rows], [5, 10, 20]
            )
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)

def evaluate_and_print(model, data_loader, args, logger, description="evaluation", save_recs=False):
    """Evaluate stochastic DiffuRec ranking on fixed, isolated inference seeds."""
    device = args.device
    use_amp = getattr(args, 'amp', False) and device == 'cuda'
    candidate_items = set(args.coverage_candidate_items)
    candidate_items.discard(0)
    candidate_mask = build_candidate_mask(
        candidate_items,
        args.item_num + 1,
        device,
    )
    result = evaluate_stochastic_ranking(
        model,
        data_loader,
        args,
        candidate_items,
        candidate_mask,
        use_amp=use_amp,
        repeats=1,
    )
    _, rec, ndcg, mrr, cov = result['means']
    _, rec_std, ndcg_std, mrr_std, cov_std = result['stds']
    all_actual = result['canonical_actual']
    all_predicted = result['canonical_predicted']
    num_users = len(all_actual)
    inference_time = result['mean_elapsed_sec']
    excluded_examples = result['excluded_examples']

    if excluded_examples:
        message = (
            f"{description}: excluded {excluded_examples} examples with an "
            "empty history or a target outside the training catalogue"
        )
        print(message)
        logger.info(message)
    repeat_message = (
        f"{description}: one fixed stochastic inference run, "
        f"seed={result['seeds'][0]}"
    )
    print(repeat_message)
    logger.info(repeat_message)
    latency_message = (
        f"{description} inference time per run: total {inference_time:.2f} sec, "
        f"avg {inference_time / num_users * 1000:.2f} ms per user"
    )
    print(latency_message)
    logger.info(latency_message)

    topN_list = args.metric_ks
    print(f"\n{description.capitalize()} results:")
    print(f"{'k':<5} {'recall':<12} {'ndcg':<12} {'mrr':<12} {'coverage':<12}")
    for i, k in enumerate(topN_list):
        print(
            f"{k:<5} {rec[i]:<12.6f} {ndcg[i]:<12.6f} "
            f"{mrr[i]:<12.6f} {cov[i]:<12.6f}"
        )
    if len(result['seeds']) > 1:
        print("Std across inference seeds:")
        for i, k in enumerate(topN_list):
            print(
                f"{k:<5} {rec_std[i]:<12.6f} {ndcg_std[i]:<12.6f} "
                f"{mrr_std[i]:<12.6f} {cov_std[i]:<12.6f}"
            )

    tracker = getattr(args, "experiment_tracker", None)
    if tracker is not None and description.lower() == "test":
        tracker.log_final_metrics(
            {k: {"recall": rec[i], "ndcg": ndcg[i], "mrr": mrr[i], "coverage": cov[i]}
             for i, k in enumerate(topN_list)},
            split="global_temporal_70_10_20",
            mask_seen=True,
            seed=args.random_seed,
            inference_total_sec=inference_time,
            inference_wall_total_sec=result['wall_elapsed_sec'],
            inference_repeats=len(result['seeds']),
            inference_seeds=result['seeds'],
            inference_metric_std={
                str(k): {
                    "recall": rec_std[i], "ndcg": ndcg_std[i],
                    "mrr": mrr_std[i], "coverage": cov_std[i],
                }
                for i, k in enumerate(topN_list)
            },
            n_users=num_users,
            maxlen=args.max_len,
            popularity_bias=recommendation_popularity(
                all_predicted, getattr(args, "train_item_popularity", {}), topN_list
            ),
        )
        tracker.close()

    if save_recs:
        import pandas as pd
        recs_df = pd.DataFrame({
            'user_id': list(range(len(all_actual))),
            'recommendations': all_predicted,
        })
        recs_df.to_csv('recommendations.csv', index=False)
        print(
            "Recommendations saved to recommendations.csv "
            f"using canonical inference seed {result['seeds'][0]}"
        )

def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    tracker = ExperimentTracker(args.dataset, "DiffuRec")
    args.experiment_tracker = tracker
    plotter = TrainingPlotter(
        save_dir=args.log_file + args.dataset,
        model_name=f"{args.description}_{time.strftime('%Y%m%d_%H%M%S')}",
        metrics=['loss', 'recall@10']
        )
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    candidate_mask = build_candidate_mask(
        args.coverage_candidate_items,
        args.item_num + 1,
        device,
    )
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    use_amp = getattr(args, 'amp', False) and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    
    selected_metrics = None
    selected_epoch = None
    best_selection_key = None
    best_model = None
    bad_count = 0
    candidate_items = set(args.coverage_candidate_items)
    candidate_items.discard(0)

    start_epoch = 0
    resumed_training = False
    resume_path = _resolve_resume_checkpoint(args)
    if resume_path is not None:
        checkpoint = _load_tuning_checkpoint(resume_path)
        checkpoint_signature = checkpoint.get('resume_signature')
        current_signature = _resume_signature(args)
        if checkpoint_signature != current_signature:
            raise ValueError(
                'The tuning checkpoint configuration does not match the current '
                f'arguments. Saved={checkpoint_signature}; current={current_signature}'
            )
        start_epoch = int(checkpoint['completed_epoch'])
        if start_epoch >= epochs:
            raise ValueError(
                f'Checkpoint already completed epoch {start_epoch}, but --epochs={epochs}'
            )
        model_joint.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        _move_optimizer_state(optimizer, device)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        scaler.load_state_dict(checkpoint.get('scaler_state_dict', {}))
        selected_metrics = checkpoint.get('selected_metrics')
        selected_epoch = checkpoint.get('selected_epoch')
        best_selection_key = checkpoint.get('best_selection_key')
        bad_count = int(checkpoint.get('bad_count', 0))
        if checkpoint.get('best_model_state_dict') is not None:
            best_model = copy.deepcopy(model_joint)
            best_model.load_state_dict(checkpoint['best_model_state_dict'])
        for metric_name, metric_data in checkpoint.get('plotter_data', {}).items():
            plotter.data[metric_name] = {
                'epoch': list(metric_data['epoch']),
                'value': list(metric_data['value']),
            }
        tracker.rows = copy.deepcopy(checkpoint.get('tracker_rows', []))
        _restore_rng_state(checkpoint['rng_state'])
        resumed_training = True
        resume_message = (
            f'Resumed tuning from {resume_path} after epoch {start_epoch}; '
            f'continuing through epoch {epochs}'
        )
        print(resume_message)
        logger.info(resume_message)

    for epoch_temp in range(start_epoch, epochs):
        completed_epoch = epoch_temp + 1
        print('Epoch: {}'.format(completed_epoch))
        logger.info('Epoch: {}'.format(completed_epoch))
        model_joint.train()

        if resumed_training:
            _prepare_resumed_epoch_loader(tra_data_loader)

        epoch_loss_sum = 0.0
        for index_temp, train_batch in enumerate(tqdm(
            tra_data_loader,
            desc=f"Epoch {completed_epoch:03d}/{epochs}",
            unit="batch",
        )):
            train_batch = [x.to(device, non_blocking=True) for x in train_batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis = model_joint(train_batch[0], train_batch[1], train_flag=True)
                loss_diffu_value = model_joint.loss_diffu_ce(diffu_rep, train_batch[1])  ## use this not above
                loss_all = loss_diffu_value
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_sum += loss_all.item()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
        average_loss = epoch_loss_sum / len(tra_data_loader)
        print("loss in epoch {}: {}".format(completed_epoch, average_loss))
        plotter.update(epoch=completed_epoch, loss=average_loss)
        tracker.log_epoch(completed_epoch, train_loss=average_loss)
        lr_scheduler.step()

        should_stop = False
        if completed_epoch % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            result = evaluate_stochastic_ranking(
                model_joint,
                val_data_loader,
                args,
                candidate_items,
                candidate_mask,
                use_amp=use_amp,
            )
            _, recalls, ndcgs, mrrs, covs = result['means']
            topN_list = args.metric_ks
            idx10 = topN_list.index(10) if 10 in topN_list else 0
            metrics_dict = {
                'Recall@10': recalls[idx10],
                'NDCG@10': ndcgs[idx10],
                'MRR@10': mrrs[idx10],
                'Coverage@10': covs[idx10],
            }
            selection_key = (
                metrics_dict['Recall@10'],
                metrics_dict['NDCG@10'],
                metrics_dict['MRR@10'],
                metrics_dict['Coverage@10'],
            )
            validation_message = (
                f"Validation after epoch {completed_epoch}: "
                f"recall@10={metrics_dict['Recall@10']:.6f}, "
                f"ndcg@10={metrics_dict['NDCG@10']:.6f}, "
                f"mrr@10={metrics_dict['MRR@10']:.6f}, "
                f"coverage@10={metrics_dict['Coverage@10']:.6f}; "
                f"mean over seeds={result['seeds']}"
            )
            print(validation_message)
            logger.info(validation_message)

            if best_selection_key is None or selection_key > best_selection_key:
                best_selection_key = selection_key
                selected_metrics = metrics_dict.copy()
                selected_epoch = completed_epoch
                bad_count = 0
                best_model = copy.deepcopy(model_joint)
                selection_message = (
                    f"Selected checkpoint updated: epoch={selected_epoch}, "
                    f"metrics={selected_metrics}"
                )
                print(selection_message)
                logger.info(selection_message)
            else:
                bad_count += 1

            plotter.update(
                epoch=completed_epoch,
                val_recall=recalls[idx10],
                val_ndcg=ndcgs[idx10],
                val_mrr=mrrs[idx10],
                val_coverage=covs[idx10],
            )
            tracker.log_epoch(completed_epoch, **{
                "val_recall@10": recalls[idx10],
                "val_ndcg@10": ndcgs[idx10],
                "val_mrr@10": mrrs[idx10],
                "val_coverage@10": covs[idx10],
            })
            plotter.plot(save=True, show=False, suffix=f'_epoch{completed_epoch}')
            if bad_count >= args.patience:
                should_stop = True
            # for key_temp, values_temp in metrics_dict.items():
            #     values_mean = values_temp  # уже число
            #     if values_mean > best_metrics_dict.get('Best_' + key_temp, -1):
            #         flag_update = 1
            #         bad_count = 0
            #         best_metrics_dict['Best_' + key_temp] = values_mean
            #         best_epoch['Best_epoch_' + key_temp] = epoch_temp
            
            # if flag_update == 0:
            #     bad_count += 1
            # else:
            #     print(best_metrics_dict)
            #     print(best_epoch)
            #     logger.info(best_metrics_dict)
            #     logger.info(best_epoch)
            #     best_model = copy.deepcopy(model_joint)
            
            # if bad_count >= args.patience:
            #     break
        
        # if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
        #     print('start predicting: ', datetime.datetime.now())
        #     logger.info('start predicting: {}'.format(datetime.datetime.now()))
        #     model_joint.eval()
        #     with torch.no_grad():
        #         metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        #         # metrics_dict_mean = {}
        #         for val_batch in val_data_loader:
        #             val_batch = [x.to(device) for x in val_batch]
        #             scores_rec, rep_diffu, _, _, _, _ = model_joint(val_batch[0], val_batch[1], train_flag=False)
        #             scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)    ### inner_production
        #             # scores_rec_diffu = model_joint.routing_rep_pre(rep_diffu)   ### routing_rep_pre
        #             metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1], metric_ks)
        #             for k, v in metrics.items():
        #                 metrics_dict[k].append(v)
                        
        #     for key_temp, values_temp in metrics_dict.items():
        #         values_mean = round(np.mean(values_temp) * 100, 4)
        #         if values_mean > best_metrics_dict['Best_' + key_temp]:
        #             flag_update = 1
        #             bad_count = 0
        #             best_metrics_dict['Best_' + key_temp] = values_mean
        #             best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    
        #     if flag_update == 0:
        #         bad_count += 1
        #     else:
        #         print(best_metrics_dict)
        #         print(best_epoch)
        #         logger.info(best_metrics_dict)
        #         logger.info(best_epoch)
        #         best_model = copy.deepcopy(model_joint)
        #     if bad_count >= args.patience:
        #         break
            
        if completed_epoch % TUNING_CHECKPOINT_INTERVAL == 0:
            checkpoint_payload = {
                'format_version': 1,
                'completed_epoch': completed_epoch,
                'resume_signature': _resume_signature(args),
                'model_state_dict': _state_dict_on_cpu(model_joint),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'selected_metrics': copy.deepcopy(selected_metrics),
                'selected_epoch': selected_epoch,
                'best_selection_key': best_selection_key,
                'best_model_state_dict': _state_dict_on_cpu(best_model),
                'bad_count': bad_count,
                'rng_state': _capture_rng_state(),
                'plotter_data': copy.deepcopy(dict(plotter.data)),
                'tracker_rows': copy.deepcopy(tracker.rows),
            }
            saved_path = save_tuning_checkpoint(
                checkpoint_payload, args, completed_epoch,
            )
            checkpoint_message = f'Tuning checkpoint saved: {saved_path}'
            print(checkpoint_message)
            logger.info(checkpoint_message)

        if should_stop:
            break

    plotter.plot(save=True, show=False, suffix='_final')
    if best_model is None:
        best_model = copy.deepcopy(model_joint)
        selected_epoch = completed_epoch
        selected_metrics = None
    final_selection_message = (
        f"Final validation selection: epoch={selected_epoch}, "
        f"metrics={selected_metrics}, rule=recall@10_then_ndcg_mrr_coverage"
    )
    print(final_selection_message)
    logger.info(final_selection_message)
    tracker.log_validation_selection(
        selected_epoch,
        selected_metrics,
        rule="recall@10_then_ndcg_mrr_coverage",
        inference_seeds=evaluation_seeds(args),
    )
    # if args.eval_interval > epochs:
    #     best_model = copy.deepcopy(model_joint)
    
    # # Тестирование на лучшей модели
    
    # top_100_item = []   # для diversity_measure, если нужно
    # with torch.no_grad():
    #     all_actual = []
    #     all_predicted = []
    #     start_time = time.time()
    #     for test_batch in test_data_loader:
    #         test_batch = [x.to(device) for x in test_batch]
    #         _, rep_diffu, _, _, _, _ = best_model(test_batch[0], test_batch[1], train_flag=False)
    #         scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)
    #         k_max = max(args.metric_ks)
    #         _, topk_indices = torch.topk(scores_rec_diffu, k=k_max, dim=-1)
    #         for i in range(len(test_batch[1])):
    #             all_actual.append([test_batch[1][i].item()])
    #             all_predicted.append(topk_indices[i].cpu().tolist())
    #         # Для diversity_measure (если нужно) собираем top-100
        #     if args.diversity_measure:
        #         _, top100 = torch.topk(scores_rec_diffu, k=100, dim=-1)
        #         top_100_item.append(top100.cpu())
        # inference_time = time.time() - start_time
        # num_users = len(all_actual)
        # print(f"Inference latency: total {inference_time:.2f} sec, avg {inference_time/num_users*1000:.2f} ms per user")
        # logger.info(f"Inference latency: total {inference_time:.2f} sec, avg {inference_time/num_users*1000:.2f} ms per user")
        # # Вычисляем метрики
        # precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
        #     all_actual, all_predicted, args.metric_ks, args.item_num
        # )
        # test_metrics_dict_mean = {}
        # for k, rec, nd, mrr, cov in zip(args.metric_ks, recalls, ndcgs, mrrs, covs):
        #     if k == 10:
        #         test_metrics_dict_mean[f'Recall@{k}'] = round(rec, 4)
        #         test_metrics_dict_mean[f'NDCG@{k}'] = round(nd, 4)
        #         test_metrics_dict_mean[f'MRR@{k}'] = round(mrr, 4)
        #         test_metrics_dict_mean[f'Coverage@{k}'] = round(cov, 4)

        
        # print('Test------------------------------------------------------')
        # logger.info('Test------------------------------------------------------')
        # print(test_metrics_dict_mean)
        # logger.info(test_metrics_dict_mean)


    # print(best_metrics_dict)
    # print(best_epoch)
    # logger.info(best_metrics_dict)
    # logger.info(best_epoch)

    # if args.diversity_measure:
    #     path_data = '../datasets/data/category/' + args.dataset +'/id_category_dict.pkl'
    #     with open(path_data, 'rb') as f:
    #         id_category_dict = pickle.load(f)
    #     id_top_100 = torch.cat(top_100_item, dim=0).tolist()
    #     category_list_100 = []
    #     for id_top_100_temp in id_top_100:
    #         category_temp_list = [] 
    #         for id_temp in id_top_100_temp:
    #             category_temp_list.append(id_category_dict[id_temp])
    #         category_list_100.append(category_temp_list)
    #     category_list_100.append(category_list_100)
    #     path_data_category = '../datasets/data/category/' + args.dataset +'/DiffuRec_top100_category.pkl'
    #     with open(path_data_category, 'wb') as f:
    #         pickle.dump(category_list_100, f)
            

    tracker.close()
    return best_model, None
    
