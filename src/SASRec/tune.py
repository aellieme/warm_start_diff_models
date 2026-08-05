import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_tools.experiment_tracking import (
    ExperimentTracker,
    capture_rng_state,
    checkpoint_due,
    restore_rng_state,
    save_torch_checkpoint,
)
from load_evaluate_pipeline import prepare_data_and_description
from training import (
    prepare_sasrec_model,
    train_sasrec_epoch,
    validate_last_item_metrics,
)


FIXED_PRESETS = {
    "ml-1m": {
        "hidden_units": 256,
        "dropout_rate": 0.2,
        "num_blocks": 2,
        "num_heads": 2,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "l2_emb": 1e-4,
    },
    "amazon_Baby": {
        "hidden_units": 128,
        "dropout_rate": 0.3,
        "num_blocks": 2,
        "num_heads": 2,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "l2_emb": 1e-4,
    },
    "amazon_Beauty": {
        "hidden_units": 128,
        "dropout_rate": 0.3,
        "num_blocks": 2,
        "num_heads": 2,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "l2_emb": 1e-4,
    },
    "amazon_Toys_and_Games": {
        "hidden_units": 128,
        "dropout_rate": 0.3,
        "num_blocks": 2,
        "num_heads": 2,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "l2_emb": 1e-4,
    },
    "amazon_Sports_and_Outdoors": {
        "hidden_units": 128,
        "dropout_rate": 0.3,
        "num_blocks": 2,
        "num_heads": 2,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "l2_emb": 1e-4,
    },
}
SELECTION_RULE = "recall@10_then_ndcg_mrr_coverage"


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def selection_key(metrics):
    return tuple(
        float(metrics[name]) for name in ("recall", "ndcg", "mrr", "coverage")
    )


def build_config(args):
    return {
        **FIXED_PRESETS[args.dataset],
        "num_epochs": args.max_epochs,
        "maxlen": args.maxlen,
        "sampler_seed": args.seed,
        "manual_seed": args.seed,
    }


def checkpoint_payload(model, config, data_description, data_index=None):
    return {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "config": dict(config),
        "data_description": data_description,
        "data_index": data_index,
        "pad_token": model.pad_token,
        "item_num": model.item_num,
        "candidate_items": model.training_candidate_mask.nonzero(
            as_tuple=False
        ).squeeze(-1).cpu().tolist(),
    }


def select_validation_checkpoint(
    config,
    train_data,
    val_data,
    data_description,
    data_index,
    tracker,
    patience,
    resume_checkpoint=None,
):
    fix_random_seed(config["manual_seed"])
    model, sampler, n_batches, criterion, optimizer = prepare_sasrec_model(
        config, train_data, data_description
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_key = None
    best_metrics = None
    best_epoch = None
    best_path = tracker.run_dir / "best_validation.pt"
    epochs_without_improvement = 0
    start_epoch = 1

    if resume_checkpoint:
        resume = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["next_epoch"])
        best_key = resume.get("best_key")
        best_metrics = resume.get("best_metrics")
        best_epoch = resume.get("best_epoch")
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
        restore_rng_state(resume.get("rng_state"))
        if resume.get("best_checkpoint"):
            best_path = Path(resume["best_checkpoint"])
        print(f"Resuming from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        losses = train_sasrec_epoch(
            model,
            n_batches,
            config["l2_emb"],
            sampler,
            optimizer,
            criterion,
            device,
        )
        metrics = validate_last_item_metrics(
            model, val_data, train_data, data_description, topn=10
        )
        mean_loss = float(np.mean(losses))
        current_key = selection_key(metrics)
        tracker.log_epoch(
            epoch,
            train_loss=mean_loss,
            **{
                "val_recall@10": metrics["recall"],
                "val_ndcg@10": metrics["ndcg"],
                "val_mrr@10": metrics["mrr"],
                "val_coverage@10": metrics["coverage"],
            },
        )
        print(
            f"Epoch {epoch}/{config['num_epochs']}: loss={mean_loss:.4f}, "
            f"Recall@10={metrics['recall']:.4f}, NDCG@10={metrics['ndcg']:.4f}, "
            f"MRR@10={metrics['mrr']:.4f}, Coverage@10={metrics['coverage']:.4f}",
            flush=True,
        )

        if best_key is None or current_key > best_key:
            best_key = current_key
            best_metrics = metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            save_torch_checkpoint(
                checkpoint_payload(model, config, data_description, data_index),
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if checkpoint_due(epoch - 1, config["num_epochs"]):
            resume_path = tracker.run_dir / "resume.pt"
            payload = checkpoint_payload(model, config, data_description, data_index)
            payload.update({
                "optimizer_state_dict": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "best_key": best_key,
                "best_metrics": best_metrics,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "best_checkpoint": str(best_path),
                "rng_state": capture_rng_state(),
            })
            save_torch_checkpoint(payload, resume_path)

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    tracker.log_validation_selection(
        best_epoch,
        {
            "Recall@10": best_metrics["recall"],
            "NDCG@10": best_metrics["ndcg"],
            "MRR@10": best_metrics["mrr"],
            "Coverage@10": best_metrics["coverage"],
        },
        rule=SELECTION_RULE,
        checkpoint=str(best_path),
        config=config,
        seed=config["manual_seed"],
        split="global_temporal_70_10_20",
        mask_seen=True,
    )
    (tracker.run_dir / "tuning_result.json").write_text(
        json.dumps(
            {
                "selection_rule": SELECTION_RULE,
                "fixed_config": config,
                "selected_epoch": best_epoch,
                "validation_metrics": best_metrics,
                "checkpoint": str(best_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return best_epoch, best_metrics, best_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m", choices=FIXED_PRESETS)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    fix_random_seed(args.seed)
    (
        train_data,
        val_data,
        _,
        _,
        data_index,
        data_description,
        _,
        _,
        _,
        _,
        _,
    ) = prepare_data_and_description(args.dataset)

    config = build_config(args)
    selection_tracker = ExperimentTracker(
        args.dataset, "SASRec", maxlen=args.maxlen, run_type="tuning"
    )
    selected_epoch, validation_metrics, selected_checkpoint = (
        select_validation_checkpoint(
            config,
            train_data,
            val_data,
            data_description,
            data_index,
            selection_tracker,
            args.patience,
            args.resume_checkpoint,
        )
    )
    selection_tracker.close()

    print("Fixed hyperparameters:", FIXED_PRESETS[args.dataset])
    print("Selected epochs:", selected_epoch)
    print("Validation metrics:", validation_metrics)
    print("Selected checkpoint:", selected_checkpoint)


if __name__ == "__main__":
    main()
