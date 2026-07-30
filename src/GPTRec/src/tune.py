import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from run_train_predict import (
    create_dataloaders,
    create_model,
    is_relevance_aggregation,
    prepare_data,
    run_relevance_aggregation_by_k,
    select_relevance_aggregation_temperature,
    training,
)
from experiment_tools.experiment_tracking import ExperimentTracker


def run_trial(config, tracker=None):
    if config.get("final_train", False):
        raise ValueError("tune.py cannot run final training or test evaluation")
    if not is_relevance_aggregation(config):
        raise ValueError("tune.py requires generation=true and relevance_aggregation")

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tracker = tracker or ExperimentTracker(
        config.dataset_name,
        str(config.model),
        maxlen=int(config.dataset.max_length),
        run_type="tuning",
    )
    train, validation, _, item_count = prepare_data(config)
    train_loader, eval_loader = create_dataloaders(train, validation, config)
    model = create_model(config, item_count=item_count)
    trainer, seqrec_module = training(
        model, train_loader, eval_loader, config, tracker=tracker
    )

    select_relevance_aggregation_temperature(
        trainer,
        seqrec_module,
        train,
        validation,
        config,
        output_dir=tracker.run_dir,
    )
    validation_last = (
        validation.sort_values("time_idx")
        .groupby("user_id")
        .last()
        .reset_index()
    )
    _, _, metrics, inference_seconds = run_relevance_aggregation_by_k(
        trainer,
        seqrec_module,
        train,
        validation,
        validation_last,
        train,
        config,
        "val_last",
        output_dir=tracker.run_dir,
    )
    selection_k = int(config.get("ra_temperature_selection_k", 10))
    objective = float(metrics[f"val_last_recall@{selection_k}"])
    selected_epoch = int(
        getattr(trainer, "selected_epoch", trainer.current_epoch + 1)
    )
    selected_checkpoint = str(
        getattr(trainer, "selected_checkpoint", "")
    )

    result_path = Path(tracker.run_dir) / "tuning_result.json"
    result_path.write_text(
        json.dumps(
            {
                "objective": f"val_last_recall@{selection_k}",
                "value": objective,
                "ra_temperature": float(config.generation_params.temperature),
                "model_params": OmegaConf.to_container(
                    config.model_params, resolve=True
                ),
                "learning_rate": float(config.seqrec_module.lr),
                "epochs": int(trainer.current_epoch) + 1,
                "validation_metrics": {
                    key: float(value) for key, value in metrics.items()
                },
                "validation_inference_seconds": inference_seconds,
                "selected_epoch": selected_epoch,
                "selected_checkpoint": selected_checkpoint,
                "seed": seed,
                "split": "global_temporal_70_10_20",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    OmegaConf.save(config, Path(tracker.run_dir) / "config.yaml")
    tracker.log_validation_selection(
        selected_epoch,
        {
            f"recall@{selection_k}": objective,
            "ra_temperature": float(config.generation_params.temperature),
        },
        rule=f"max_validation_recall@{selection_k}",
        checkpoint=selected_checkpoint,
        decoding_strategy="relevance_aggregation",
        num_return_sequences=int(
            config.generation_params.num_return_sequences
        ),
    )
    top_k = [int(k) for k in config.evaluator.top_k]
    tracker.plot_metrics_by_k({
        k: {
            "recall": metrics.get(f"val_last_recall@{k}", 0.0),
            "ndcg": metrics.get(f"val_last_ndcg@{k}", 0.0),
            "mrr": metrics.get(f"val_last_mrr@{k}", 0.0),
            "coverage": metrics.get(f"val_last_coverage@{k}", 0.0),
        }
        for k in top_k
    })
    tracker.close()
    return objective


@hydra.main(version_base=None, config_path="configs", config_name="GPT_Optuna")
def main(config):
    return run_trial(config)


if __name__ == "__main__":
    main()
