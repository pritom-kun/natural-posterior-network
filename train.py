# pylint: disable=missing-function-docstring
import logging
import os
import tempfile
from pathlib import Path
from typing import cast, Dict, Optional
# import click
import argparse
import pytorch_lightning as pl
import torch
# from lightkit.utils import PathType
from pytorch_lightning.loggers import WandbLogger
from wandb.wandb_run import Run
from natpn import NaturalPosteriorNetwork, suppress_pytorch_lightning_logs
from natpn.model import EncoderType, FlowType
from natpn.nn import CertaintyBudget
from natpn.datasets import DATASET_REGISTRY

logger = logging.getLogger("natpn")

def main(
    dataset: str,
    seed: Optional[int],
    data_path,
    output_path,
    experiment: Optional[str],
    latent_dim: int,
    flow_type: FlowType,
    flow_layers: int,
    certainty_budget: CertaintyBudget,
    ensemble_size: Optional[int],
    learning_rate: float,
    use_learning_rate_decay: bool,
    max_epochs: int,
    entropy_weight: float,
    warmup_epochs: int,
    run_finetuning: bool,
):
    """
    Trains the Natural Posterior Network or an ensemble thereof on a single dataset and evaluates
    its performance.
    """

    logging.getLogger("natpn").setLevel(logging.INFO)
    suppress_pytorch_lightning_logs()

    # Fix randomness
    pl.seed_everything(seed)
    logger.info("Using seed %s.", os.getenv("PL_GLOBAL_SEED"))

    # Initialize logger if needed
    if experiment is not None:
        remote_logger = WandbLogger()
        cast(Run, remote_logger.experiment).config.update(
            {
                "seed": os.getenv("PL_GLOBAL_SEED"),
                "dataset": dataset,
                "latent_dim": latent_dim,
                "flow_type": flow_type,
                "flow_layers": flow_layers,
                "certainty_budget": certainty_budget,
                "ensemble_size": ensemble_size,
                "learning_rate": learning_rate,
                "use_learning_rate_decay": use_learning_rate_decay,
                "max_epochs": max_epochs,
                "entropy_weight": entropy_weight,
                "warmup_epochs": warmup_epochs,
                "run_finetuning": run_finetuning,
            }
        )
    else:
        remote_logger = None

    # Initialize data
    dm = DATASET_REGISTRY[dataset](data_path, seed=int(os.getenv("PL_GLOBAL_SEED") or 0))

    # Initialize estimator
    encoder_map: Dict[str, EncoderType] = {
        "concrete": "tabular",
        "sensorless-drive": "tabular",
        "bike-sharing-normal": "tabular",
        "bike-sharing-poisson": "tabular",
        "mnist": "image-shallow",
        "fashion-mnist": "image-shallow",
        "cifar10": "image-deep",
        "cifar100": "resnet",
        "nyu-depth-v2": "dense-depth",
    }
    cpu_datasets = {
        "concrete",
        "sensorless-drive",
        "bike-sharing-normal",
        "bike-sharing-poisson",
    }

    estimator = NaturalPosteriorNetwork(
        latent_dim=latent_dim,
        encoder=encoder_map[dataset],
        flow=flow_type,
        flow_num_layers=flow_layers,
        certainty_budget=certainty_budget,
        learning_rate=learning_rate,
        learning_rate_decay=use_learning_rate_decay,
        entropy_weight=entropy_weight,
        warmup_epochs=warmup_epochs,
        finetune=run_finetuning,
        ensemble_size=ensemble_size,
        trainer_params=dict(
            max_epochs=max_epochs,
            logger=remote_logger,
            accelerator='gpu' if dataset not in cpu_datasets else 'cpu',
        )
    )

    # Run training
    estimator.fit(dm)
    # estimator.load_attributes(Path('saved_models/natpn'))

    # Evaluate model
    scores = estimator.score(dm)
    ood_scores = estimator.score_ood_detection(dm)

    # Print scores
    logger.info("Test scores:")
    for key, value in scores.items():
        logger.info("  %s: %.2f", key, value * 100 if key != "rmse" else value)

    logger.info("OOD detection scores:")
    for ood_dataset, metrics in ood_scores.items():
        logger.info("in-distribution vs. '%s'...", ood_dataset)
        for key, value in metrics.items():
            logger.info("  %s: %.2f", key, value * 100)

    # Save model if required
    if output_path is not None:
        estimator.save(output_path)
    if remote_logger is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            estimator.save(tmpdir)
            cast(Run, remote_logger.experiment).log_artifact(tmpdir, name=dataset, type="model")

    logger.info("Done 🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--dataset",
    # type=click.Choice(list(DATASET_REGISTRY.keys())),
    required=True,
    help="The dataset to train and evaluate on.",
    )
    # GLOBAL CONFIGURATION
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A fixed seed to reproduced experiments.",
    )
    parser.add_argument(
        "--data_path",
        # type=str,
        default="./opt/data/natpn/",
        help="The directory where input data is stored.",
    )
    parser.add_argument(
        "--output_path",
        # type=str,
        default="./saved_models/natpn/",
        help="The local directory where the final model should be stored. Only uploaded "
        "or discarded if not provided.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="If provided, tracks the run using Weights & Biases and uploads the trained model.",
    )

    parser.add_argument(
    "--latent_dim",
    default=16,
    help="The dimension of the model's latent space.",
    )
    parser.add_argument(
        "--flow_type",
        # type=click.Choice(["radial", "maf"]),
        default="radial",
        help="The type of normalizing flow to use.",
    )
    parser.add_argument(
        "--flow_layers",
        default=8,
        help="The number of sequential normalizing flow transforms to use.",
    )
    parser.add_argument(
        "--certainty_budget",
        # type=click.Choice(["constant", "exp-half", "exp", "normal"]),
        default="normal",
        help="The certainty budget to allocate in the latent space.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        help="The number of NatPN models to ensemble for NatPE. Disabled if set to None (default).",
    )
    # -------------------------------------------------------------------------------------------------
    # TRAINING
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        help="The learning rate for the Adam optimizer for both training and fine-tuning.",
    )
    parser.add_argument(
        "--use_learning_rate_decay",
        default=False,
        help="Whether to decay the learning rate if the validation loss plateaus for some time.",
    )
    parser.add_argument(
        "--max_epochs",
        default=200,
        help="The maximum number of epochs to run both training and fine-tuning for.",
    )
    parser.add_argument(
        "--entropy_weight",
        default=1e-5,
        help="The weight for the entropy regularizer.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=3,
        help="The number of warm-up epochs to run prior to training.",
    )
    parser.add_argument(
        "--run_finetuning",
        default=True,
        help="Whether to run fine-tuning after training.",
    )

    args = parser.parse_args()
    print(args)
    # hparams = Hyperparameters(**vars(args))

    main(
        dataset=args.dataset,
        seed=args.seed,
        data_path=args.data_path,
        output_path=args.output_path,
        experiment=args.experiment,
        latent_dim=args.latent_dim,
        flow_type=args.flow_type,
        flow_layers=args.flow_layers,
        certainty_budget=args.certainty_budget,
        ensemble_size=args.ensemble_size,
        learning_rate=args.learning_rate,
        use_learning_rate_decay=args.use_learning_rate_decay,
        max_epochs=args.max_epochs,
        entropy_weight=args.entropy_weight,
        warmup_epochs=args.warmup_epochs,
        run_finetuning=args.run_finetuning
    )
