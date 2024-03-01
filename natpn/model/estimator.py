from __future__ import annotations
import json
import warnings
import pickle
import inspect
import logging
import tempfile
from abc import ABC
from pathlib import Path
from typing import Any, cast, Dict, List, Literal, Optional, Union
import torch
# from lightkit import BaseEstimator
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from natpn.datasets import DataModule, OutputType
from natpn.nn import CertaintyBudget, NaturalPosteriorEnsembleModel, NaturalPosteriorNetworkModel
from natpn.nn.encoder import (
    DeepImageEncoder,
    DenseDepthEncoder,
    ResnetEncoder,
    ShallowImageEncoder,
    TabularEncoder,
)
from natpn.nn.flow import MaskedAutoregressiveFlow, RadialFlow
from natpn.nn.output import CategoricalOutput, NormalOutput, PoissonOutput
from .lightning_module import NaturalPosteriorNetworkLightningModule
from .lightning_module_flow import NaturalPosteriorNetworkFlowLightningModule
from .lightning_module_ood import NaturalPosteriorNetworkOodTestingLightningModule

logger = logging.getLogger(__name__)

FlowType = Literal["radial", "maf"]
"""
A reference to a flow type that can be used with :class:`NaturalPosteriorNetwork`:

- `radial`: A :class:`~natpn.nn.flow.RadialFlow`.
- `maf`: A :class:`~natpn.nn.flow.MaskedAutoregressiveFlow`.
"""

EncoderType = Literal["tabular", "image-shallow", "image-deep", "resnet", "dense-depth"]
"""
A reference to an encoder class that can be used with :class:`NaturalPosteriorNetwork`:

- `tabular`: A :class:`~natpn.nn.encoder.TabularEncoder`.
- `image-shallow`: A :class:`~natpn.nn.encoder.ShallowImageEncoder`.
- `image-deep`: A :class:`~natpn.nn.encoder.DeepImageEncoder`.
- `resnet`: A :class:`~natpn.nn.encoder.ResnetEncoder`.
- `dense-depth`: A :class:`~natpn.nn.encoder.DenseDepthEncoder`.
"""


class NaturalPosteriorNetwork(ABC):
    """
    Estimator for the Natural Posterior Network and the Natural Posterior Ensemble.
    """

    #: The fitted model.
    model_: Union[NaturalPosteriorNetworkModel, NaturalPosteriorEnsembleModel]
    #: The input size of the model.
    input_size_: torch.Size
    #: The output type of the model.
    output_type_: OutputType
    #: The number of classes the model predicts if ``output_type_ == "categorical"``.
    num_classes_: Optional[int]

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        encoder: EncoderType = "tabular",
        flow: FlowType = "radial",
        flow_num_layers: int = 8,
        certainty_budget: CertaintyBudget = "normal",
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = False,
        entropy_weight: float = 1e-5,
        warmup_epochs: int = 3,
        finetune: bool = True,
        ensemble_size: Optional[int] = None,
        # max_epochs: int = 10,
        # accelerator: str = 'auto',
        # logger = None
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            latent_dim: The dimension of the latent space that the encoder should map to.
            encoder: The type of encoder to use which maps the input to the latent space.
            flow: The type of flow which produces log-probabilities from the latent
                representations.
            flow_num_layers: The number of layers to use for the flow. If ``flow`` is set to
                ``"maf"``, this sets the number of masked autoregressive layers. In between each
                of these layers, another batch normalization layer is added.
            certainty_budget: The certainty budget to use to scale the log-probabilities produced
                by the normalizing flow.
            dropout: The dropout probability to use for dropout layers in the encoder.
            learning_rate: The learning rate to use for training encoder, flow, and linear output
                layer. Applies to warm-up, actual training, and fine-tuning.
            learning_rate_decay: Whether to use a learning rate decay by reducing the learning rate
                when the validation loss plateaus.
            entropy_weight: The strength of the entropy regularizer for the Bayesian loss used for
                the main training procedure.
            warmup_epochs: The number of epochs to run warm-up for. Should be used if the latent
                space is high-dimensional and/or the normalizing flow is complex, i.e. consists of
                many layers.
            finetune: Whether to run fine-tuning after the main training loop. May be set to
                ``False`` to speed up the overall training time if the data is simple. Otherwise,
                it should be kept as ``True`` to improve out-of-distribution detection.
            ensemble_size: The number of NatPN models to ensemble for the final predictions. This
                constructs a Natural Posterior Ensemble which trains multiple NatPN models
                independently and combines their predictions via Bayesian combination. By default,
                this is set to ``None`` which does not create a NatPE.
            trainer_params: Additional parameters which are passed to the PyTorch Ligthning
                trainer. These parameters apply to all fitting runs as well as testing.
        """
        # super().__init__()

        # self.max_epochs = max_epochs
        # self.logger = logger
        # self.accelerator = accelerator

        # self.user_params = dict(
        #     max_epochs=self.max_epochs,
        #     accelerator=self.accelerator,
        #     logger=self.logger
        # )
        self.overwrite_params = dict(
            log_every_n_steps=1,
            enable_checkpointing=True,
            enable_progress_bar=True,
            devices=[0]
        )

        self.trainer_params = {
            **dict(
                logger=False,
                log_every_n_steps=1,
                enable_progress_bar=logger.getEffectiveLevel() <= logging.INFO,
                enable_checkpointing=logger.getEffectiveLevel() <= logging.DEBUG,
                enable_model_summary=logger.getEffectiveLevel() <= logging.DEBUG,
            ),
            **(trainer_params or {}),
            **(self.overwrite_params or {}),
        }

        self.latent_dim = latent_dim
        self.encoder = encoder
        self.flow = flow
        self.flow_num_layers = flow_num_layers
        self.certainty_budget: CertaintyBudget = certainty_budget
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.entropy_weight = entropy_weight
        self.warmup_epochs = warmup_epochs
        self.finetune = finetune
        self.ensemble_size = ensemble_size

    # ---------------------------------------------------------------------------------------------
    # RUNNING THE MODEL
    def trainer(self, **kwargs: Any) -> pl.Trainer:
        """
        Returns the trainer as configured by the estimator. Typically, this method is only called
        by functions in the estimator.

        Args:
            kwargs: Additional arguments that override the trainer arguments registered in the
                initializer of the estimator.

        Returns:
            A fully initialized PyTorch Lightning trainer.

        Note:
            This function should be preferred over initializing the trainer directly. It ensures
            that the returned trainer correctly deals with LightKit components that may be
            introduced in the future.
        """
        return pl.Trainer(**{**self.trainer_params, **kwargs})


    def fit(self, data: DataModule) -> NaturalPosteriorNetwork:
        """
        Fits the Natural Posterior Network with the provided data. Fitting sequentially runs
        warm-up (if ``self.warmup_epochs > 0``), the main training loop, and fine-tuning (if
        ``self.finetune == True``).

        Args:
            data: The data to fit the model with.

        Returns:
            The estimator whose ``model_`` property is set.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            if self.ensemble_size is None:
                model = self._init_model(
                    data.output_type,
                    data.input_size,
                    data.num_classes if data.output_type == "categorical" else 0,
                )
                self.model_ = self._fit_model(model, data, Path(tmp_dir))
            else:
                models = []
                for i in range(self.ensemble_size):
                    logger.info("Fitting model %d/%d...", i + 1, self.ensemble_size)
                    model = self._init_model(
                        data.output_type,
                        data.input_size,
                        data.num_classes if data.output_type == "categorical" else 0,
                    )
                    models.append(self._fit_model(model, data, Path(tmp_dir)))
                self.model_ = NaturalPosteriorEnsembleModel(models)

        # Assign additional fitted attributes
        self.input_size_ = data.input_size
        self.output_type_ = data.output_type
        try:
            self.num_classes_ = data.num_classes
        except NotImplementedError:
            self.num_classes_ = None

        # Return self
        return self

    def score(self, data: DataModule) -> Dict[str, float]:
        """
        Measures the model performance on the given data.

        Args:
            data: The data for which to measure the model performance.

        Returns:
            A dictionary mapping metrics to their values. This dictionary includes a measure of
            accuracy (`"accuracy"` for classification and `"rmse"` for regression) and a
            calibration measure (`"brier_score"` for classification and `"calibration"` for
            regression).
        """
        logger.info("Evaluating on test set...")
        module = NaturalPosteriorNetworkLightningModule(self.model_)
        out = self.trainer().test(module, data, verbose=False)
        return {k.split("/")[1]: v for k, v in out[0].items()}

    def score_ood_detection(self, data: DataModule) -> Dict[str, Dict[str, float]]:
        """
        Measures the model's ability to detect out-of-distribution data.

        Args:
            data: The data module which provides one or more datasets that contain test data along
                with out-of-distribution data.

        Returns:
            A nested dictionary which provides for multiple out-of-distribution datasets (first
            key) multiple metrics for measuring epistemic and aleatoric uncertainty.
        """
        results = {}
        for dataset, loader in data.ood_dataloaders().items():
            logger.info("Evaluating in-distribution vs. %s...", dataset)
            module = NaturalPosteriorNetworkOodTestingLightningModule(
                self.model_, logging_key=f"ood/{dataset}"
            )
            result = self.trainer().test(module, loader, verbose=False)
            results[dataset] = {k.split("/")[2]: v for k, v in result[0].items()}

        return results

    # ---------------------------------------------------------------------------------------------
    # PERSISTENCE

    @property
    def persistent_attributes(self) -> List[str]:
        return [k for k in self.__annotations__ if k != "model_"]

    def save(self, path) -> None:
        """Saves the estimator to the provided directory. It saves a file named
        ``estimator.pickle`` for the configuration of the estimator and
        additional files for the fitted model (if applicable). For more
        information on the files saved for the fitted model or for more
        customization, look at :meth:`get_params` and
        :meth:`lightkit.nn.Configurable.save`.

        Args:
            path: The directory to which all files should be saved.

        Note:
            This method may be called regardless of whether the estimator has already been fitted.

        Attention:
            If the dictionary returned by :meth:`get_params` is not JSON-serializable, this method
            uses :mod:`pickle` which is not necessarily backwards-compatible.
        """
        path = Path(path)
        assert not path.exists() or path.is_dir(), "Estimators can only be saved to a directory."

        path.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path)
        try:
            self.save_attributes(path)
        except:
            # In case attributes are not fitted, we just don't save them
            pass
    
    def get_params(self, deep: bool = True) -> dict[str, Any]:  # pylint: disable=unused-argument
        """
        Returns the estimator's parameters as passed to the initializer.

        Args:
            deep: Ignored. For Scikit-learn compatibility.

        Returns:
            The mapping from init parameters to values.
        """
        signature = inspect.signature(self.__class__.__init__)
        parameters = [p.name for p in signature.parameters.values() if p.name != "self"]
        return {p: getattr(self, p) for p in parameters}

    def save_parameters(self, path: Path) -> None:
        params = {
            k: (
                v
                if k != "trainer_params"
                else {kk: vv for kk, vv in cast(Dict[str, Any], v).items() if kk != "logger"}
            )
            for k, v in self.get_params().items()
        }
        data = json.dumps(params, indent=4)
        with (path / "params.json").open("w+") as f:
            f.write(data)

    def save_attributes(self, path: Path) -> None:
        """
        Saves the fitted attributes of this estimator. By default, it uses JSON and falls back to
        :mod:`pickle`. Subclasses should overwrite this method if non-primitive attributes are
        fitted.

        Typically, this method should not be called directly. It is called as part of :meth:`save`.

        Args:
            path: The directory to which the fitted attributed should be saved.

        Raises:
            NotFittedError: If the estimator has not been fitted.
        """
        if len(self.persistent_attributes) == 0:
            return

        attributes = {
            attribute: getattr(self, attribute) for attribute in self.persistent_attributes
        }
        try:
            data = json.dumps(attributes, indent=4)
            with (path / "attributes.json").open("w+") as f:
                f.write(data)
        except TypeError:
            warnings.warn(
                f"Failed to serialize fitted attributes of `{self.__class__.__name__}` to JSON. "
                "Falling back to `pickle`."
            )
            with (path / "attributes.pickle").open("wb+") as f:
                pickle.dump(attributes, f)
        torch.save(self.model_.state_dict(), path / "parameters.pt")

    def load_attributes(self, path: Path) -> None:
        """
        Loads the fitted attributes that are stored at the fitted path. If subclasses overwrite
        :meth:`save_attributes`, this method should also be overwritten.

        Typically, this method should not be called directly. It is called as part of :meth:`load`.

        Args:
            path: The directory from which the parameters should be loaded.

        Raises:
            FileNotFoundError: If the no fitted attributes have been stored.
        """
        json_path = path / "attributes.json"
        pickle_path = path / "attributes.pickle"

        if json_path.exists():
            with json_path.open() as f:
                self.set_params(json.load(f))
        else:
            with pickle_path.open("rb") as f:
                self.set_params(pickle.load(f))
        parameters = torch.load(path / "parameters.pt")
        if self.ensemble_size is None:
            model = self._init_model(self.output_type_, self.input_size_, self.num_classes_ or 0)
            model.load_state_dict(parameters)
            self.model_ = model
        else:
            model = NaturalPosteriorEnsembleModel(
                [
                    self._init_model(self.output_type_, self.input_size_, self.num_classes_ or 0)
                    for _ in range(self.ensemble_size)
                ]
            )
            model.load_state_dict(parameters)
            self.model_ = model

    def set_params(self, values: dict[str, Any]):
        """
        Sets the provided values on the estimator. The estimator is returned as well, but the
        estimator on which this function is called is also modified.

        Args:
            values: The values to set.

        Returns:
            The estimator where the values have been set.
        """
        for key, value in values.items():
            setattr(self, key, value)
        return self

    # ---------------------------------------------------------------------------------------------
    # UTILS

    def _fit_model(
        self, model: NaturalPosteriorNetworkModel, data: DataModule, tmp_dir: Path
    ) -> NaturalPosteriorNetworkModel:
        level = logging.getLogger("pytorch_lightning").getEffectiveLevel()

        # Run warmup
        if self.warmup_epochs > 0:
            warmup_module = NaturalPosteriorNetworkFlowLightningModule(
                model, learning_rate=self.learning_rate, early_stopping=False
            )

            # Get trainer and print information
            logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
            trainer = self.trainer(
                accumulate_grad_batches=data.gradient_accumulation_steps,
                # log_every_n_steps=1,
                # enable_progress_bar=True,
                enable_checkpointing=False,
                enable_model_summary=True,
                max_epochs=self.warmup_epochs,
                # accelerator = self.accelerator,
                # logger=self.logger
            )
            logging.getLogger("pytorch_lightning").setLevel(level)

            logger.info("Running warmup...")
            trainer.fit(warmup_module, data)

        # Run training
        trainer_checkpoint = ModelCheckpoint(tmp_dir / "training", monitor="val/loss", mode='min')

        logging.getLogger("pytorch_lightning").setLevel(
            logging.INFO if self.warmup_epochs == 0 else level
        )
        trainer = self.trainer(
            accumulate_grad_batches=data.gradient_accumulation_steps,
            # log_every_n_steps=1,
            # enable_progress_bar=True,
            # enable_checkpointing=True,
            enable_model_summary=self.warmup_epochs == 0,
            callbacks=[trainer_checkpoint],
            # max_epochs=self.max_epochs,
            # accelerator = self.accelerator,
            # logger=self.logger
        )
        logging.getLogger("pytorch_lightning").setLevel(level)

        logger.info("Running training...")
        train_module = NaturalPosteriorNetworkLightningModule(
            model,
            learning_rate=self.learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            entropy_weight=self.entropy_weight,
        )
        trainer.fit(train_module, data)

        best_module = NaturalPosteriorNetworkLightningModule.load_from_checkpoint(
            trainer_checkpoint.best_model_path
        )

        # Run fine-tuning
        if self.finetune:
            finetune_checkpoint = ModelCheckpoint(tmp_dir / "finetuning", monitor="val/log_prob", mode='max')
            trainer = self.trainer(
                accumulate_grad_batches=data.gradient_accumulation_steps,
                # log_every_n_steps=1,
                # enable_progress_bar=True,
                # enable_checkpointing=True,
                callbacks=[finetune_checkpoint],
                # max_epochs=self.max_epochs,
                # accelerator = self.accelerator,
                # logger=self.logger
            )

            logger.info("Running fine-tuning...")
            finetune_module = NaturalPosteriorNetworkFlowLightningModule(
                cast(NaturalPosteriorNetworkModel, best_module.model),
                learning_rate=self.learning_rate,
                learning_rate_decay=self.learning_rate_decay,
            )
            trainer.fit(finetune_module, data)

            # Return model
            return NaturalPosteriorNetworkFlowLightningModule.load_from_checkpoint(
                finetune_checkpoint.best_model_path
            ).model
        return cast(NaturalPosteriorNetworkModel, best_module.model)

    def _init_model(
        self, output_type: OutputType, input_size: torch.Size, num_classes: int
    ) -> NaturalPosteriorNetworkModel:
        # Initialize encoder
        if self.encoder == "tabular":
            assert len(input_size) == 1, "Tabular encoder only allows for one-dimensional inputs."
            encoder = TabularEncoder(
                input_size[0], [64] * 3, self.latent_dim, dropout=self.dropout
            )
        elif self.encoder == "image-shallow":
            encoder = ShallowImageEncoder(input_size, self.latent_dim, dropout=self.dropout)
        elif self.encoder == "image-deep":
            encoder = DeepImageEncoder(input_size, self.latent_dim, dropout=self.dropout)
        elif self.encoder == "resnet":
            assert len(input_size) == 3, "Resnet encoder requires three-dimensional inputs."
            encoder = ResnetEncoder(self.latent_dim, dropout=self.dropout)
        elif self.encoder == "dense-depth":
            assert input_size == torch.Size(
                [3, 640, 480]
            ), "DenseDepth encoder requires input of shape [3, 640, 480]."
            encoder = DenseDepthEncoder(self.latent_dim, dropout=self.dropout)
        else:
            raise NotImplementedError

        # Initialize flow
        if self.flow == "radial":
            flow = RadialFlow(self.latent_dim, num_layers=self.flow_num_layers)
        elif self.flow == "maf":
            flow = MaskedAutoregressiveFlow(self.latent_dim, num_layers=self.flow_num_layers)
        else:
            raise NotImplementedError

        # Initialize output
        if output_type == "categorical":
            output = CategoricalOutput(self.latent_dim, num_classes)
        elif output_type == "normal":
            output = NormalOutput(self.latent_dim)
        elif output_type == "poisson":
            output = PoissonOutput(self.latent_dim)
        else:
            raise NotImplementedError

        return NaturalPosteriorNetworkModel(
            self.latent_dim,
            encoder=encoder,
            flow=flow,
            output=output,
            certainty_budget=self.certainty_budget,
        )
