from __future__ import annotations

from typing import Sequence

import hydra
import tensorflow as tf
from omegaconf import DictConfig
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback


def get_compiled_model(
    model: Model,
    optimizer_config: DictConfig,
    custom_loss_config: DictConfig,
    metrics_config: DictConfig,
) -> Model:
    loss = CustomLoss(
        original_output_size=custom_loss_config.output_shape,
        loss_function=custom_loss_config.loss,
        mode=custom_loss_config.mode,
    )
    optimizer = hydra.utils.instantiate(optimizer_config)
    metrics = get_metrics(metrics=metrics_config)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return model


def get_metrics(metrics: DictConfig) -> Sequence[keras.metrics.Metric]:
    return [hydra.utils.instantiate(metrics[metric]) for metric in metrics.keys()]


class CustomLoss(keras.losses.Loss):
    def __init__(
        self,
        original_output_size: tuple[int, int],
        loss_function: DictConfig,
        mode: str = 'mirror',
        name='CustomLoss',
    ):
        super().__init__(name=name)
        self.original_output_size = original_output_size
        self.mode = mode
        self.loss_function = hydra.utils.instantiate(loss_function)

    def call(self, y_true, y_pred):

        if self.mode == 'mirror':
            return self.loss_function(y_true=y_true, y_pred=y_pred)

        else:
            ratio = self.original_output_size[0]/y_true.shape[1]
            y_true = tf.image.central_crop(y_true, ratio)
            y_pred = tf.image.central_crop(y_pred, ratio)

            return self.loss_function(y_true=y_true, y_pred=y_pred)


def build_callbacks(callbacks: DictConfig) -> list[Callback]:
    """Instantiate the callbacks given their configuration.
    Args:
        cfg: a list of callbacks instantiable configuration
    Returns:
        the complete list of callbacks to use
    """
    return [hydra.utils.instantiate(callbacks[callback]) for callback in callbacks.keys()]
