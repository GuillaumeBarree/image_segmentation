"""This module contains useful fonction for the class DataModule."""
from __future__ import annotations

from functools import reduce
from typing import Any
from typing import Callable

import hydra
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


def load_train_valid_images(data_path: tuple[str, str]) -> tuple[EagerTensor, EagerTensor]:
    """Read the image and mask from the disk, decode it, convert the data
    type to floating points.

    Args:
        data_path (Tuple[str, str]): path to image and its associated mask

    Returns:
        Tuple[EagerTensor, EagerTensor]: image and mask tensors
    """
    img = tf.io.read_file(data_path[0])
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    mask = tf.io.read_file(data_path[1])
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)

    return img, mask


def load_test_images(img_path: str) -> EagerTensor:
    """Read the image from the disk, decode it, convert the data
    type to floating points.

    Args:
        img_path (str): path to image

    Returns:
        EagerTensor: image tensor
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    return img


class Augment(tf.keras.layers.Layer):
    """Personal data Augmentation class.

    The purpose of this class is to apply a series of transformations to images
    and their associated masks from a configuration file.
    """

    def __init__(self, data_aug):
        super().__init__()
        self.data_augmentation = data_aug

        # Use instantiate function from hydra for simplicity
        self.augment_inputs = [hydra.utils.instantiate(self.data_augmentation[key]) for key in self.data_augmentation.keys()]
        self.augment_masks = [hydra.utils.instantiate(self.data_augmentation[key]) for key in self.data_augmentation.keys()]

    @staticmethod
    def composite_function(*func):
        """Compose functions"""
        def compose(f: Callable[[Any], Any], g: Callable[[Any], Any]) -> Callable[[Any], Any]:
            """Two functions to compose

            Args:
                f (Callable): first function
                g (Callable): second function

            Returns:
                Callable: g(f(x))
            """
            return lambda x: g(f(x))
        return reduce(compose, func, lambda x: x)

    def call(self, inputs: EagerTensor, masks: EagerTensor) -> tuple[EagerTensor, EagerTensor]:
        """Apply all transformations to the image and its mask

        Args:
            inputs (EagerTensor): raw image
            masks (EagerTensor): mask associated to the image

        Returns:
            Tuple[EagerTensor, EagerTensor]: transformed image and mask
        """
        inputs = self.composite_function(*self.augment_inputs)(inputs)
        masks = self.composite_function(*self.augment_masks)(masks)
        return inputs, masks
