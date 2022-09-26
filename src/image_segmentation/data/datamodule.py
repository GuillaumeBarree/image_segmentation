from __future__ import annotations

import os
import random
import sys
from typing import Sequence

import tensorflow as tf
from image_segmentation.common import DATA_LOADER_SEED
from image_segmentation.common import PROJECT_ROOT
from image_segmentation.data.utils import Augment
from image_segmentation.data.utils import load_test_images
from image_segmentation.data.utils import load_train_valid_images
from image_segmentation.data.utils import Resize
from imutils import paths
from omegaconf import DictConfig

random.seed(DATA_LOADER_SEED)


class DataModule:
    """This DataModule standardizes the training, val, test splits, data preparation and transforms. The main
    advantage is consistent data splits, data preparation and transforms across models."""

    def __init__(
        self, data_path: DictConfig,
        batch_size: DictConfig,
        resize: DictConfig,
        stage: str | None,
        data_augmentation: DictConfig,
        val_percentage: float,
    ) -> None:
        self.data_path = data_path
        self.batch_size = batch_size
        self.resize = resize
        self.stage = stage
        self.data_augmentation = data_augmentation
        self.val_percentage = val_percentage

        self.train_paths: Sequence[str] | None = None
        self.val_paths: Sequence[str] | None = None
        self.test_paths: Sequence[str] | None = None

        self.train_dataset: tf.data.Dataset | None = None
        self.val_dataset: tf.data.Dataset | None = None
        self.test_dataset: tf.data.Dataset | None = None

    def prepare_data(self) -> None:
        """This function aims to download data if necessary.
        Here, data is already downloaded thus it only checks if the data path exists.
        """
        if not os.path.isdir(os.path.join(PROJECT_ROOT, self.data_path, 'train')):
            print('Path to data as not been set properly')
            sys.exit()

    def get_image_list_from_path(
        self,
        is_splitting_necessary: bool = True,
    ) -> tuple[Sequence[str], Sequence[str], Sequence[str]]:
        """Retrieve all image/mask paths from

        Args:
            is_splitting_necessary (bool, optional): Is it necessary to devide the training set into train and val.
            If validation folder does not exist, thus it is necessary to split the training folder. Defaults to True.

        Returns:
            Tuple[Sequence[str], Sequence[str], Sequence[str]]: list of (image, mask) paths for train, validation and test.
        """
        if is_splitting_necessary:
            train_val_paths = list(paths.list_images(os.path.join(PROJECT_ROOT, self.data_path, 'train/image')))
            random.shuffle(train_val_paths)

            train_paths = train_val_paths[:-int(len(train_val_paths)*self.val_percentage)]
            train_paths = [(path, path.replace('/image/', '/label/')) for path in train_paths]

            val_paths = train_val_paths[-int(len(train_val_paths)*self.val_percentage):]
            val_paths = [(path, path.replace('/image/', '/label/')) for path in val_paths]

        else:
            train_paths = list(paths.list_images(os.path.join(PROJECT_ROOT, self.data_path, 'train/image')))
            train_paths = [(path, path.replace('/image/', '/label/')) for path in train_paths]

            val_paths = list(paths.list_images(os.path.join(PROJECT_ROOT, self.data_path, 'valid/image')))
            val_paths = [(path, path.replace('/image/', '/label/')) for path in val_paths]

        test_paths = list(paths.list_images(os.path.join(PROJECT_ROOT, self.data_path, 'test')))

        return train_paths, val_paths, test_paths

    def setup(self) -> None:
        """Instanciate train, validation and test dataset.
        """
        # Create our train and valid Dataset
        if (self.stage is None or self.stage == 'fit') or (self.train_dataset is None and self.val_dataset is None):
            # Find if it exists a validation folder
            is_splitting_necessary = not os.path.isdir(os.path.join(PROJECT_ROOT, self.data_path, 'valid'))

            self.train_paths, self.val_paths, _ = self.get_image_list_from_path(is_splitting_necessary=is_splitting_necessary)

            self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_paths)
            self.val_dataset = tf.data.Dataset.from_tensor_slices(self.val_paths)

        if self.stage is None or self.stage == 'test':
            _, _, self.test_paths = self.get_image_list_from_path(is_splitting_necessary=is_splitting_necessary)

            self.test_dataset = tf.data.Dataset.from_tensor_slices(self.test_paths)

    def train_dataloader(self) -> tf.data.Dataset:
        """Apply transformations to the train dataset.
        - Load images and mask
        - Apply data augmentation
        - Create batch ...

        Returns:
            tf.data.Dataset: Train dataset
        """
        if self.train_dataset is None:
            self.setup()

        resize = Resize(resize=self.resize)
        augment = Augment(data_aug=self.data_augmentation)
        return (
            self.train_dataset
                .shuffle(len(self.train_paths))
                .map(load_train_valid_images, num_parallel_calls=tf.data.AUTOTUNE)
                .map(resize)
                .map(augment)
                .cache()
                .batch(self.batch_size.train)
                .prefetch(tf.data.AUTOTUNE)
        )

    def val_dataloader(self) -> tf.data.Dataset:
        """Apply transformations to the train dataset.
        - Load images and mask
        - Apply data augmentation
        - Create batch ...

        Returns:
            tf.data.Dataset: Valid dataset
        """
        if self.val_dataset is None:
            self.setup()

        resize = Resize(resize=self.resize)
        return (
            self.val_dataset
                .map(load_train_valid_images, num_parallel_calls=tf.data.AUTOTUNE)
                .map(resize)
                .cache()
                .batch(self.batch_size.val)
                .prefetch(tf.data.AUTOTUNE)
        )

    def test_dataloader(self) -> tf.data.Dataset:
        """Apply transformations to the test dataset.
        - Load images
        - Create batch ...

        Returns:
            tf.data.Dataset: Test dataset
        """
        if self.test_dataset is None:
            self.setup()

        resize = Resize(resize=self.resize)
        return (
            self.test_dataset
                .map(load_test_images, num_parallel_calls=tf.data.AUTOTUNE)
                .map(resize)
                .cache()
                .batch(self.batch_size.test)
                .prefetch(tf.data.AUTOTUNE)
        )
