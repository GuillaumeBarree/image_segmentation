"""Define useful function for the package"""
from __future__ import annotations

import os
from shutil import copyfile
from typing import Sequence

import tensorflow as tf
from image_segmentation.common import PROJECT_ROOT
from image_segmentation.models.models import get_compiled_model
from image_segmentation.models.tf_unet.unet import unet_constructor
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict


def generate_unique_logpath(logdir, raw_run_name):
    """Verify if the path already exist
    Args:
        logdir (str): path to log dir
        raw_run_name (str): name of the file
    Returns:
        str: path to the output file
    """
    i = 0
    while True:
        run_name = raw_run_name + '_' + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def save_config_file(save_dir: str, load_weights: str | None) -> None:
    if load_weights is not None:
        copyfile(
            os.path.join(os.path.dirname(load_weights), 'config_file.yaml'),
            os.path.join(save_dir, 'config_file.yaml'),
        )
        copyfile(
            load_weights,
            os.path.join(save_dir, 'model_start_weights.hdf5'),
        )
    else:
        copyfile(
            os.path.join(PROJECT_ROOT, 'conf/nn/default.yaml'),
            os.path.join(save_dir, 'config_file.yaml'),
        )


def edit_callback_paths(cfg: DictConfig, save_dir: str):
    save_dir_name = os.path.split(save_dir)[1]
    with open_dict(cfg):
        cfg.train.callbacks.model_checkpoint.filepath = os.path.join(save_dir, 'best_model.hdf5')
        cfg.train.callbacks.tensorboard.log_dir = os.path.join(
            PROJECT_ROOT,
            'tensorboard',
            save_dir_name,
        )


def get_model(cfg: DictConfig, load_weights: str | None):

    if load_weights is not None:
        cfg.nn = OmegaConf.load(os.path.join(os.path.dirname(load_weights), 'config_file.yaml'))

    # Instantiate useful constant
    original_height = cfg.nn.data.resize.height
    original_width = cfg.nn.data.resize.width
    original_img_size = (original_height, original_width)

    unet_model = unet_constructor(
        image_size=original_img_size,
        blocks_config=cfg.nn.unet.blocks,
        **cfg.nn.unet.structure,
    )

    if load_weights is not None:
        print('Model weights have been loaded successfully!')
        unet_model.load_weights(load_weights)

    # Instantiate UNet model
    original_output_shape = unet_model.get_layer('final_block').output_shape[1:3]

    with open_dict(cfg):
        cfg.nn.module.custom_loss.output_shape = original_output_shape

    unet_model = get_compiled_model(
        model=unet_model,
        optimizer_config=cfg.nn.module.optimizer,
        custom_loss_config=cfg.nn.module.custom_loss,
        metrics_config=cfg.nn.module.metrics,
    )

    return unet_model


def save_results(test_paths: Sequence[str], save_dir, tests_imgs, predictions) -> None:

    save_dir = os.path.join(save_dir, 'predictions')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, img in enumerate(tests_imgs):
        img_name = os.path.split(test_paths[i])[1]

        tf.keras.utils.save_img(os.path.join(save_dir, img_name), img, file_format='png')
        tf.keras.utils.save_img(os.path.join(save_dir, f'{img_name.split(".")[0]}_predict.png'), predictions[i], file_format='png')
    print(f'Predictions have been save here {save_dir}')
