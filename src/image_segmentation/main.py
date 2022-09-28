from __future__ import annotations

import os
import sys

import hydra
import tensorflow as tf
from image_segmentation.common import PROJECT_ROOT
from image_segmentation.common.utils import edit_callback_paths
from image_segmentation.common.utils import generate_unique_logpath
from image_segmentation.common.utils import get_model
from image_segmentation.common.utils import save_config_file
from image_segmentation.common.utils import save_results
from image_segmentation.data.datamodule import DataModule
from image_segmentation.models.models import build_callbacks
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tensorflow.keras import Model


def training(cfg: DictConfig, model: Model):
    # Instantiate datamodule
    datamodule: DataModule = hydra.utils.instantiate(cfg.nn.data, stage='fit', _recursive_=False)

    # Check data path
    datamodule.prepare_data()

    # Setup dataset
    datamodule.setup()

    # Create training and validation data generator
    train_dataloader: tf.data.Dataset = datamodule.train_dataloader()
    val_dataloader: tf.data.Dataset = datamodule.val_dataloader()

    print(f'Train DataLoader shape: {len(train_dataloader)}')
    print(f'Valid DataLoader shape: {len(val_dataloader)}')

    # Callback
    callbacks = build_callbacks(cfg.train.callbacks)

    # Fit model
    model.fit(
        x=train_dataloader,
        epochs=cfg.train.trainer.max_epochs,
        validation_data=val_dataloader,
        callbacks=callbacks,
    )

    print('The training ended successfully !')
    print(f'Training artefacts are available here: {cfg.train.callbacks.model_checkpoint.filepath}')


def predict(cfg: DictConfig, model: Model, save_dir: str):
    # Instantiate datamodule
    datamodule: DataModule = hydra.utils.instantiate(cfg.nn.data, stage='test', _recursive_=False)

    # Check data path
    datamodule.prepare_data()

    # Setup dataset
    datamodule.setup()

    # Create test data generator
    test_dataloader: tf.data.Dataset = datamodule.test_dataloader()
    print(f'Test DataLoader shape: {len(test_dataloader)}')

    predictions = model.predict(x=test_dataloader)

    save_results(
        test_paths=datamodule.test_paths,
        save_dir=save_dir,
        predictions=predictions,
        tests_imgs=test_dataloader.unbatch(),
    )


@hydra.main(version_base='1.2', config_path=str(PROJECT_ROOT / 'conf'), config_name='default.yaml')
def main(cfg: DictConfig):
    """Execute the main function

    Args:
        cfg (DictConfig): configuration, defined by Hydra in /conf
    """
    # Authorize config file edition
    OmegaConf.set_struct(cfg, True)

    # Retrieve basic model info
    model_name, mode, load_weights = cfg.nn.module.info.values()

    # Init directory to save model and configuration
    top_logdir = os.path.join(PROJECT_ROOT, 'models')
    save_dir = generate_unique_logpath(top_logdir, model_name.lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Save configuration file (either from a load model or from basic config)
    save_config_file(save_dir=save_dir, load_weights=load_weights)

    # Edit checkpoint save path and tensorboard path
    edit_callback_paths(cfg=cfg, save_dir=save_dir)

    # Retrieve model weights if given, else build the model
    unet_model = get_model(cfg=cfg, load_weights=load_weights)

    if mode == 'training':
        training(cfg=cfg, model=unet_model)
    elif mode == 'predict':
        predict(cfg=cfg, model=unet_model, save_dir=save_dir)
    else:
        print('Two modes are available: training or predict')
        sys.exit()


if __name__ == '__main__':
    main()
