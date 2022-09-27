from __future__ import annotations

import logging
import os

import hydra
import omegaconf
import tensorflow as tf
from image_segmentation.common import PROJECT_ROOT
from image_segmentation.data.datamodule import DataModule
from image_segmentation.models.models import build_callbacks
from image_segmentation.models.models import get_compiled_model
from image_segmentation.models.tf_unet.unet import unet_constructor
from omegaconf import DictConfig
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig):
    """Main function to train DL models for image segmentation.
    Depending on the stage, it can either be trained or used for inference.

    Args:
        cfg (DictConfig): configuration, defined by Hydra in /conf
    """
    # Instantiate useful constant
    original_height = cfg.nn.data.resize.height
    original_width = cfg.nn.data.resize.width
    original_img_size = (original_height, original_width)

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: DataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    # Check data path
    pylogger.info(f'Verify data path at <{os.path.join(PROJECT_ROOT, cfg.nn.data.data_path)}>')
    datamodule.prepare_data()

    # Setup dataset
    pylogger.info(f'Setup dataset for stage <{cfg.nn.data.stage}>')
    datamodule.setup()

    if cfg.nn.data.stage is None or cfg.nn.data.stage == 'fit':
        train_dataloader: tf.data.Dataset = datamodule.train_dataloader()
        val_dataloader: tf.data.Dataset = datamodule.val_dataloader()

        print(f'Train DataLoader shape: {len(train_dataloader)}')
        print(f'Valid DataLoader shape: {len(val_dataloader)}')

    if cfg.nn.data.stage is None or cfg.nn.data.stage == 'test':
        test_dataloader: tf.data.Dataset = datamodule.test_dataloader()

        print(f'Test DataLoader shape: {len(test_dataloader)}')

    # Instantiate UNet model
    unet_model = unet_constructor(
        image_size=original_img_size,
        blocks_config=cfg.nn.unet.blocks,
        **cfg.nn.unet.structure,
    )

    unet_model = get_compiled_model(
        model=unet_model,
        optimizer_config=cfg.nn.module.optimizer,
        custom_loss_config=cfg.nn.module.custom_loss,
        metrics_config=cfg.nn.module.metrics,
    )

    # Callback
    callbacks = build_callbacks(cfg.train.callbacks)

    unet_model.fit(train_dataloader, epochs=cfg.train.trainer.max_epochs, validation_data=val_dataloader, callbacks=callbacks)

    results = unet_model.predict(test_dataloader, verbose=1)

    print(results.shape)
    print('DONE')


@hydra.main(version_base='1.2', config_path=str(PROJECT_ROOT / 'conf'), config_name='default.yaml')
def main(cfg: omegaconf.DictConfig):
    """Execute the main function

    Args:
        cfg (omegaconf.DictConfig): configuration, defined by Hydra in /conf
    """
    run(cfg)


if __name__ == '__main__':
    main()
