from __future__ import annotations

import hydra
from image_segmentation.models.tf_unet.unet_block import ConcatCropFeatureMapBlock
from image_segmentation.models.tf_unet.unet_block import MirrorPadding
from omegaconf import DictConfig
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model


def unet_constructor(
    image_size: tuple[int, int],
    num_pooling: int,
    num_layers_before_pooling: int,
    initial_num_filters: int,
    pool_size: int,
    block_type: DictConfig,
    blocks_config: DictConfig,
) -> Model:
    """Build UNet model.

    Args:
        image_size (tuple[int, int]): size of the input image.
        num_pooling (int): the number of pooling operations.
        num_layers_before_pooling (int): the number of layers before each pooling.
        initial_num_filters (int): number of filters for the first convolution layer.
        pool_size (int):  window size over which to take the maximum for MaxPooling2D.
        block_type (DictConfig): which type block to use for your UNet architecture.
        blocks_config (DictConfig): configuration for the different block types.

    Returns:
        Model: UNet model
    """

    inputs = Input(shape=(image_size[0], image_size[1], 1), name='inputs')
    x = inputs
    residual_connection = {}

    # Retrieve block config
    block_config = select_block_config(block_type=block_type, blocks_config=blocks_config)
    block_config_up = select_block_config_up(block_type=block_type, blocks_config=blocks_config)

    # Contracting path
    for id_pooling in range(num_pooling):
        num_filters = 2 ** id_pooling * initial_num_filters
        # Iterate over the number of layers before pooling
        for _ in range(num_layers_before_pooling):
            x = hydra.utils.instantiate(block_type.path, num_filters=num_filters, **block_config)(x)

        residual_connection[id_pooling] = x

        x = layers.MaxPooling2D((pool_size, pool_size))(x)

    for _ in range(num_layers_before_pooling):
        x = hydra.utils.instantiate(block_type.path, num_filters=2**(id_pooling+1)*initial_num_filters, **block_config)(x)

    # Expansive path
    for id_pooling_up in range(id_pooling, -1, -1):
        num_filters = 2 ** id_pooling_up * initial_num_filters
        # Up Samp block
        x = hydra.utils.instantiate(block_type.path_up, num_filters=num_filters, pool_size=pool_size, **block_config_up)(x)

        # Concatenate vector with residual connection
        # x = layers.Concatenate(axis=-1)([residual_connection[id_pooling_up], x])
        x = ConcatCropFeatureMapBlock()(x, residual_connection[id_pooling_up])
        # Iterate over the number of layers before pooling
        for _ in range(num_layers_before_pooling):
            x = hydra.utils.instantiate(block_type.path, num_filters=num_filters, **block_config)(x)

    x = hydra.utils.instantiate(blocks_config.final_block)(x)

    outputs = MirrorPadding(img_size=(image_size[0], image_size[1], 1))(x)

    model = Model(inputs, outputs, name='unet')

    return model


def select_block_config(block_type: DictConfig, blocks_config: DictConfig) -> DictConfig:
    """Retrieve the block configuration you have selected.

    Args:
        block_type (DictConfig): which type block to use for your UNet architecture.
        blocks_config (DictConfig): configuration for the different block types.

    Returns:
        DictConfig: the configuration of the block you have selected
    """
    if block_type.name == 'standard_block':
        return blocks_config.standard_block
    elif block_type.name == 'resnet_block':
        return blocks_config.resnet_block
    else:
        return blocks_config.standard_block


def select_block_config_up(block_type: DictConfig, blocks_config: DictConfig) -> DictConfig:
    """Retrieve the block configuration for the up sampling of the expansive path.
    For now, only StardardUpBlock is available, but Conv2DTranspose will be available in
    the next release.

    Args:
        block_type (DictConfig): which type block to use for your UNet architecture.
        blocks_config (DictConfig): configuration for the different block types.

    Returns:
        DictConfig: the configuration of the block you have selected
    """
    if block_type.name_up == 'standard_up_block':
        return blocks_config.standard_up_block
    else:
        return blocks_config.standard_up_block
