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

    # Contracting path
    for id_pooling in range(num_pooling):
        num_filters = 2 ** id_pooling * initial_num_filters
        # Iterate over the number of layers before pooling
        for _ in range(num_layers_before_pooling):
            x = hydra.utils.instantiate(blocks_config[block_type.name_down], num_filters=num_filters)(x)

        residual_connection[id_pooling] = x

        x = layers.MaxPooling2D((pool_size, pool_size))(x)

    for _ in range(num_layers_before_pooling):
        x = hydra.utils.instantiate(blocks_config[block_type.name_down], num_filters=2**(id_pooling+1)*initial_num_filters)(x)

    # Expansive path
    for id_pooling_up in range(id_pooling, -1, -1):
        num_filters = 2 ** id_pooling_up * initial_num_filters
        # Up Samp block
        x = hydra.utils.instantiate(blocks_config[block_type.name_up], num_filters=num_filters, pool_size=pool_size)(x)

        # Concatenate vector with residual connection
        x = ConcatCropFeatureMapBlock()(x, residual_connection[id_pooling_up])
        # Iterate over the number of layers before pooling
        for _ in range(num_layers_before_pooling):
            x = hydra.utils.instantiate(blocks_config['standard_block'], num_filters=num_filters)(x)

    x = hydra.utils.instantiate(blocks_config.final_block)(x)

    # Resize the output if necessary to be able to compare it with the mask
    if x.shape[1] != image_size[0]:
        outputs = MirrorPadding(
            img_size=(image_size[0], image_size[1], 1),
            img_output_shape=(x.shape[1:3]),
        )(x)
    else:
        outputs = x

    model = Model(inputs, outputs, name='unet')

    return model
