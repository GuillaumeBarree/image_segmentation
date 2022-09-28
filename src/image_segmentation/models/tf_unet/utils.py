from __future__ import annotations

from omegaconf import DictConfig


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


def image_size_end_contract(img_size, id_pool, num_pooling, num_layers_before, kernel_size, padding):
    if id_pool < num_pooling:
        if padding == 'valid':
            end_conv_size = img_size - (int((kernel_size-1)/2)*2*num_layers_before)
        elif padding == 'same':
            end_conv_size = img_size
        next_img_size = end_conv_size // 2

        return image_size_end_contract(
            next_img_size,
            id_pool=id_pool+1,
            num_pooling=num_pooling,
            num_layers_before=num_layers_before,
            kernel_size=kernel_size,
            padding=padding,
        )
    else:
        if padding == 'valid':
            return img_size - (int((kernel_size-1)/2)*2*num_layers_before)
        elif padding == 'same':
            return img_size


def image_size_end_contract_invert(img_size, id_pool, num_pooling, num_layers_before, kernel_size, padding):
    if id_pool < num_pooling:
        if padding == 'valid':
            end_conv_size = img_size + (int((kernel_size-1)/2)*2*num_layers_before)
        elif padding == 'same':
            end_conv_size = img_size
        next_img_size = end_conv_size * 2

        return image_size_end_contract_invert(
            next_img_size,
            id_pool=id_pool+1,
            num_pooling=num_pooling,
            num_layers_before=num_layers_before,
            kernel_size=kernel_size,
            padding=padding,
        )
    else:
        if padding == 'valid':
            return img_size + (int((kernel_size-1)/2)*2*num_layers_before)
        elif padding == 'same':
            return img_size


def image_size_end_expens(img_size, id_pool, num_pooling, num_layers_before, kernel_size, padding):
    if id_pool < num_pooling:
        end_upconv_size = 2*img_size

        if padding == 'valid':
            next_img_size = end_upconv_size - (int((kernel_size-1)/2)*2*num_layers_before)
        elif padding == 'same':
            next_img_size = end_upconv_size

        return image_size_end_expens(
            next_img_size,
            id_pool=id_pool+1,
            num_pooling=num_pooling,
            num_layers_before=num_layers_before,
            kernel_size=kernel_size,
            padding=padding,
        )
    else:
        return img_size
