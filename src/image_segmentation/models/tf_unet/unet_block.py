"""Define all the different blocks that can be used in the UNet architecture."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class StandardBlock(layers.Layer):
    """Standard Convolution Block.
    It is compose of:
        - Conv2D
        - Batch Norm (Optinal)
        - Activation
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        padding: str,
        num_filters: int,
        kernel_initializer: str,
        activation: str,
        batch_norm: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.padding = padding
        self.num_filters = num_filters
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.batch_norm = batch_norm

        self.conv2d = layers.Conv2D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
        )

        self.batch_norm_layer = layers.BatchNormalization() if self.batch_norm else None

        self.activation_layer = layers.Activation(self.activation)

    def call(self, inputs):
        x = inputs
        x = self.conv2d(x)

        if self.batch_norm_layer is not None:
            x = self.batch_norm_layer(x)

        x = self.activation_layer(x)
        return x

    def get_config(self):
        return dict(
            kernel_size=self.kernel_size,
            padding=self.padding,
            num_filters=self.num_filters,
            activation=self.activation,
            batch_norm=self.batch_norm,
            **super().get_config(),
        )


class StandardUpBlock(layers.Layer):
    """Standard Up sampling Block.
    It is compose of:
        - UpSampling2D
        - Conv2D
    """

    def __init__(
        self,
        pool_size: int,
        kernel_size: int | tuple[int, int],
        padding: str,
        num_filters: int,
        kernel_initializer: str,
        activation: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_filters = num_filters
        self.kernel_initializer = kernel_initializer
        self.activation = activation

        self.upsamp = layers.UpSampling2D((self.pool_size, self.pool_size))

        self.conv2d = layers.Conv2D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
        )

        self.activation_layer = layers.Activation(activation)

    def call(self, inputs):
        x = inputs
        x = self.upsamp(x)
        x = self.conv2d(x)
        x = self.activation_layer(x)
        return x

    def get_config(self):
        return dict(
            pool_size=self.pool_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            num_filters=self.num_filters,
            activation=self.activation,
            **super().get_config(),
        )


class ConcatCropFeatureMapBlock(layers.Layer):
    """Concatenate the output of the StandardUpBlock and the corresponding
    feature map from the contracting path.
    It assures that concatenation can be applied by cropping the feature map
    """

    def call(self, x, down_layer):
        down_layer_shape = tf.shape(down_layer)
        x_shape = tf.shape(x)

        height_diff = (down_layer_shape[1] - x_shape[1]) // 2
        width_diff = (down_layer_shape[2] - x_shape[2]) // 2

        down_layer_cropped = down_layer[
            :,
            height_diff: (x_shape[1] + height_diff),
            width_diff: (x_shape[2] + width_diff),
            :
        ]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x


class FinalBlock(layers.Layer):
    """Final Conv Block.
    It is compose of:
        - Conv2D
        - Conv2D

    It assures that the output has 1 channel.
    """

    def __init__(
        self,
        filters: int,
        padding: str,
        kernel_size: int | tuple[int, int],
        kernel_initializer: str,
        activation: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.filters = filters
        self.padding = padding
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.activation = activation

        self.conv2d_1 = layers.Conv2D(
            filters=filters,
            kernel_size=self.kernel_size,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
            activation=self.activation,
        )

        self.conv2d_2 = layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            activation='sigmoid',
        )

    def call(self, inputs):
        x = inputs
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        return x

    def get_config(self):
        return dict(
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.activation,
            **super().get_config(),
        )


class MirrorPadding(layers.Layer):
    """This layers pads the output from UNet to match the input image shape,
    hence the mask/target shape.
    This layer is necessary is padding mode is VALID.
    """

    def __init__(
        self,
        img_size,
        mode: str = 'REFLECT',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.img_size = img_size
        self.mode = mode

    def call(self, inputs):
        x = inputs
        x_shape = tf.shape(x)

        height_diff = (self.img_size[0] - x_shape[1]) // 2
        width_diff = (self.img_size[1] - x_shape[2]) // 2

        x = tf.pad(x, [[0, 0], [height_diff, height_diff], [width_diff, width_diff], [0, 0]], mode=self.mode)
        return x

    def get_config(self):
        return dict(
            img_size=self.img_size,
            mode=self.mode,
            **super().get_config(),
        )
