# <p style="text-align:center"> Project configuration files with Hydra</p> <br>
<div id="top"></div>
<br />
<div align="center">
  <p align="center">
    Hydra is an open-source Python framework that simplifies the development of research and other complex applications.
    <br />
    <a href="https://hydra.cc/"><strong>Explore the docs »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#folder-structure">Navigate into the configuration folder</a>
    </li>
    <li>
      <a href="#nn-configuration-file">NN configuration file</a>
      <ul>
        <li><a href="#data">Data</a></li>
        <li><a href="#unet">U-Net architecture</a></li>
        <li><a href="#module">Module</a></li>
      </ul>
    </li>
    <li><a href="#train-configuration-file">Train configuration file</a></li>
    <li><a href="#change-params-in-command-line">Change params in command line</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->
## Getting Started

### Folder structure

```bash
.
├── README.md
├── default.yaml
├── hydra
│   └── default.yaml
├── nn
│   └── default.yaml
└── train
    └── default.yaml
```

* `./default.yaml`: Root configuration file. It contains the path of the sub configurations files.
* `./hydra`: Hydra configuration file.
* `./nn`: Configuration file related to data preprocessing and model architectures.
* `./train`: Configuration file related to model training.

<p align="right">(<a href="#top">back to top</a>)</p>

### NN configuration file

This `default.yaml` file of the `nn` folder contains all the information related to data load and preprocessing and model architecture design.

```yaml
data: ...
unet: ...
module: ...
```

* `data`: Related to DataModule and Dataset creation.
* `unet`: Related to U-Net architecture.
* `module`: Related to compiling models.

#### Data

```yaml
  _target_: image_segmentation.data.datamodule.DataModule

  data_path: data/membrane

  batch_size:
    train: 2
    val: 2
    test: 2

  repeat: 4

  resize:
      _target_: tensorflow.keras.layers.Resizing
      height: 256
      width: 256
      crop_to_aspect_ratio: False

  data_augmentation:
    random_flip:
      _target_: tensorflow.keras.layers.RandomFlip
      seed: 0
    random_translation:
      _target_: tensorflow.keras.layers.RandomTranslation
      seed: 0
      height_factor: 0.05
      width_factor: 0.05
      fill_mode: nearest
    random_rotation:
      _target_: tensorflow.keras.layers.RandomRotation
      seed: 0
      factor: 0.2
      fill_mode: nearest
    random_zoom:
      _target_: tensorflow.keras.layers.RandomZoom
      seed: 0
      height_factor: 0.05
      fill_mode: nearest

  val_percentage: 0.2
```

* `resize`: Layer that allows you you to resize you input images. For now it calls the [Resizing layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Resizing), but you can change it with [any resizing layers offered by Tensorflow](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/).
* `data_augmentation`: Define all the image augmentation layers you want to use with their parameters. Do not forget to fix the seed to garantee that the same transformation is applied to the image and its associated mask. Available image augmentation layers are listed [here](https://keras.io/api/layers/preprocessing_layers/image_augmentation/).

<p align="right">(<a href="#top">back to top</a>)</p>

#### Unet

```yaml
structure:
    num_pooling: 3
    num_layers_before_pooling: 1
    initial_num_filters: 32
    pool_size: 2
    block_type:
      name: resnet_block
      path:
        _target_: image_segmentation.models.tf_unet.unet_block.ResnetBlock
      name_up: standard_up_block
      path_up:
        _target_: image_segmentation.models.tf_unet.unet_block.StandardUpBlock

  blocks:
    standard_block:
      kernel_size: 3
      padding: valid
      kernel_initializer: he_normal
      activation: relu
      batch_norm: True

    resnet_block:
      kernel_size: 3
      kernel_initializer: he_normal
      activation: relu

    standard_up_block:
      kernel_size: 2
      padding: same
      kernel_initializer: he_normal
      activation: relu

    final_block:
      _target_: image_segmentation.models.tf_unet.unet_block.FinalBlock
      filters: 2
      kernel_size: 3
      padding: same
      kernel_initializer: he_normal
      activation: relu
```

Here we will find the U-Net structure configuration

* `num_pooling`: number of pooling operations.
* `num_layers_before_pooling`: number of layers before each pooling.
* `initial_num_filters`: number of filter of the first convolution layer.
* `pool_size`: pool size for MaxPooling2D layer.
* `block_type`: This will defined the block type you want to use for your U-Net architectude. It is composed of two parts
  * The first one defines the block you want to use for the contracting path. Two possibilities are available.

  ```yaml
  name: standard_block
  path:
    _target_: image_segmentation.models.tf_unet.unet_block.StandardBlock
  ```

  ```yaml
  name: resnet_block
  path:
    _target_: image_segmentation.models.tf_unet.unet_block.ResnetBlock
  ```

  * The second one defines the block you want to use for the expensive path. Only one block type is available for this part. Next step could be implementing a TransConv2D layer.

And bellow the struccture section, you will find all the information to configure the different block you have just selected.

* `standard_block`: input -> convolution (conv) -> batch normalization (BN) (optional) -> activation -> output
* `resnet_block`: input -> conv -> BN -> activation -> conv -> BN -> activation -> tmp_output -> Add(input, tmp_output) -> output
* `standard_up_block`: input -> UpSampling -> conv -> output
* `final_block`: inpur -> conv -> conv -> output

<p align="right">(<a href="#top">back to top</a>)</p>

#### Module

```yaml
info:
    name: unet
    mode: training
    load_weights: ~

optimizer:
    _target_: tensorflow.keras.optimizers.Adam
    learning_rate: 0.001
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1e-07
    amsgrad: False
    name: Adam

custom_loss:
    mode: crop
    loss:
        _target_: tensorflow.keras.losses.BinaryCrossentropy
        from_logits: False

metrics:
    categorical_accuracy:
        _target_: tensorflow.keras.metrics.CategoricalAccuracy
    categorical_crossentropy:
        _target_: tensorflow.keras.metrics.CategoricalCrossentropy
        from_logits: False
```

* `info`: Contains information about how you want to use the package.
  * Training or predict
  * From scratch of from existing weights
  * This parameters are directly set in the command line
* `optimized`: Declare your optimizer and its parameters
* `custom_loss`: This is an on loss function layers. Two modes are available
  * `crop`: If the output original output shape of the model differ from the image mask size, compute the loss function only between the center cropped mask and the output
  * `mirror`: Reflect padding is apply to the output image and the loss is computed with the mask image.
* `metrics`: Define you metrics.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
### Train configuration file

```yaml
trainer:
  max_epochs: 40

callbacks:
  earlystopping:
    _target_: tensorflow.keras.callbacks.EarlyStopping
    monitor: val_loss
    min_delta: 0
    patience: 5
    verbose: 0
    mode: auto
    restore_best_weights: False

  model_checkpoint:
    _target_: tensorflow.keras.callbacks.ModelCheckpoint
    filepath: models/unet/unet_membrane.hdf5
    monitor: val_loss
    verbose: 0
    save_best_only: True
    save_weights_only: False
    mode: auto
    save_freq: epoch

  tensorboard:
    _target_: tensorflow.keras.callbacks.TensorBoard
    log_dir: tensorboard/

  reduce_on_plateau:
    _target_: tensorflow.keras.callbacks.ReduceLROnPlateau
    monitor: val_loss
    factor: 0.2
    patience: 3
    verbose: 0
    mode: auto
    min_delta: 0.01

```

* `epoch`: Define the number of epoch
* `callbacks`: List all the callbacks you want to use for you training
  * For `model_checkpoint`, filepath will be automatically set for you
  * For `tensorboard`, log_dir will be automatically set for you

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Change param in command line -->
## Change params in command line

Even if your configuration file is save, you can still change some parameters when calling the main script.

For example, if you want to change the mode:

```sh
python src/image_segmentation/main.py nn.module.info.mode=training
```

<p align="right">(<a href="#top">back to top</a>)</p>
