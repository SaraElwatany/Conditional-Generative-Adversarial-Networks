# InterfaceGAN Facial Attribute Editing

This branch implements facial attribute manipulation using InterfaceGAN boundaries on StyleGAN2 latent space.

The approach modifies facial attributes such as pose, smile, and age by moving a latent vector along pretrained semantic boundaries discovered in the StyleGAN latent space.

This repository is part of a larger project exploring GAN-based facial attribute manipulation.


## Method Overview

The pipeline follows these steps:

#### 1. Image Generation

Images are generated using a pretrained StyleGAN2 generator trained on the FFHQ dataset.

#### 2. Latent Projection

Images are projected into the W+ latent space using the StyleGAN2 projector.

#### 3. Boundary Manipulation

Pretrained InterfaceGAN boundaries are applied to manipulate attributes such as:

* Pose
* Smile
* Age

#### 4. Attribute Control

A scalar **α (alpha)** controls the strength of the attribute change:

<p align="center">
w′ = w + αd
</p>

Where:

* w = original latent code
* d = boundary direction
* α = manipulation strength

#### 5. Image Reconstruction

The modified latent vector is passed back through StyleGAN2 to produce the edited image.



## Playground (Kaggle Notebook)

You can try the full pipeline interactively using the Kaggle notebook:

Kaggle Playground:
[![Kaggle Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/saraaymanelwatany/styleganv2)

This notebook allows you to:

* Generate images
* Project images to latent space
* Apply InterfaceGAN boundaries
* Visualize attribute edits

#### Important Kaggle Setup

Before running the notebook:

1. Open Notebook Settings
2. Enable GPU acceleration
3. Select T4 GPU

⚠️ The StyleGAN projector and generation steps require GPU acceleration and will be very slow on CPU.



## Supported Attributes

Only the following pretrained boundaries are currently used:
* Pose
* Smile
* Age


#### Notes:

- Each attribute can be controlled using a configurable alpha value.
- Increasing alpha increases the strength of the transformation.



## Limitations

* Projection using the StyleGAN2 projector can be computationally expensive.
* Boundaries are currently available only for a limited number of attributes.
* Additional attributes (e.g., hair color, hairstyle) would require training new boundaries using labeled data.



## Future Work

Possible improvements include:

1. Training additional InterfaceGAN boundaries for more attributes.
2. Improving projection speed using encoders such as e4e.
3. Combining boundary manipulation with text-guided methods such as StyleCLIP.



## References

1. [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main)
2. [InterfaceGAN](https://github.com/genforce/interfacegan/tree/master)
3. [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
