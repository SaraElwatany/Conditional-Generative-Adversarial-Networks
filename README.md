# InterfaceGAN Facial Attribute Editing

This branch implements facial attribute manipulation using InterfaceGAN boundaries on StyleGAN2 latent space.

The approach modifies facial attributes such as pose, smile, and age by moving a latent vector along pretrained semantic boundaries discovered in the StyleGAN latent space.

This repository is part of a larger project exploring GAN-based facial attribute manipulation.


## Method Overview

The pipeline follows these steps:

* ### Image Generation

Images are generated using a pretrained StyleGAN2 generator trained on the FFHQ dataset.

### * Latent Projection

Images are projected into the W+ latent space using the StyleGAN2 projector.

### * Boundary Manipulation

Pretrained InterfaceGAN boundaries are applied to manipulate attributes such as:

* Pose
* Smile
* Age

### * Attribute Control

A scalar α (alpha) controls the strength of the attribute change:

𝑤′ = 𝑤 + 𝛼 𝑑

Where:

* w = original latent code
* d = boundary direction
* α = manipulation strength

### * Image Reconstruction

The modified latent vector is passed back through StyleGAN2 to produce the edited image.



## Playground (Kaggle Notebook)

You can try the full pipeline interactively using the Kaggle notebook:

Kaggle Playground:
[INSERT NOTEBOOK LINK HERE]

This notebook allows you to:

* Generate images
* Project images to latent space
* Apply InterfaceGAN boundaries
* Visualize attribute edits

Important Kaggle Setup

Before running the notebook:

Open Notebook Settings

Enable GPU acceleration

Select T4 GPU

⚠️ The StyleGAN projector and generation steps require GPU acceleration and will be very slow on CPU.

Supported Attributes

The following pretrained boundaries are currently used:
* Pose
* Smile
* Age

Each attribute can be controlled using a configurable alpha value.

Example:

edited_latent = latent + alpha * pose_boundary

Increasing alpha increases the strength of the transformation.



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

StyleGAN2
InterfaceGAN
FFHQ Dataset
