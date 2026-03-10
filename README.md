# GAN-Based Facial Attribute Manipulation

This project explores facial attribute manipulation using Generative Adversarial Networks (GANs).
Two complementary approaches are implemented:

* InterfaceGAN — latent space boundary manipulation
* StyleCLIP — text-driven semantic editing using CLIP

Both methods operate on StyleGAN2 latent representations and allow modification of facial attributes such as:

* Age
* Facial expressions
* Hair color
* Hairstyle
* Facial pose

The project demonstrates how latent space geometry and vision-language models can be leveraged to perform realistic image editing without retraining the generator.


## Kaggle Playground

You can explore the full pipeline interactively using the Kaggle notebook: [![Kaggle Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)]([https://www.kaggle.com/code/saraaymanelwatany/styleganv2](https://www.kaggle.com/code/saraaymanelwatany/styleclip))

⚠️ **Important**

Before running the notebook:

1. Open Notebook Settings
2. Enable GPU acceleration
3. Select T4 GPU

Some steps such as StyleGAN projection and StyleCLIP optimization require GPU acceleration.



## Project Overview

The system combines three main components:

* StyleGAN2 – high-resolution face generator
* e4e Encoder – projects real images into StyleGAN latent space
* Editing Methods:
  1. InterfaceGAN boundaries
  2. StyleCLIP text guidance


## Pipeline

The full pipeline follows these steps:

#### 1. Image Projection

An input image is first projected into the StyleGAN W+ latent space using the e4e encoder.

This produces a latent representation that reconstructs the original face.

#### 2. Attribute Editing

Two different editing techniques are used depending on the attribute.


#### InterfaceGAN (Latent Boundary Editing)

InterfaceGAN finds linear directions in latent space corresponding to semantic attributes.

A latent vector is modified using:

<p align="center"> <b>w′ = w + αd</b> </p>

Where:

* w — original latent code
* d — attribute boundary direction
* α — manipulation strength

This method works well for geometric transformations such as pose.

#### StyleCLIP (Text-Guided Editing)

StyleCLIP uses CLIP image-text similarity to guide edits using natural language prompts.

The optimization objective balances semantic alignment with identity preservation.

<p align="center"> <b>L = L<sub>CLIP</sub> + λ<sub>id</sub>L<sub>id</sub> + λ<sub>l2</sub>L<sub>l2</sub></b> </p>

Where:

* LCLIP — similarity between generated image and text prompt
* Lid — identity preservation loss
* Ll2 — latent regularization

This method enables flexible edits such as:

* Hair color
* Hairstyle
* Facial expressions
* Age




## Final Editing Strategy

Both approaches are combined to leverage their strengths:


| Method       | Attributes                                   |
|-------------|----------------------------------------------|
| InterfaceGAN | Pose                                         |
| StyleCLIP    | Age, hair color, hairstyle, facial expressions |


This hybrid strategy provides both controllability and semantic flexibility.

Supported Attributes

### Hair Style

| ID | Prompt              |
|----|-------------------|
| 0  | Mohawk hairstyle   |
| 1  | Bowl cut           |
| 2  | Bob cut            |
| 3  | Curly hair         |

---

### Hair Colour

| ID | Prompt       |
|----|-------------|
| 4  | Black hair  |
| 5  | Blonde hair |
| 6  | Brown hair  |

---

### Facial Expression

| ID  | Prompt      |
|-----|------------|
| 7   | Smiling    |
| 8   | Sad        |
| 9   | Surprised  |
| 10  | Angry      |
| 11  | Frightened |

---

### Age

| ID  | Prompt             |
|-----|------------------|
| 12  | Child             |
| 13  | Person in their 20s |
| 14  | Person in their 50s |
| 15  | Person in their 70s |
| 16  | Person in their 90s |



## Limitations

1. Latent optimization in StyleCLIP can take several minutes per image.
2. Some attributes (e.g., pose) are difficult to edit reliably using CLIP prompts alone.
3. InterfaceGAN boundaries require labeled data to train new attributes.




## References

1. [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main)
2. [InterfaceGAN](https://github.com/genforce/interfacegan/tree/master)
3. [StyleCLIP](https://github.com/orpatashnik/StyleCLIP/tree/main)
4. [CLIP](https://github.com/openai/CLIP)
5. [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)


