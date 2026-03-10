# StyleCLIP Facial Attribute Editing

This branch implements text-driven facial attribute manipulation using StyleCLIP with StyleGAN2.

Unlike traditional boundary-based methods, this approach allows editing facial attributes using natural language prompts such as:

* "person in their 70s"
* "blonde hair"
* "curly hair"
* "angry expression"

The edits are performed by optimizing the latent representation so that the generated image becomes semantically closer to the target text description according to a CLIP similarity score.


## Kaggle Playground

You can experiment with the full pipeline interactively using the Kaggle notebook: [![Kaggle Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)]([https://www.kaggle.com/code/saraaymanelwatany/styleganv2](https://www.kaggle.com/code/saraaymanelwatany/styleclip))

⚠️ **Important:**:
Before running the notebook:

1. Open Notebook Settings
2. Enable GPU acceleration
3. Select T4 GPU

StyleCLIP optimization requires GPU acceleration and will be extremely slow on CPU.



## Method Overview

The pipeline combines StyleGAN2, CLIP, and latent optimization.

#### 1. Image Projection

The input image is first projected into the StyleGAN W+ latent space using the e4e encoder.

This produces a latent representation that reconstructs the original image.

#### 2. Text Guidance with CLIP

CLIP encodes both:

* the generated image
* the target text prompt

into the same embedding space.

The similarity between the two embeddings guides the optimization.

#### 3. Latent Optimization

The latent code is iteratively updated to maximize the similarity between the generated image and the text description.

The optimization objective balances three losses:

* CLIP Loss (semantic alignment with text)
* Identity Loss (preserve identity using ArcFace)
* L2 Regularization (keep latent close to original)

#### Optimization Objective

The latent vector is optimized according to:

<p align="center"> <b>L = L<sub>CLIP</sub> + λ<sub>id</sub>L<sub>id</sub> + λ<sub>l2</sub>L<sub>l2</sub></b> </p>

Where:

* LCLIP — similarity loss between generated image and text prompt
* Lid — identity preservation loss
* Ll2 — latent regularization term
* λ — hyperparameters controlling the strength of each constraint


#### Supported Attribute Editing

This implementation supports multiple attribute edits via predefined prompts.

### Hair Style

| ID | Prompt |
|----|------|
| 0 | Mohawk hairstyle |
| 1 | Bowl cut |
| 2 | Bob cut |
| 3 | Curly hair |


### Hair Colour

| ID | Prompt |
|----|------|
| 4 | Black hair |
| 5 | Blonde hair |
| 6 | Brown hair |


### Facial Expression

| ID | Prompt |
|----|------|
| 7 | Smiling |
| 8 | Sad |
| 9 | Surprised |
| 10 | Angry |
| 11 | Frightened |


### Age

| ID | Prompt |
|----|------|
| 12 | Child |
| 13 | Person in their 20s |
| 14 | Person in their 50s |
| 15 | Person in their 70s |
| 16 | Person in their 90s |

#### Key Hyperparameters

The optimization process is controlled by:
* n_steps: Number of optimization iterations
* id_lambda: Strength of identity preservation
* l2_lambda: Latent regularization
* lr:	Optimization learning rate

These parameters influence the strength of the edit vs identity preservation.




## Limitations

1. Latent optimization can take several minutes per image.

2. Certain attributes (such as pose) are difficult to manipulate reliably using CLIP guidance alone.

3. Some prompts may produce unexpected or exaggerated edits depending on the optimization settings.




## Future Work

Potential improvements include:

* Faster inversion using improved encoders
* Combining StyleCLIP with InterfaceGAN boundaries
* Exploring direction-based StyleCLIP editing instead of full optimization




## References

1. [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main)
2. [StyleCLIP](https://github.com/orpatashnik/StyleCLIP/tree/main)
3. [CLIP](https://github.com/openai/CLIP)
4. [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
