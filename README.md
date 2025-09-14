# ContQuat-PT

_Incorporation of **Cont**rastive learning and **Quat**ernion-based pose encoding 
in a **P**rogressive **T**ransformers-like model for sign language production._

Source code for the paper ["Towards Skeletal and Signer Noise Reduction in Sign Language Production via 
Quaternion-Based Pose Encoding and Contrastive Learning"](https://doi.org/10.48550/arXiv.2508.14574)
(Guilhem Fauré, Mostafa Sadeghi, Sam Bigeard, Slim Ouni - IVA 2025 SLTAT Workshop).

## Table of Contents
[1. Description](#1-description)

[2. Installation](#2-installation)

[3. Usage](#3-usage)

[4. Demos](#4-demos)

## 1. Description

*ContQuat-PT* uses the [Progressive Transformers](https://doi.org/10.48550/arXiv.2004.14874)
backbone architecture while allowing the following modifications and/or extensions:
- possibility to encode en predict skeletal poses via bone rotations using quaternion-based parametrization, and
replacing the MSE loss by a geodesic loss
- possibility to add a supervised contrastive learning loss (either based on gloss or SBERT similarity between sentences)
on the decoder's self-attention outputs in the definition of the global loss, as a regularization term controlled
by a parameter $\lambda$

The following diagram provides an overview of the architecture and shows how our contributions integrate into it:

![Architecture](./images/architecture.png)

## 2. Installation

```commandline
git clone https://github.com/GFaure9/ContQuat-PT.git
cd ./ContQuat-PT
pip install -r requirements.txt
```

## 3. Usage

TODO

## 4. Demos

TODO

## How to cite?

If you use this code in your research, please cite the following paper:

```
@inproceedings{faure2025contquatpt,
	title		=	{Towards Skeletal and Signer Noise Reduction in Sign Language Production via Quaternion-Based Pose Encoding and Contrastive Learning},
	author		=	{Fauré, Guilhem and Sadeghi, Mostafa and Bigeard, Sam and Ouni, Slim},
	booktitle   =   {ACM International Conference on Intelligent Virtual Agents (IVA Adjunct ’25)},
	year		=	{2025}}
```

### Acknowledgments

<sub>
This code is a modified version of the code at <a href="https://github.com/BenSaunders27/ProgressiveTransformersSLP">
https://github.com/BenSaunders27/ProgressiveTransformersSLP</a> associated to the following papers:</sub>

```
@inproceedings{saunders2020progressive,
	title		=	{Progressive Transformers for End-to-End Sign Language Production},
	author		=	{Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
	booktitle   =   {Proceedings of the European Conference on Computer Vision (ECCV)},
	year		=	{2020}}

@inproceedings{saunders2020adversarial,
	title		=	{Adversarial Training for Multi-Channel Sign Language Production},
	author		=	{Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
	booktitle   =   {Proceedings of the British Machine Vision Conference (BMVC)},
	year		=	{2020}}

@inproceedings{saunders2021continuous,
	title		=	{Continuous 3D Multi-Channel Sign Language Production via Progressive Transformers and Mixture Density Networks},
	author		=	{Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
	booktitle   =   {International Journal of Computer Vision (IJCV)},
	year		=	{2021}}
```

<sub>We thank the authors for this very useful implementation.</sub>