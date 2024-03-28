# ___***SmartControl: Enhancing ControlNet for Handling Rough Visual Conditions***___


<a href='https://arxiv.org/abs/2308.06721'><img src='https://img.shields.io/badge/arXiv-2302.08453-b31b1b.svg?style=flat-square'></a> 

---


## Introduction

For handling the disagreements between the text prompts and rough visual conditions, we propose a novel text-to-image generation method dubbed SmartControl, which is designed to align well with the text prompts while adaptively keeping useful information from the visual conditions. Specifically, we introduce a control scale predictor to identify conflict regions between the text prompt and visual condition and predict spatial adaptive scale based on the degree of conflict. The predicted control scale is employed to adaptively integrate the information from rough conditions and text prompts to achieve the flexible generation.

![arch](assets/figs/method.png)

## Release
- [2024/3/31] 🔥 We release the code and models for depth condition.


## Installation

```
pip install -r requirements.txt
```

## Download Models

you can download our control scale predictor models from [here](https://drive.google.com/file/d/1iu7eE-XtxFkIupvJyesQnustuujXAW61/view?usp=drive_link). To run the demo, you should also download the following models:
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [SG161222/Realistic_Vision_V5.1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE)
- [ControlNet models](https://huggingface.co/lllyasviel)
- [realisticvision-negative-embedding](https://civitai.com/models/36070/negative-embedding-for-realistic-vision-v20)

## How to Use


- [**smartcontrol_demo**](smartcontrol_demo.ipynb)


- [**smartcontrol_ipadapter_demo**](smartcontrol_ipadapter_demo.ipynb)



## Citation
If you find SmartControl useful for your research and applications, please cite using this BibTeX:
```bibtex

```
