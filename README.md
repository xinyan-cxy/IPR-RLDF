# Learning from Mistakes: Iterative Prompt Relabeling for Text-to-Image Diffusion Model Training

This is the official code implementation of the paper [Learning from Mistakes: Iterative Prompt Relabeling for Text-to-Image Diffusion Model Training](https://arxiv.org/abs/2312.16204).

## Brief introduction

<div align="center">
    <img src="./images/IPR_overview.png" alt="figure1" style="zoom: 50%;" />
</div>

We propose Iterative Prompt Relabeling (IPR), a novel algorithm designed to enhance the alignment of images with text through
an iterative process of image sampling and prompt relabeling.

### Training Pipeline:

<div align="center">
<img src="./images/pipeline.png" alt="figure3" style="zoom: 40%;" />
</div>


**(1) Sampling Images from Diffusion Models:** sample images from a diffusion model conditioned on textual prompts. 

**(2) Prompt Relabeling:** detect the generated image to yield a bounding box; analyze the box to modify original prompts.

**(3) Detection-Based Loss Re-scaling:** apply a detection model to rescale the loss function. 

**(4) Iterative Training:** retrain the model with the updated dataset iteratively.

## Installation and Setup

***Environment*** This repo requires Pytorch=1.10.1 and torchvision.

Install the diffusers:

```
pip install diffusers
```

Then install the following packages for GLIP:

```
cd GLIP
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo opencv-python nltk scipy pycocotools 
pip install transformers 
pip install numpy==1.23.5
python setup.py build develop --user
```

***Backbone Checkpoints.*** Download the ImageNet pre-trained backbone checkpoints into the `GLIP/MODEL` folder.

```
mkdir MODEL
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/swin_tiny_patch4_window7_224.pth -O swin_tiny_patch4_window7_224.pth
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/swin_large_patch4_window12_384_22k.pth -O swin_large_patch4_window12_384_22k.pth
```

Then download the GLPT-L model:

```
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
```

## Training

Finetune the *stable-diffusion-2-1* for 2 iterations:

```
cd scripts
bash run_sdv2.sh
```

Finetune the *stable-diffusion-2-1* with LoRA for 2 iterations:

```
cd scripts
bash run_sdv2_lora.sh
```

Finetune the *stable-diffusion-xl-base-1.0* with LoRA for 2 iterations:

```
cd scripts
bash run_sdxl_lora.sh
```


## Citation

If you find our work useful for your research and applications, please kindly cite using this BibTeX:

```bib
@misc{chen2024learningmistakesiterativeprompt,
      title={Learning from Mistakes: Iterative Prompt Relabeling for Text-to-Image Diffusion Model Training}, 
      author={Xinyan Chen and Jiaxin Ge and Tianjun Zhang and Jiaming Liu and Shanghang Zhang},
      year={2024},
      eprint={2312.16204},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2312.16204}, 
}
```