#  Pix2Pix Image-to-Image Translation using TensorFlow 2.x

This project implements the **Pix2Pix** model — a powerful **Image-to-Image Translation GAN (Generative Adversarial Network)** using TensorFlow 2.x.  
It converts one type of image into another (e.g., sketches → photos, maps → buildings, etc.) using paired datasets.
author shankar sutar
---

##  Overview

Pix2Pix is a **conditional GAN** that learns a mapping from an input image to an output image.  
It consists of:
- **Generator (U-Net)**: Translates the input image into the desired target image.
- **Discriminator (PatchGAN)**: Classifies image patches as real or fake to enforce local realism.

This project trains the Pix2Pix model using the **Facades dataset**, where the task is to convert architectural labels into building facades.

---

##  Features

 TensorFlow 2.x implementation (Tested on TensorFlow 2.19+)  
 Clean U-Net based generator and PatchGAN discriminator  
 Automatic dataset download and extraction  
 Works directly on Google Colab  
 Simple visualization of results  
 Fully commented and modular code  

---

##  Dataset

- Dataset: **Facades**  
- Source: [Berkeley Pix2Pix Datasets](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)
- It contains paired images:
  - Left half → Input (label)
  - Right half → Target (real facade)

The dataset is automatically downloaded and extracted using TensorFlow utilities.

---

##  Requirements

Before running the notebook or script, ensure you have the following installed:

```bash
pip install tensorflow==2.19 matplotlib
