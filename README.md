# Self-Supervised Image Denoising of Third Harmonic Generation Microscopic Images of Human Glioma Tissue by Transformer-based Blind Spot (TBS) Network

**Yuchen Wu**, **Siqi Qiu**, **Marie Louise Groot**, **Zhiqing Zhang**

**Abstract**:
_Third harmonic generation (THG) microscopy shows great potential for instant pathology of brain tumor tissue during surgery. However, due to the maximal permitted exposure of laser intensity and inherent noise of the imaging system, the noise level of THG image is relatively high, which affects subsequent feature extraction analysis. Denoising of THG images is challenging for modern deep-learning based methods because of the rich morphologies contained and the difficulty in obtaining the noise-free counterparts. To address this, in this work, we propose an unsupervised deep-learning network for denoising of THG images which combines a self-supervised blind spot method and a U-shape Transformer using a dynamic sparse attention mechanism._

**Official Pytorch implementation of our model.**

## Python Requirements

This code was tested on:

- Python 3.8.*
- Pytorch 1.8.*

## Preparing Training Dataset

THG images in the training set are from fresh, unprocessed tissue samples surgically resected from 23 patients having  grade II-IV diffuse glioma.
The field of view of each THG image is 273 × 273 μm2 (1125 × 1125 pixels) and the intensities of all images were scaled to [0, 255].Because of limited computational resource, each THG image having 1125 × 1125 pixels was divided into 5×5 smaller images having 256 × 256 pixels, with partial overlap.


## Training

To train a network, run:

```bash
python train.py 
```
- selected optional arguments:
  - `data_dir` Path to the training set
  - `val_dirs` Path to the validation sets
  - `noisetype` Distribution of image noise, choosing from `gauss25`, `gauss5_50`, `poisson30`, or `poisson5_50`
  - `save_model_path` Base-path to the saved files
  - `log_name` Path to the saved files
  
## Demo

https://github.com/YCWu-nk/Denoising_THG/assets/154118877/8dd906dd-ec37-4b8a-8a05-583e4eec1ce9



