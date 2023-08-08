# Under-Display Camera Image Restoration with Scattering Effect


> **Abstract:** 
The under-display camera (UDC) provides consumers with a full-screen visual experience without any obstruction due to notches or punched holes. However, the semi-transparent nature of the display inevitably introduces the severe degradation into UDC images. In this work, we address the UDC image restoration problem with the specific consideration of the scattering effect caused by the display. We explicitly model the scattering effect by treating the display as a piece of homogeneous scattering medium. With the physical model of the scattering effect, we improve the image formation pipeline for the image synthesis to construct a realistic UDC dataset with ground truths. To suppress the scattering effect for the eventual UDC image recovery, a two-branch restoration network is designed. More specifically, the scattering branch leverages global modeling capabilities of the channel-wise self-attention to estimate parameters of the scattering effect from degraded images. While the image branch exploits the local representation advantage of CNN to recover clear scenes, implicitly guided by the scattering branch. Extensive experiments are conducted on both real-world and synthesized data, demonstrating the superiority of the proposed method over the state-of-the-art UDC restoration techniques.

## Getting started

### Install

We test the code on PyTorch 1.12.1 + CUDA 11.3 + cuDNN 8.3.2.

1. Create a new conda environment
```
conda create -n pt1121 python=3.9
conda activate pt1121
```

2. Install dependencies
```
conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Download
### Prepare Dataset

Download and unzip the [Dataset](https://drive.google.com/drive/folders/14Zp2Ff4Ke5491qmyb-BXIx77l9jNMxkG?usp=sharing), and then copy them to `data`.

### Download Pre-trained Model

Download and unzip our [pre-trained model](https://drive.google.com/drive/folders/1x1MB88uUBGSlUBc68UT_Kg7abropMS0j?usp=sharing), and then copy them to `save_models/UDC`.


The final file path should be the same as the following:

```
┬─ save_models
│   └─ UDC
│       ├─ SRUDC-f.pth
│       └─ SRUDC-l.pth
└─ data
    ├─ train
    │   ├─ HQ_syn
    │   │   └─ ... (image filename)
    │   └─ LQ_syn
    │       └─ ... (corresponds to the former)
    └─ test
        ├─ HQ_syn
        │   └─ ... (image filename)
        └─ LQ_syn
            └─ ... (corresponds to the former)
```

## Training and Evaluation

### Train

You can modify the training settings for each experiment in the `configs` folder.
Then run the following script to train the model:

```sh
python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 --master_addr="master ip address" --master_port=6379 train.py --model SRUDC_f --use_mp --use_ddp
```

### Test

Run the following script to test the trained model:

```sh
python test.py
```


## Acknowledgement

Our code is based on [gUnet](https://github.com/IDKiro/gUNet). We thank the authors for sharing the codes.
