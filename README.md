# When Schrödinger Bridge Meets Real-World Image Dehazing with Unpaired Training (ICCV2025)
Recent advancements in unpaired dehazing, particularly those using GANs, show promising performance in processing real-world hazy images. However, these methods tend to face limitations due to the generator's limited transport mapping capability, which hinders the full exploitation of their effectiveness in unpaired training paradigms. To address these challenges, we propose DehazeSB, a novel unpaired dehazing framework based on the Schrödinger Bridge. By leveraging optimal transport (OT) theory, DehazeSB directly bridges the distributions between hazy and clear images. This enables optimal transport mappings from hazy to clear images in fewer steps, thereby generating high-quality results. To ensure the consistency of structural information and details in the restored images, we introduce detail-preserving regularization, which enforces pixel-level alignment between hazy inputs and dehazed outputs. Furthermore, we propose a novel prompt learning to leverage pre-trained CLIP models in distinguishing hazy images and clear ones, by learning a haze-aware vision-language alignment. Extensive experiments on multiple real-world datasets demonstrate our method's superiority.

## Model Architecture

![Model Overview](https://github.com/ywxjm/DehazeSB/blob/main/image/figure2.jpg)

### Installation

python==3.10
pip install -r requirements.txt

## Download training Dataset 

Our training dataset can be downloaded in train_data.txt

## Pretrained weight 

The pre-trained weights can be found in the **./pretrained** directory.  


## Training / testing on Dataset
Start the training process by running the following command:
```sh
bash /code/train_DehazeSB.sh 
```
Run the testing script:
```sh
bash /code/test_dehaze.sh
```

If you wish to train or test on your own dataset, simply modify the paths in the UnpairedDataSet and ValidationHaze2020 classes within data/unpaired_dataset.py.

### Thanks for the code provided by:

UNSB: https://github.com/cyclomon/UNSB
DSCNet: https://github.com/yaoleiqi/dscnet
CUNSB: https://github.com/Retinal-Research/CUNSB-RFIE



