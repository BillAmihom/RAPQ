# RAPQ: Rescuing Accuracy for Power-of-Two Low-bit Post-training Quantization.
Pytorch implementation of RAPQ, IJCAI 2022. [link](http://arxiv.org/abs/2204.12322)

# Contact
Any question can be asked, please contract with E-mail: yhy@stu.pku.edu.cn

# Notice
- RAPQ provides the Power-of-Two quantization scheme for PTQ specially. Because of BRECQ's SOTA performance in PTQ area, this hub implements RAPQ based on [BRECQ](https://github.com/yhhhli/BRECQ) by Yuhang Li @yhhhli. .

- Please download the pretrained models before running this program! 
# Getting start

## 1. Download pretrained models.(Thanks for pretrained models provided by @yhhhli ! )

[resnet18](https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar)

[resnet50](https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet50_imagenet.pth.tar)

[mobilenetv2](https://github.com/yhhhli/BRECQ/releases/download/v1.0/mobilenetv2.pth.tar)

[regnetx_600m](https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_600m.pth.tar)

[regnetx_3200m](https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_3200m.pth.tar)

After downloading, please put it into "~/.cache/torch/checkpoints" of your user path

## 2. Prepare the Environment

This program is done in the Pytorch framework, so please prepare the environment first!

## 3. Prepare the Dataset

ImageNet dataset is also Necessary!

## 4. All ready,GO!

- Use Naive Powers-of-Two PTQ:

`CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --data_path /path/to/ImageNet/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 4 --act_quant --test_before_calibration`

- Use RAPQ Quick Mode:

`CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --RAPQ --data_path /path/to/ImageNet/ --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 4 --act_quant --test_before_calibration`

- Use RAPQ:

`CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --RAPQ --data_path /path/to/ImageNet/ --arch mobilenetv2 --n_bits_w 2 --iters_w 80000 --channel_wise --n_bits_a 4 --act_quant --test_before_calibration`


# LICENSE
RAPQ is release under MIT license. 


