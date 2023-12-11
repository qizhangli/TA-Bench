# TA-Bench

**Towards Evaluating Transfer-based Attacks Systematically, Practically, and Fairly**\
*Qizhang*, *Yiwen Guo*, *Wangmeng Zuo*, *Hao Chen*\
NeurIPS 2023\
[https://arxiv.org/abs/2311.01323](https://arxiv.org/abs/2311.01323)

TA-Bench is a comprehensive benchmark of transfer-based attacks, which allowing researchers to compare a variety of methods in a fair and reliable manner.

## Features
**Methods**

*Input Augmentation and Optimizer.*\
I-FGSM, PGD, DI $^2$-FGSM, TI-FGSM, SI-FGSM, Admix, MI-FGSM, NI-FGSM, PI-FGSM, UN (adding uniform noise to the input), DP (randomly dropping some patches of the perturbation).

*Gradient Computation.*\
TAP, NRDM, FDA, ILA, SGM, ILA++, LinBP, ConBP, SE, FIA, PNA, NAA, VT, IR, TAIG.


*Substitute Model Training.*\
RFA, LGV, DRA, MoreBayesian.

*Generative Modeling.*\
CDA, GAPF, TTP, BIA, BIA+RN, BIA+DA, C-GSP.

**Dataset**

5,000 randomly selected images from the ImageNet validation set, that could be correctly classified by 10 victim models (including ResNet-50, VGG-19, Inception v3, EfficientNetV2-M, ConvNeXt-B, ViT-B, DeiT-B, BEiT-B, Swin-B, and MLP Mixer-B).

**Models**

*Source Models:*\
ResNet-50, VGG-19, Inception v3, EfficientNetV2-M, ConvNeXt-B, ViT-B, DeiT-B, BEiT-B, Swin-B, and MLP Mixer-B.

*Victim Models:*\
TA-Bench supports all models in ```timm``` to play as the victim model. Besides, you can also provide your own model as the victim model.

## Quick Start
### Requirements
torch==1.12.0\
torchvision==0.13.0\
timm==0.6.11\
ml_collections\
absl-py

### Preparation

#### Dataset

ImageNet should be prepared into the following structure:

```
ILSVRC2012_img_val
├── ILSVRC2012_val_00023809.JPEG
├── ILSVRC2012_val_00010482.JPEG
├── ILSVRC2012_val_00028406.JPEG
├── ILSVRC2012_val_00043992.JPEG
├── ILSVRC2012_val_00029651.JPEG
...
```

We have provided a list of the selected 5,000 images in ```data/rand_5000.csv```. But you can choose benign samples yourself and write them into a .csv file, with the format as follows:
```
class_index,class,image_name
915,n04613696,ILSVRC2012_val_00023809.JPEG
368,n02483362,ILSVRC2012_val_00010482.JPEG
804,n04254120,ILSVRC2012_val_00028406.JPEG
...
```

#### Models
Some methods require their own trained models, such as "substitute model training" methods and "generative modeling" methods. Download these models in [Google Drive](TBD).

### Attack
To generate adversarial examples using {method} and {substitute_model}, and save the adversarial examples in {save_dir}:
```
python3 attack.py --config="configs/${method}.py" \
                  --config.save_dir ${save_dir} \
                  --config.model_name ${substitute_model}
```

You can set the ```method``` as the file names in configs, for instance, ```linbp```, ```linbp_newbackend```, ```newbackend```, and ```ifgsm```.

You can change some hyperparameters by appending ```--config.{hyper_parameter} {value}```. For instance, adding ```--config.epsilon 16``` to change the $\epsilon$ to 16/255, or adding ```--config.model_path {model_path}``` to specify the path of the model file when you performing MoreBayesian method. More hyperparameters that can be changed can be found in the ```configs```.

After the attack, the adversarial examples are saved in .npy files named ```batch_{index}.npy```, along with a ```labels.npy``` file that records the ground truth of these examples. The ```batch_{index}.npy``` file contains a batch of adversarial examples formatted as a NumPy array, with the shape [N, C, H, W] and the data type np.uint8.

### Evaluation
To evaluate the performance of the generated adversarial examples, use the following Python snippet:
```
from evaluation import Evaluation
evaluator = Evaluation(data_dir=adv_img_dir, mode=mode, victims=victims,
                       batch_size=batch_size, log_dir=log_dir)
evaluator.evaluate()
```
The {mode} can be set as "standard", "custom-timm", and "custom-custom".

```mode="standard"```, the victim models are ResNet-50, VGG-19, Inception v3, EfficientNetV2-M, ConvNeXt-B, ViT-B, DeiT-B, BEiT-B, Swin-B, and MLP Mixer-B, and ```victims``` need not to be set.

```mode="custom-timm"```, you can select the victim models from ```timm```, for instance, set ```victims=["tv_resnet152", "resnext101_64x4d"]```.

```mode="custom-custom"```, you can also provide your own victim models by setting ```victims=[victim_dict_1, victim_dict_2]```. For each ```victim_dict```, prepare them as ```{"model_name": model_name, "model": model, "preprocessing": preprocessing}```, in which the ```model``` is your victim model, and ```preprocessing``` is the pre-processing pipeline.

## Citation
```
@inproceedings{li2023towards,
  title={Towards Evaluating Transfer-based Attacks Systematically, Practically, and Fairly},
  author={Li, Qizhang and Guo, Yiwen and Zuo, Wangmeng and Chen, Hao},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
