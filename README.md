# Elastic Interaction-based Loss
This is the official code for the paper "An Elastic Interaction-Based Loss Function for Medical Image Segmentation" presented at MICCAI 2020 (https://arxiv.org/abs/2007.02663).

## Update 2023.07
- Updated the code to Pytorch 1.11.0
- Fixed some bugs and improved performance

## Requirements
- Python 3.7 or higher
- Pytorch 1.11.0 or higher

## Usage
To train your model, run the following command:

```bash
python train.py --train_dataset '/your_training_data_path' --test_dataset '/your_test_data' --save_path '/save_model_path'
```
You can also specify other arguments such as batch size, learning rate, number of epochs, etc. See `train.py` for more details.

If you want to customize your Dataset: modify the `ImageToImage2D` in `./unet/dataset.py`. 

The elastic interaction loss file is located in `./unet` folder. You can import it and use it as a custom loss function for your segmentation model.

## Citation
If you find this code useful, please cite our paper:

```
@inproceedings{LanXZ20,
  author       = {Yuan Lan and
                  Yang Xiang and
                  Luchan Zhang},
  title        = {An Elastic Interaction-Based Loss Function for Medical Image Segmentation},
  booktitle    = {Medical Image Computing and Computer Assisted Intervention - {MICCAI}
                  2020 - 23rd International Conference, Lima, Peru, October 4-8, 2020,
                  Proceedings, Part {V}},
  series       = {Lecture Notes in Computer Science},
  volume       = {12265},
  pages        = {755--764},
  publisher    = {Springer},
  year         = {2020}
}
```

## Acknowledgements
This code is based on the Pytorch UNet template from https://github.com/cosmic-cortex/pytorch-UNet. We thank the authors for their work.


