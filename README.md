# Detectron2 SAM Integration 
This repository demonstrates the integration of th SAM model with the Detectron2 framework. 

## Installation
To run this project, you must have PyTorch and Detectron2 installed. Follow these guides for installation:
* PyTorch: [Installation Guide](https://pytorch.org/get-started/locally/) 
* Detectron2 [Installation Guide](https://github.com/facebookresearch/detectron2) 

## Folder Structure
The repository is organized as follows:
- `checkpoints/`: Stores pre-trained model weights (`.pth` or `.pkl`).  
- `configs/`: Contains configuration files for models, data loaders, trainers, optimizers, etc., where most integration adjustments are made.
- `data/`: Holds dataset files, specifically `dataset_train.json` and `dataset_test.json`, for Detectron2 data loading.
- `experiments/`: Includes scripts for training and evaluation, alongside logs, models, and outputs.
- `utils/`: 

## Quick Start
To begin using the integrated models, follow these commands:

For the MViTv2 model:

```bash
python experiments/train_net.py --config-file configs/mvitv2_mask_rcnn.py --num-gpus 1 --dataset iwp 
```

For the SAM model (specify `box` or `point` for the prompt):

```bash
python experiments/train_net.py --config-file configs/sam.py --num-gpus 1 --dataset iwp --prompt box (or point)
```

The `train_net.py` script allows further customization through the `custom_cfg` function, enabling adjustments to the evaluation period, output directory, batch size, etc.

## Detailed Information
Key components of a Detectron2 training setup include:

- dataloader: Manages dataset loading and data augmentation. Customize data format for models like SAM by defining a custom `DatasetMapper`.
- model: Customized in `configs/model.py`, handling training mode (loss return) and evaluation mode (predictions return).
- optimizer: Detectron2's default optimizer.
- trainer: Detectron2's default trainer.
- evaluator: COCO evaluator for assessment.

## Integration
The key to integrating a custom model lies in successfully connecting its inputs and outputs. The model's interface with the framework may not align with Detectron2's standard formats. So, 

1. Input connection: customize a "DatasetMapper" to apply data augmentation and format the data to meet the model's requirements.

2. Output connection: Ensure the model's output aligns with Detectron2's expectations. Adjust the model's forward function such that
    * During training, it returns the loss. 
    * During evaluation, it returns the standard prediction format for the COCO evaluator.

## Known Issues
- By using current data augmentation, training the SAM model will result in errors. The error is due to the random cropping of the images which makes SAM to predict zero masks. To fix this issue, you can remove the random cropping from `configs/common/coco_loader_sam.py` file.

- The current SAM training only takes the first 100 masks from each image. This is due to the memory constraints. To change this, you can modify the `CustomDatasetMapper` in `configs/common/coco_loader_sam.py`. 



