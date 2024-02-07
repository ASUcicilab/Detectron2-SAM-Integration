import torch 
import copy 
import numpy as np
from omegaconf import OmegaConf
from typing import List, Union
from detectron2.data import detection_utils as utils

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from segment_anything.utils.transforms import ResizeLongestSide


class CustomDatasetMapper(DatasetMapper):
    """
    Customized mapper for SAM finetuning.
    batched_input (list(dict)): A list over input images, each a dictionary with
    the following keys. A prompt key can be excluded if it is not present.
        'image': The image as a torch tensor in 3xHxW format, already transformed for input to the model.
        'original_size': (tuple(int, int)) The original size of the image before transformation, as (H, W).
        'point_coords': (torch.Tensor) Batched point prompts for this image, with shape BxNx2.
            Already transformed to the input frame of the model.
        'point_labels': (torch.Tensor) Batched labels for point prompts, with shape BxN.
        'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
            Already transformed to the input frame of the model.
    """
    def __init__(
            self, 
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            use_instance_mask: bool = False,
            instance_mask_format: str = "polygon",
            prompt: str = "box",
    ):
        super().__init__(
            is_train=is_train,
            augmentations=augmentations,
            image_format=image_format,
            use_instance_mask=use_instance_mask,
            instance_mask_format=instance_mask_format,
        )
        self.prompt = prompt

    
    
    def __call__(self, dataset_dict):
        # basically the same as the parent class, but we need annotations for both training and testing

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        self._transform_annotations(dataset_dict, transforms, image_shape)

        # take the first 100 instances for training to fit into the memory
        # may remove this line if you have enough memory
        if self.is_train:
            dataset_dict["instances"] = dataset_dict["instances"][:100]

        # custom SAM input
        dataset_dict["original_size"] = (dataset_dict["height"], dataset_dict["width"])
                                 
        point_coords = [] 
        for box in dataset_dict["instances"].gt_boxes.tensor:
            x_center, y_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            point_coords.append([x_center, y_center])

        if self.prompt == "box":
            dataset_dict["boxes"] = dataset_dict["instances"].gt_boxes.tensor
        elif self.prompt == "point":
            dataset_dict["point_coords"] = torch.as_tensor(point_coords, dtype=torch.float32).unsqueeze(1)
            dataset_dict["point_labels"] = torch.ones(len(point_coords), 1) # all foreground

        return dataset_dict
 

dataloader = OmegaConf.create()

dataloader.prompt = "box"

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train"),
    mapper=L(CustomDatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomApply)(
                tfm_or_aug=L(T.AugmentationList)(
                    augs=[
                        L(T.ResizeShortestEdge)(
                            short_edge_length=[400, 500, 600], sample_style="choice"
                        ),
                        # L(T.RandomCrop)(crop_type="absolute_range", crop_size=(384, 600)),
                    ]
                ),
                prob=0.5,
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="RGB",
        use_instance_mask=True,
        prompt="${...prompt}",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(CustomDatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
        prompt="${...prompt}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
