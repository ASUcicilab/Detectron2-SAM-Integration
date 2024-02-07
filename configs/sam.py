import torch
import numpy as np 

from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.structures import Instances, Boxes 

from .common.coco_loader_sam import dataloader 
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

from typing import Any, Dict, List, Tuple


class CustomSam(Sam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    """
    custom SAM model for training purposes
    add loss function to the forward method
    image_encoder and prompt_encoder are not trained in default (with torch.no_grad())
    """

    def dice_loss(self, pred, target):
        smooth = 1e-6
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        return 1 - (2 * intersection + smooth) / (union + smooth)
    

    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            multimask_output: bool = False,
            required_grad: bool = False,

    ) -> List[Dict[str, torch.Tensor]]:
        # move batched_input to device
        batched_input = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()} for x in batched_input]
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        with torch.no_grad():
          image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            with torch.no_grad():
              sparse_embeddings, dense_embeddings = self.prompt_encoder(
                  points=points,
                  boxes=image_record.get("boxes", None),
                  masks=image_record.get("mask_inputs", None),
              )
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            # TODO: remove this

            if self.training: 
              m = torch.nn.Sigmoid()
              masks = m(masks)
            else:
              masks = masks > self.mask_threshold
            
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )

        if not self.training:
            # transferm to standard detectron2 model output format for evaluation
            # the default batch_size for inference is 1
            transformed_outputs = []
            transformed_outputs.append(
                {"instances": Instances(image_size=batched_input[0]["instances"].image_size)}
            )
            transformed_outputs[0]["instances"].set(
                "pred_masks", outputs[0]["masks"].squeeze(1)
            )
            # resize boxes to the original size
            rh, rw = batched_input[0]["height"] / batched_input[0]["instances"].image_size[0], batched_input[0]["width"] / batched_input[0]["instances"].image_size[1]  
            gt_boxes = batched_input[0]["instances"].gt_boxes.tensor
            gt_boxes = gt_boxes * torch.tensor([rw, rh, rw, rh], device=gt_boxes.device)

            transformed_outputs[0]["instances"].set(
                "pred_boxes", Boxes(gt_boxes)
            )
            transformed_outputs[0]["instances"].set(
                "pred_classes", batched_input[0]["instances"].gt_classes
            )
            transformed_outputs[0]["instances"].set(
                "scores", outputs[0]["iou_predictions"].squeeze(1)
            )
            
            return transformed_outputs
        
        # return loss if training 
        # calculate loss for training
        loss_dict = torch.zeros(1, requires_grad=True).to(self.device)
        for image_record, output in zip(batched_input, outputs):
            pred_masks = output["masks"]
            pred_masks = pred_masks.squeeze(1)
            gt_masks = image_record["instances"].gt_masks.polygons
            gt_masks = [polygons_to_bitmask(p, image_record["instances"].image_size[0], image_record["instances"].image_size[1]) 
                        for p in gt_masks]
            gt_masks = np.stack(gt_masks, axis=0)
            gt_masks = torch.as_tensor(gt_masks, dtype=torch.float32, device=pred_masks.device) 
            # resize gt_masks to the same size as pred_masks
            gt_masks = torch.nn.functional.interpolate(gt_masks.unsqueeze(0), size=pred_masks.shape[-2:], mode="nearest").squeeze(0)
            loss_dict += self.dice_loss(pred_masks, gt_masks)
            
        loss_dict /= len(batched_input)

        return loss_dict


# sam_vit_h
model = L(CustomSam)(
    image_encoder=L(ImageEncoderViT)(
        depth=32,
        embed_dim=1280,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=16,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[7, 15, 23, 31],
        window_size=14,
        out_chans=256,
    ),
    prompt_encoder=L(PromptEncoder)(
        embed_dim=256,
        image_embedding_size=(1024 // 16, 1024 // 16), # image_size // vit_patch_size
        input_image_size=(1024, 1024), 
        mask_in_chans=16,
    ),
    mask_decoder=L(MaskDecoder)(
        num_multimask_outputs=3,
        transformer=L(TwoWayTransformer)(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ),
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
)


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.ddp.find_unused_parameters = True

dataloader.train.total_batch_size = 64

# 36 epochs
train.max_iter = 67500
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[52500, 62500, 67500],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {
    "pos_embed": {"weight_decay": 0.0},
    "rel_pos_h": {"weight_decay": 0.0},
    "rel_pos_w": {"weight_decay": 0.0},
}
optimizer.lr = 1.6e-4
