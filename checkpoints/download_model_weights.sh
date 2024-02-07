#! /bin/bash

# download MViTv2 weights
wget -O mvitv2_mask_rcnn.pkl https://dl.fbaipublicfiles.com/detectron2/MViTv2/mask_rcnn_mvitv2_t_3x/f307611773/model_final_1a1c30.pkl

# download SAM weights
wget -O sam.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth