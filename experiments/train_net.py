#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
from detectron2.engine import (
    AMPTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
    SimpleTrainer,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)

    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
        model, train_loader, optim
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)
    

def get_data_dics(dataset, split):
    """
    Args:
        dataset (str): name of the dataset
        split (str): name of the split, e.g. "train", "val", "test"
    """
    import json
    return json.load(open("data/{}/{}_{}.json".format(dataset, dataset, split)))


def custom_cfg(cfg, args):
    """
    Custom logic to setup the config and the environment.
    """

    # Register datasets
    if args.dataset == "iwp":
        args.things_classes = ["iwp"]
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    
    for d in ["train", "test"]:
        DatasetCatalog.register(
            args.dataset + "_" + d, lambda d=d: get_data_dics(args.dataset, d)
        )
        MetadataCatalog.get(args.dataset + "_" + d).set(
            thing_classes=args.things_classes
        )


    # custom logic to setup the config
    args.model = args.config_file.split("/")[-1].split(".")[0]
    
    if args.model == "sam":
        assert args.prompt is not None, "Prompt is required for SAM model"
        cfg.dataloader.prompt = args.prompt

    if args.dataset == "iwp":
        # custom model parameters
        if args.model != "sam": # sam model does not have mask rcnn head
            cfg.model.proposal_generator.anchor_generator.sizes = [[32], [64], [128], [256], [512]]
            cfg.model.roi_heads.num_classes = len(args.things_classes)

        #cfg.model.pixel_mean = [123.675, 116.28, 103.53]
        #cfg.model.pixel_std = [58.395, 57.12, 57.375]
           
        # customize training parameters
        cfg.train.output_dir = "experiments/output/{}/{}".format(args.model, args.dataset)
        # cfg.train.init_checkpoint = "checkpoints/{}.pkl".format(args.model)
        cfg.train.checkpointer.period = 50
        cfg.train.eval_period = 50
        cfg.train.seed = 1129 

        # use args.model to find the corresponding checkpoint
        # could be *.pkl or *.pth
        checkpoints = os.listdir("checkpoints")
        checkpoints = [c for c in checkpoints if args.model in c]
        cfg.train.init_checkpoint = "checkpoints/{}".format(checkpoints[0])

        # customize lr_multiplier parameters
        # customize dataloader parameters
        cfg.dataloader.train.dataset.names = args.dataset + "_train"
        cfg.dataloader.test.dataset.names  = args.dataset + "_test"
        cfg.dataloader.train.total_batch_size = 4 # images per gpu * num_gpus
        cfg.dataloader.evaluator.max_dets_per_image = 500
        cfg.dataloader.evaluator.output_dir = cfg.train.output_dir

        # learning rate
        cfg.optimizer.lr = cfg.optimizer.lr / 8 * args.num_gpus # linear scaling rule, 8 gpus in the pre-training phase
                                                               
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    
    return cfg


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg = custom_cfg(cfg, args)
    default_setup(cfg, args)
    
    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


def invoke_main() -> None:
    parser = default_argument_parser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--prompt", type=str, help="prompt type for SAM model")
    args = parser.parse_args()
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover