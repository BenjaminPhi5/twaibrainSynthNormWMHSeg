print("loading....")

# general utils
import torch
import pandas as pd
import numpy as np
import json
import argparse
import os

# data
from twaibrain.braintorch.data import SingleVisitDataset_V1, MultiIndependentVisitDataset_V1, SitkImageDataset_V1, RandomSubjectVisitDataset_V1, SingleVisitDatasetInRam_V1, MultiIndependentVisitDatasetInRam_V1, RandomSubjectVisitDatasetInRam_V1
from torch.utils.data import DataLoader
from twaibrain.braintorch.data.datasets.reshuffled_datasets import reshuffle_datasets

# augmentation
from twaibrain.braintorch.data.datasets.transformed_dataset import TransformedDataset
from twaibrain.braintorch.augmentation.nnunet_augmentations_v2 import simple_augmentations, simple_augmentations_with_biasfield, nnunet_augmentations, aggressive_augmentations, get_val_transforms

# model architecture
from twaibrain.braintorch.models.nnUNet.nnUNetV2_model_loader import get_network_from_plans
from twaibrain.braintorch.models.ssn import SSN_Wrapped_Deep_Supervision, SSN_Wrapped_Deep_Supervision_LLO, Hierarchical_SSN_with_ConvRefine, Hierarchical_SSN_with_ConvSpatialAttention

# fitting code
from twaibrain.braintorch.fitting_and_inference.get_trainer import get_trainer
from twaibrain.braintorch.fitting_and_inference.get_scratch_dir import scratch_dir
from twaibrain.braintorch.fitting_and_inference.optimizer_constructor import OptimizerConfigurator
from twaibrain.braintorch.fitting_and_inference.lightning_fitter import StandardLitModelWrapper

# loss functions
from twaibrain.braintorch.losses.ssn_losses_V2 import SSN_ComboLoss, SSNDeepSupervisionLoss, LLOMultiDeepSupervisionLoss
from twaibrain.braintorch.losses.xent import dice_xent_loss_V2_BCEWL, dice_xent_loss_V2
from twaibrain.braintorch.losses.generic_deep_supervision import DeepSupervisionLoss

import warnings
warnings.filterwarnings("ignore", module="torchio")
"""
filtering out the following warning because I can't even find anywhere where a Subject is being
returned as opposed to a torch tensor, and the data appears correctly in the batch so I'm not sure what is going on...
Warning:
Using TorchIO images without a torchio.SubjectsLoader in PyTorch >= 2.3 might have unexpected consequences, e.g., the collated batches will be instances of torchio.Subject with 5D images. Replace your PyTorch DataLoader with a torchio.SubjectsLoader so that the collated batch becomes a dictionary, as expected. See https://github.com/TorchIO-project/torchio/issues/1179 for more context about this issue.
"""

def construct_parser():
    parser = argparse.ArgumentParser(description = "basic_lbc_model_fit")
    
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-nw', '--num_workers', default=8, type=int)
    parser.add_argument('-a', '--augmentation', default='simple', type=str)
    parser.add_argument('-rs', '--train_val_reshuffle_seed', default=None, type=int)
    
    parser.add_argument('-mp', '--matmul_precision', default='highest', type=str)
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    parser.add_argument('-wd', '--weight_decay', default=0.0001, type=float)
    parser.add_argument('-i', '--max_iters', default=800, type=int)
    parser.add_argument('-ep', '--early_stop_patience', default=300, type=int)
    parser.add_argument('-es', '--early_stop', action='store_true')
    parser.add_argument('-gb', '--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('-swa', '--stochastic_weight_average', action='store_true')
    parser.add_argument('-m', '--model_name', default='model', type=str)
    parser.add_argument('-ckpt', '--ckpt_dir', default='.', type=str)
    parser.add_argument('-gc', '--gradient_clip', action='store_true')
    parser.add_argument('-gz', '--global_zscore', action='store_true')
    parser.add_argument('-gm', '--global_minmax', action='store_true')
    parser.add_argument('-bce', '--bce_loss', action='store_true')
    parser.add_argument('--clamp_logits', action='store_true')
    parser.add_argument('--loss_epsilon', default=1e-5, type=float)
    parser.add_argument('--nn_optim', action='store_true')

    return parser


def main(args):

    ####################################################################
    # load data
    ####################################################################
    lbc_val = SingleVisitDatasetInRam_V1(
    # lbc_val = SingleVisitDataset_V1(
        dataset_folder="/home/s2208943/preprocessed_data/",
        config_name="synthetic_augmentation",
        ds_name="LBC",
        visit_key="ses",
        ds_experiment_name='',
        split='val',
        resample=True,
        out_spacing=[1,1,2],
        fit_to_mask=True,
        mask_pad=0,
        fit_to_shape=True,
        output_shape=[80,192,160],
        key_renames = {'wave2-WMH':'label', 'wave3-WMH':'label', 'wave3-WMH':'label', 'wave4-WMH':'label', 'wave5-WMH':'label'},
        print_subject=False,
    )

    lbc_train = SingleVisitDatasetInRam_V1(
    # lbc_train = SingleVisitDataset_V1(
        dataset_folder="/home/s2208943/preprocessed_data/",
        config_name="synthetic_augmentation",
        ds_name="LBC",
        visit_key="ses",
        ds_experiment_name='',
        split='train',
        resample=True,
        out_spacing=[1,1,2],
        fit_to_mask=True,
        mask_pad=0,
        fit_to_shape=True,
        output_shape=[80,192,160],
        key_renames = {'wave2-WMH':'label', 'wave3-WMH':'label', 'wave3-WMH':'label', 'wave4-WMH':'label', 'wave5-WMH':'label'},
        print_subject=False,
                           )

    # if args.train_val_reshuffle_seed is not None:
    #     print("reshuffling")
    #     print(len(lbc_train), len(lbc_val))
    #     lbc_train, lbc_val = reshuffle_datasets(lbc_train, lbc_val, args.train_val_reshuffle_seed)
    #     print(len(lbc_train), len(lbc_val))
        

    ####################################################################
    # add augmentation
    ####################################################################
    val_transforms = get_val_transforms()
    if args.augmentation == "simple":
        transforms = simple_augmentations()
        print("simple augmentation")

    elif args.augmentation == "simple_perlin_bias":
        transforms = simple_augmentations_with_biasfield(bias_field='perlin')
        print("simple augmentation with perlin bias field")

    elif args.augmentation == "simple_synthseg_bias":
        transforms = simple_augmentations_with_biasfield(bias_field='synthseg')
        print("simple augmentation with synthseg bias field")
        
    elif args.augmentation == "nnunet":
        transforms = nnunet_augmentations()
        print("nnunet augmentation")
        
    elif args.augmentation == "nnunet_synthetic":
        transforms = nnunet_augmentations(add_synthetic=True, synthetic_realistic=False)
        print("nnunet synthetic (unrealistic mode) augmentation")

    elif args.augmentation == "nnunet_synthetic_realistic":
        transforms = nnunet_augmentations(add_synthetic=True, synthetic_realistic=True)
        print("nnunet synthetic augmentation but realistic synthetic images")
        
    elif args.augmentation == "synthetic_aggressive":
        transforms = aggressive_augmentations(synthetic_realistic=True, global_minmax=args.global_minmax, global_zscore=args.global_zscore)
        print("synthetic aggressive augmentation")
        
    elif args.augmentation == "synthetic_aggressive_2drot":
        transforms = aggressive_augmentations(synthetic_realistic=True, axial_rot=True, global_minmax=args.global_minmax, global_zscore=args.global_zscore)
        print("synthetic aggressive with 2d rotations augmentation")

    elif args.augmentation == "synthetic_unrealistic_aggressive":
        transforms = aggressive_augmentations(synthetic_realistic=False, global_minmax=args.global_minmax, global_zscore=args.global_zscore)
        print("synthetic unrealistic aggressive augmentation")
        
    elif args.augmentation == "synthetic_unrealistic_aggressive_2drot":
        transforms = aggressive_augmentations(synthetic_realistic=False, axial_rot=True, global_minmax=args.global_minmax, global_zscore=args.global_zscore)
        print("synthetic unrealistic aggressive with 2d rotations augmentation")

    elif args.augmentation == "aggressive_non_synthetic":
        transforms = aggressive_augmentations(add_synthetic=False, global_minmax=args.global_minmax, global_zscore=args.global_zscore)
        print("synthetic aggressive augmentation")
        
    else:
        raise ValueError(f"augmentation type: {args.augmentation} is unknown")

    transformed_train = TransformedDataset(
        lbc_train,
        transforms,
    )

    transformed_val = TransformedDataset(
        lbc_val,
        val_transforms
    )

    ####################################################################
    # configure model
    ####################################################################
    config_file_M = "/home/s2208943/projects/twaibrain/twaibrain/braintorch/models/nnUNet/cvd_configs/nnUNetResEncUNetMPlans.json"
    model_config = config_file_M
    with open(model_config) as f:
            model_config = json.load(f)

    dims = "3d_fullres"
    config = model_config['configurations'][dims]['architecture']
    network_name = config['network_class_name']
    kw_requires_import = config['_kw_requires_import']

    model = get_network_from_plans(
        arch_class_name=network_name,
        arch_kwargs=config['arch_kwargs'],
        arch_kwargs_req_import=kw_requires_import,
        input_channels=2,
        output_channels=1 if args.bce_loss else 2,
        allow_init=True,
        deep_supervision=True,
    )

    train_dl = DataLoader(transformed_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=False)
    val_dl = DataLoader(transformed_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=False)

    ####################################################################
    # configure loss
    ####################################################################
    if args.bce_loss:
        loss = DeepSupervisionLoss(dice_xent_loss_V2_BCEWL(dice_weight=1, xent_weight=1, clamp=args.clamp_logits, dice_epsilon=args.loss_epsilon))
    else:
        loss = DeepSupervisionLoss(dice_xent_loss_V2(dice_weight=1, xent_weight=1, clamp=args.clamp_logits, dice_epsilon=args.loss_epsilon))
        

    ####################################################################
    # configure fitter
    ####################################################################
    print(f"batch size and workers: {args.batch_size} {args.num_workers}")
    print(f"accumulate: {args.accumulate_grad_batches}")
    print(f"max iters and early stop and early stop patience: {args.max_iters} {args.early_stop} {args.early_stop_patience}")

    if args.matmul_precision != "highest":
        torch.set_float32_matmul_precision(args.matmul_precision)

    if args.nn_optim:
        optim_configurator = OptimizerConfigurator(f"SGD lr:{args.learning_rate} weight_decay:{args.weight_decay} nesterov:True momentum:0.99", f"PolynomialLR total_iters:{args.max_iters} power:0.9")
    else:
        optim_configurator = OptimizerConfigurator(f"Adam lr:{args.learning_rate} weight_decay:{args.weight_decay}", f"PolynomialLR total_iters:{args.max_iters} power:0.9")
    
    litmodel = StandardLitModelWrapper(
        model,
        loss,
        optim_configurator,
        check_finiteness_of_data=False,
    )
    trainer = get_trainer(
        args.max_iters,
        os.path.join(args.ckpt_dir, args.model_name),
        save_top_k=1, 
        early_stop_patience=args.early_stop_patience,
        use_early_stopping=args.early_stop,
        accumulate_grad_batches=args.accumulate_grad_batches, 
        scheduled_accumuate_gradients=False,
        do_gradient_clip=args.gradient_clip,
        gradient_clip_val=1,
        do_stochastic_weight_averaging=args.stochastic_weight_average
    )

    ####################################################################
    # train model
    ####################################################################
    print("training model")
    trainer.fit(litmodel, train_dl, val_dl)
    print("DONE")


if __name__ == '__main__':
    # the code below set multiprocessing to use spawn in the main script (bottom) instead of fork (this allows augmentation on the gpu for the bias field)
    # I then need to set persistent_workers=True in the train dataloader otherwise it starts off new processes every batch and we don't want that 
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True) # see comment at the top about spawning processes
    # mp.set_sharing_strategy('file_system')
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
