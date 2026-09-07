print("importing")
import argparse
import SimpleITK as sitk
import torch
import pandas as pd
import os
import math
import time
import numpy as np

from twaibrain.brainpreprep.utils.VPrint import VPrint
from twaibrain.brainpreprep.utils.resample import sitk_resample
from twaibrain.brainpreprep.utils.image_io import save_manipulated_sitk_image_array
from twaibrain.braintorch.utils.entropy_map import entropy_map_from_samples_3d
from twaibrain.braintorch.data.inference_loading_and_formatting import load_and_format_flair_image_v2, reformat_model_prediction_v2
from twaibrain.braintorch.utils.reorder_samples import reorder_samples

from twaibrain.brainexperiments.segmentation_model_performance.model_inference.V4_model_eval.model_loading import load_model_ensemble
from twaibrain.brainexperiments.segmentation_model_performance.model_inference.V4_model_eval.model_predictions import ensemble_single_image_pred_standard, ensemble_single_image_pred_ssn

print("loaded imports")


SEED = 42
OUT_SPACING = [1., 1., 2.]
OUT_SHAPE = [80, 192, 160]

"""
this script just runs the model to produce the uq map, samples and segmentation outputs
it is a simplified script, it requires that the user resamples and resizes the images, hence, no options here.

"""

def construct_parser():
    parser = argparse.ArgumentParser(description = "WMH UQ / Fazekas model outputs")
    
    # inputs
    parser.add_argument('-i', type=str, help='path to csv file of input paths', required=True)
    parser.add_argument('--imgfolder', type=str, help='absolute path to image folder', default='/home/s2208943/preprocessed_data/')
    
    # output folder
    parser.add_argument('-o', type=str, help="output folder for outputs", required=True)
    
    # options
    parser.add_argument('-s', type='str', default='nearest', help="interpolation method used when resampling the predicted soft probabilities and uncertainty map back to the space of the input image. Must be one of nearest | linear | bspline")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--checkpoints_file', required=True, help="txt file giving comma separated list of paths to checkpoints")
    parser.add_argument('--model_size', default='large', help="size of nnUnet model")
    parser.add_argument('--ssn', action='store_true', help='include flag for SSN model')
    parser.add_argument('--ssn_num_samples', default=2)
    return parser
    
    
def process_subject(input_files, args, ensemble, vprint):
    
    # load input data
    vprint("loading image data")
    imageid, flair_path, mask_path = input_files

    # out paths
    if not os.path.exists(os.path.join(args.o, args.model_name)):
        os.makedirs(os.path.join(args.o, args.model_name), exist_ok=True)
    uq_out_path = os.path.join(args.o, args.model_name, flair_path.split(".nii")[0].split("/")[-1] + f"_{args.model_name}_uqimg.nii.gz")
    prob_out_path = os.path.join(args.o, args.model_name,  flair_path.split(".nii")[0].split("/")[-1] + f"_{args.model_name}_wmhprob.nii.gz")
    seg_out_path = os.path.join(args.o, args.model_name,  flair_path.split(".nii")[0].split("/")[-1] + f"_{args.model_name}_wmhseg.nii.gz")
    samples_out_path = os.path.join(args.o, args.model_name, flair_path.split(".nii")[0].split("/")[-1] + f"_{args.model_name}_samples.nii.gz")

    if os.path.exists(uq_out_path):
        return 

    img, metadata, orig_sitk_image = load_and_format_flair_image_v2(flair_path, mask_path, mask_pad=1, output_spacing=OUT_SPACING, output_shape=OUT_SHAPE, stack_mask=True, verbose=args.verbose, device=args.device)
    
    # run the model
    vprint("running segmentation model")
    if args.ssn:
        mean, samples = ensemble_single_image_pred_ssn(img, ensemble, args.device, args.ssn_num_samples)
    else:
        mean, samples = ensemble_single_image_pred_standard(img, ensemble, args.device)
    vprint(f"mean shape: {mean.shape} | samples shape: {samples.shape}")

    # generate umap, pred and samples
    vprint("generating umap and p_hat and samples")
    umap = entropy_map_from_samples_3d(samples, normalize=True) / -math.log(0.5)

    # print("WARNING CHECK 2d VS 3d REORER SAMPLES, IS THAT WHAT WE WANT")
    # samples = reorder_samples(samples) # what if we just skip reordering the samples?
    
    if mean.shape[0] == 1:
        p_hat = mean.sigmoid()[0]
        samples = samples.sigmoid()[:,0]
    else:
        p_hat = mean.softmax(dim=0)[1]
        samples = samples.softmax(dim=1)[:,1]

    vprint("reformatting output")
    vprint("reformatting p_hat")
    p_hat = reformat_model_prediction_v2(
        p_hat,
        **metadata,
        is_label= True if args.s == 'nearest' else False,
        interpolation= None if args.s == 'nearest' else args.s
    )
    vprint("reformatting umap")
    umap = reformat_model_prediction_v2(
        umap,
        **metadata,
        is_label= True if args.s == 'nearest' else False,
        interpolation= None if args.s == 'nearest' else args.s
    )
    vprint("reformatting samples")
    samples_seg = []
    for s in samples:
        samples_seg.append(
            reformat_model_prediction_v2(
                (s > 0.5) * 1,
                **metadata,
                is_label=True,
        ))
    samples_seg = np.stack(samples_seg)

    # binarizing seg
    seg = (p_hat > 0.5) * 1

    # our outputs are now: samples_seg, seg, p_hat, umap.
    vprint("saving outputs")
    save_manipulated_sitk_image_array(orig_sitk_image, umap, uq_out_path, new_dtype=np.float32, clip=True, float_to_int=False, useCompression=True)
    save_manipulated_sitk_image_array(orig_sitk_image, p_hat, prob_out_path, new_dtype=np.float32, clip=True, float_to_int=False, useCompression=True)
    save_manipulated_sitk_image_array(orig_sitk_image, seg, seg_out_path, new_dtype=np.uint8, clip=True, float_to_int=False, useCompression=True)
    samples_seg = np.moveaxis(samples_seg, 0, -1)
    save_manipulated_sitk_image_array(orig_sitk_image, samples_seg, samples_out_path, new_dtype=np.uint8, clip=True, float_to_int=False, useCompression=True)

    return p_hat, seg, samples_seg, umap
    
def main(args):

    # verbose?
    vprint = VPrint(args.verbose)
    
    # get paths to input files
    inputs = pd.read_csv(args.i, header=None)
    column_names = 'ID flair mask'.split(' ')
    inputs.columns = column_names
    N = inputs.shape[0]

    vprint("Warning, this script assumes FLAIR images have been z-score normalized already")
    args.device = "cuda" if args.gpu else "cpu"

    # load the model and model weights of the ensemble
    vprint("Loading segmentation model weights")
    with open(args.checkpoints_file, 'r') as f:
        checkpoints = f.readlines()[0].split(",")[:-1]
    print(f"ensemble size: {len(checkpoints)}")

    ensemble = load_model_ensemble(args.model_size, args.ssn, checkpoints, vprint, ssn_pre_head_layers=32, ssn_rank=25, device=args.device)
        
    # run subject loop
    for idx, row in inputs.iterrows():
        print(f"# processing image-set: {idx+1}/{N} : {row[0]}")
        start = time.time()
        try:
            process_subject(row, args, ensemble, vprint)
            
        except Exception as e:
            print(f"failed for image id: {row[0]}")
            print(e)
            # raise e
        end = time.time()
        print(f"time: {end - start}")
    

if __name__ == '__main__':
    torch.manual_seed(SEED)
    parser = construct_parser()
    args = parser.parse_args()
    main(args)