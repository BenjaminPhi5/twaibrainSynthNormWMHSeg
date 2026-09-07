"""
scripts for calling the SynthSeg and SynthStrip functions.
Currently relies on setting up a separate conda environment for SynthSeg.
In future I should add a script that detects the location of the SynthSeg directory, it should look from the ~/.bashrc etc.
"""
import os
import numpy as np
import subprocess
import SimpleITK as sitk
import tempfile
import shutil
from scipy.ndimage import distance_transform_edt
from twaibrain.brainpreprep.utils.image_io import save_manipulated_sitk_image_array, fileending
from twaibrain.brainpreprep.utils.file_io import create_symlinks
from twaibrain.brainpreprep.utils.resample import resample_match_if_necessary

_sspath_a = "/home/s2208943/miniconda3/envs/synthseg_38/bin/python"
_sspath_b = "/opt/conda/envs/synthseg_38/bin/python"

_sspredict_a = "/home/s2208943/tools/SynthSeg/scripts/commands/SynthSeg_predict.py"
_sspredict_b = "/workspace/projectlib/tools/SynthSeg/scripts/commands/SynthSeg_predict.py"

SYNTH_SEG_PYTHON_PATH = _sspath_a if os.path.exists(_sspath_a) else (_sspath_b if os.path.exists(_sspath_b) else None)
SYNTH_SEG_PREDICT_PATH =  _sspredict_a if os.path.exists(_sspredict_a) else (_sspredict_b if os.path.exists(_sspredict_b) else None)

VENTRICLE_1 = 4
VENTRICLE_2 = 43
VENTRICLE_INFERIOR_L = 5
VENTRICLE_INFERIOR_R = 44

CORTEX_1 = 3
CORTEX_2 = 42
CORTEX_3 = 8
CORTEX_4 = 47
CORTEX_PARC = 1000


def run_synthseg_multiple_files_fixed_dir(tmpdirname, file_paths, output_folders, threads=12, parc=True, fast=False, symbolic=True, cpu=False):
    """
    wrapper function on run_synthseg_folder that allows us to run synthseg on a collection of folders by copying the relevant files to a temp folder first
    
    tmpdirname: a provided temporary folder to store the output
    
    file_paths: list of filepaths that we want to run synthseg on
    output_paths: matching output paths for each file where each synthseg output can be stored
    so filepath i should be stored in output folder i
    
    use gpu by default if the gpu is available.
    
    symbolic: whether the temp folder should be comprised of symbolic links, or whether the data should be copied
    
    prefix: the prefix to the temporary folder used to store output data
    
    """
    os.makedirs(tmpdirname, exist_ok=True) 

    if symbolic:
        create_symlinks(file_paths, tmpdirname)

    else:
        for fp in file_paths:
            shutil.copy(fp, tmpdirname)

    run_synthseg_folder(tmpdirname, tmpdirname, threads=threads, parc=parc, fast=fast, cpu=cpu)

    filenames_in = [f.split("/")[-1].split(".nii")[0] for f in file_paths]
    filenames_out = [f for f in os.listdir(tmpdirname)]

    print("filenames out computed: ", filenames_out)

    files_out = []
    for i, fni in enumerate(filenames_in):
        out_file = os.path.join(tmpdirname, fni + "_synthseg.nii.gz")
        out_folder = output_folders[i]
        if os.path.exists(out_file):
            os.makedirs(out_folder, exist_ok=True)
            shutil.copy(out_file, out_folder)
            files_out.append(os.path.join(out_folder, fni + "_synthseg.nii.gz"))
        else:
            print("couldn't find: ", out_file)
    return files_out

def run_synthseg_multiple_files(file_paths, output_folders, threads=12, parc=True, fast=False, symbolic=True, cpu=False, prefix="/home/s2208943/"):
    """
    wrapper function on run_synthseg_folder that allows us to run synthseg on a collection of folders by copying the relevant files to a temp folder first
    
    file_paths: list of filepaths that we want to run synthseg on
    output_paths: matching output paths for each file where each synthseg output can be stored
    so filepath i should be stored in output folder i
    
    use gpu by default if the gpu is available.
    
    symbolic: whether the temp folder should be comprised of symbolic links, or whether the data should be copied
    
    prefix: the prefix to the temporary folder used to store output data
    
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdirname:
        print('created temporary directory for synthseg running', tmpdirname)
        
        if symbolic:
            create_symlinks(file_paths, tmpdirname)
                
        else:
            for fp in file_paths:
                shutil.copy(fp, tmpdirname)
            
        run_synthseg_folder(tmpdirname, tmpdirname, threads=threads, parc=parc, fast=fast, cpu=cpu)
        
        filenames_in = [f.split("/")[-1].split(".nii")[0] for f in file_paths]
        filenames_out = [f for f in os.listdir(tmpdirname)]
        
        print("filenames out computed: ", filenames_out)
        
        files_out = []
        for i, fni in enumerate(filenames_in):
            out_file = os.path.join(tmpdirname, fni + "_synthseg.nii.gz")
            out_folder = output_folders[i]
            if os.path.exists(out_file):
                os.makedirs(out_folder, exist_ok=True)
                shutil.copy(out_file, out_folder)
                files_out.append(os.path.join(out_folder, fni + "_synthseg.nii.gz"))
            else:
                print("couldn't find: ", out_file)
    return files_out

def run_synthseg_folder(in_folder, out_folder, threads=12, parc=True, fast=False, cpu=False):
    """
    runs synthseg on all images in in_folder and saves the results to out_folder
    use gpu by default if the gpu is available.
    """
    
    subprocess.run([
        SYNTH_SEG_PYTHON_PATH,
        SYNTH_SEG_PREDICT_PATH,
        "--i", in_folder,
        "--o", out_folder, 
        "--threads", str(threads)] + (["--parc"] if parc else []) + (["--fast"] if fast else []) + (["--cpu"] if cpu else []),
        #stdout=subprocess.DEVNULL
    )
            

def run_synthseg_folder_with_distance_maps(in_folder, out_folder, threads=12, parc=True, fast=False):
    """
    runs synthseg on all images in in_folder and saves the results to out_folder
    use gpu by default if the gpu is available.
    """
    
    subprocess.run([
        SYNTH_SEG_PYTHON_PATH,
        SYNTH_SEG_PREDICT_PATH,
        "--i", in_folder,
        "--o", out_folder, 
        "--threads", str(threads)] + (["--parc"] if parc else []) + (["--fast"] if fast else []),
        #stdout=subprocess.DEVNULL
    )
    
    out_images = []
    
    for file in os.listdir(in_folder):
        # synthseg and ventdist clauses are so we don't operate on any files already created by this function at an earlier time.
        if (file.endswith(".nii") or file.endswith(".nii.gz")) and ("synthseg" not in file) and ("ventdist" not in file):
            in_imagename = file.split(".nii")[0]
            in_image = os.path.join(in_folder, file)
            in_filetype = fileending(file)
            synthseg_outimage = os.path.join(out_folder, in_imagename + "_synthseg" + in_filetype)
            ventmap_outimage = os.path.join(out_folder, in_imagename + "_ventdist" + in_filetype)
            cortexmap_outimage = os.path.join(out_folder, in_imagename + "_cortexdist" + in_filetype)
            
            # copy the synthseg image to 1x1x1 space
            command = [
                "mri_convert", "-vs", "1", "1", "1", "-rt", "nearest",
                synthseg_outimage, ventmap_outimage
            ]
            _ = subprocess.call(command, stdout=subprocess.DEVNULL)
            
            # create ventricle distance map imag
            create_ventricle_distance_map(ventmap_outimage, ventmap_outimage, cortexmap_outimage)
            
            # put ventricle dist, cortex dist and synthseg image back into the right space
            command = [
                'mri_vol2vol', '--mov', ventmap_outimage,
                '--targ', in_image,
                '--o', ventmap_outimage,
                '--regheader' 
            ]
            _ = subprocess.call(command, stdout=subprocess.DEVNULL)
            
            command = [
                'mri_vol2vol', '--mov', cortexmap_outimage,
                '--targ', in_image,
                '--o', cortexmap_outimage,
                '--regheader' 
            ]
            _ = subprocess.call(command, stdout=subprocess.DEVNULL)
            
            command = [
                'mri_label2vol', '--seg', synthseg_outimage,
                '--temp', in_image,
                '--o', synthseg_outimage,
                '--regheader' 
            ]
            _ = subprocess.call(command, stdout=subprocess.DEVNULL)
            
            out_images.append((file, synthseg_outimage, ventmap_outimage, cortexmap_outimage))
            
    return out_images

def run_synthseg_file(in_image, out_folder, threads=12, parc=True, gpu=False):
    # do not delete, useful for debugging and fixing individual files even if no longer in main preprep pipeline.
    """
    runs synth seg on in_image and saves the result to out_image
    use gpu by default if the gpu is available.
    """
    
    in_imagename = in_image.split(".nii")[0].split("/")[-1]
    in_filetype = fileending(in_image)
    
    subprocess.run([
        SYNTH_SEG_PYTHON_PATH,
        SYNTH_SEG_PREDICT_PATH,
        "--i", in_image,
        "--o", out_folder, 
        "--threads", str(threads)]
         + (["--parc"] if parc else [])
         + (["--cpu"] if not gpu else [])
    )
    
    synthseg_outimage = os.path.join(out_folder, in_imagename + "_synthseg" + in_filetype)
    ventmap_outimage = os.path.join(out_folder, in_imagename + "_ventdist" + in_filetype)
    cortexmap_outimage = os.path.join(out_folder, in_imagename + "_cortexdist" + in_filetype)
        
    # copy synthseg output into 1x1x1 space
    command = [
        "mri_convert", "-vs", "1", "1", "1", "-rt", "nearest",
        synthseg_outimage, ventmap_outimage
    ]
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)

    # create ventricle distance map image
    create_ventricle_distance_map(ventmap_outimage, ventmap_outimage, cortexmap_outimage)

    # put ventricle dist and cortex dist image back into the native space
    command = [
        'mri_vol2vol', '--mov', ventmap_outimage,
        '--targ', in_image,
        '--o', ventmap_outimage,
        '--regheader' 
    ]
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    command = [
        'mri_vol2vol', '--mov', cortexmap_outimage,
        '--targ', in_image,
        '--o', cortexmap_outimage,
        '--regheader' 
    ]
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    # cleanup
    # put synthseg image back into native space
    command = [
        'mri_label2vol', '--seg', synthseg_outimage,
        '--temp', in_image,
        '--o', synthseg_outimage,
        '--regheader' 
    ]
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    return synthseg_outimage, ventmap_outimage, cortexmap_outimage
        
        
def create_ventricle_distance_map(synthseg_file, outfile_ventricle, outfile_cortex):
    """
    loads the synth_seg segmentation, extracts the ventricles segmentation and the cortex segmentation
    and creates a euclidian distance map from each voxel to the ventricles.
    This distance map is then saved under the name out_file
    """
    
    synthseg_img = sitk.ReadImage(synthseg_file)
    
    spacing = synthseg_img.GetSpacing()
    if not ((0.95 <= spacing[0] <= 1.05) and (0.95 <= spacing[1] <= 1.05) and (0.95 <= spacing[2] <= 1.05)):
        raise ValueError(f"image spacing must be approx (1, 1, 1) to compute distance map, not {spacing}")
        
    def extract_distance(condition, outfile):
        condition = condition.astype(np.float32)
        distance_map = distance_transform_edt(1 - condition)
        save_manipulated_sitk_image_array(synthseg_img, distance_map, outfile)
    
    synthseg = sitk.GetArrayFromImage(synthseg_img)
    vent_dist = extract_distance((synthseg == VENTRICLE_1) | (synthseg == VENTRICLE_2) | (synthseg == VENTRICLE_INFERIOR_L) | (synthseg == VENTRICLE_INFERIOR_R), outfile_ventricle)
    extract_distance((synthseg == CORTEX_1) | (synthseg == CORTEX_2) | (synthseg == CORTEX_3) | (synthseg == CORTEX_4) | (synthseg > CORTEX_PARC), outfile_cortex)


def postprocess_synthseg(in_image, out_folder):
    """
    takes the synthseg output, creates the ventricle distance and cortex distance map
    and then resamples all three images to the space of the original input image synthseg was run on.

    in_image: the image that synthseg was run on
    out_folder: the path to the derivatives folder where the synthseg imgage is stored and the distance maps will be created.
    """
    in_imagename = in_image.split(".nii")[0].split("/")[-1]
    in_filetype = fileending(in_image)
    synthseg_outimage = os.path.join(out_folder, in_imagename + "_synthseg" + in_filetype)
    ventmap_outimage = os.path.join(out_folder, in_imagename + "_ventdist" + in_filetype)
    cortexmap_outimage = os.path.join(out_folder, in_imagename + "_cortexdist" + in_filetype)

    # ensure the synthseg image is in 1x1x1 space
    command = [
        "mri_convert", "-vs", "1", "1", "1", "-rt", "nearest",
        synthseg_outimage, ventmap_outimage
    ]
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    # create ventricle distance map imag
    create_ventricle_distance_map(ventmap_outimage, ventmap_outimage, cortexmap_outimage)

    # resample all of the output images back to the space of the in_image
    resample_match_if_necessary(in_image, synthseg_outimage, use_nearest_neighbor=True)
    resample_match_if_necessary(in_image, ventmap_outimage, use_nearest_neighbor=False)
    resample_match_if_necessary(in_image, cortexmap_outimage, use_nearest_neighbor=False)

        
def run_synthstrip_file(in_image, out_folder, threads=12, save_stripped_brain=False, use_gpu=False, skip_if_exist=True, b=None):
    """
    runs synth strip on in_image and saves the resulting mask in out_folder
    if save_stripped_brain is true, then saves the stripped input image as well.

    can optionally give the -b flag to increase the bondary distance from the brain (maybe useful for DTI images...)
    """

    in_imagename = in_image.split(".nii")[0].split("/")[-1]
    in_filetype = fileending(in_image)
    
    os.makedirs(out_folder, exist_ok=True)
    
    synthstrip_maskimage = os.path.join(out_folder, in_imagename + "_synthstripmask" + in_filetype)
    synthstrip_strippedimage = os.path.join(out_folder, in_imagename + "_stripped" + in_filetype)
    
    if skip_if_exist and os.path.exists(synthstrip_maskimage):
        return
    
    command = [
        "mri_synthstrip",
        "-i", in_image] + ([
        "-o", synthstrip_strippedimage] if save_stripped_brain else []) + [
        "-m", synthstrip_maskimage,
        "--threads", str(threads)
    ] + (["-g"] if use_gpu else []) + (["-b", str(b)] if b is not None else [])
    
    print(command)
    subprocess.run(command, stdout=subprocess.DEVNULL)
    
    
def create_combined_mask(file):
    """
    computes union of synthseg > 0 and synthstrip.
    returns path of mask file.
    """
    ending = fileending(file)
    
    filename = file.split(".nii")[0]
    synthseg_file = filename + "_synthseg" + ending
    synthstrip_file = filename + "_synthstripmask" + ending
    
    mask_file = filename + "_brainmask" + ending
    
    synthseg_img = sitk.ReadImage(synthseg_file)
    synthstrip_img = sitk.ReadImage(synthstrip_file)

    synthseg_arr = sitk.GetArrayFromImage(synthseg_img)
    synthstrip_arr = sitk.GetArrayFromImage(synthstrip_img)
    
    if synthseg_arr.sum() == 0:
        raise ValueError("Synthseg failed to produce a valid output")
    if synthstrip_arr.sum() == 0:
        raise ValueError("SynthStrip failed to produce a valid output")
        
    mask = (synthseg_arr > 0).astype(np.int16) | synthstrip_arr.astype(np.int16)
    
    save_manipulated_sitk_image_array(synthstrip_img, mask, mask_file)
    
    return mask_file

def get_maskfile_path(file):
    """
    finds the mask file for any given file (regardless of whether its a t1, or a flair, or a dwi, or a derivative file etc"
    """
    ending = fileending(file)
    splits = file.split("/")
    splits[-2] = "derivatives" 
    folder = "/".join(splits[:-1])
    deriv_files = os.listdir(folder)
    for f in deriv_files:
        if f.endswith(f"_brainmask{ending}"):
            return os.path.join(folder, f)
    
    raise ValueError(f"brainmask file for {file} not found")
    
def apply_mask(file, outfile, mask_file=None):
    
    if mask_file is None:
        ending = fileending(file)
        mask_file = file.split(".nii")[0] + "_brainmask" + ending
    
    img = sitk.ReadImage(file)
    mask_img = sitk.ReadImage(mask_file)
    
    arr = sitk.GetArrayFromImage(img)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    
    arr *= (mask_arr == 1)
    
    save_manipulated_sitk_image_array(img, arr, outfile)
