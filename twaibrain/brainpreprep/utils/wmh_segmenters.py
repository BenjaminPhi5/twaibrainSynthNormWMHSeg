"""

the methods below are for calling automated WMH segmentation tools.
All of these tools are slow, but SynthsegWMH is only around 2 minutes, 
whereas SAMSEG is > 20 mins and LST-AI > 10 mims (with skull stripping, > 5 mins without skull stripping)

some tools provide other information, e.g samseg and wmhsynthseg both provide anatomical segmentations as well
since I will run synthseg main version separately, since it is more accurate, I might want to remove all other classes
from the wmhsynthseg to save space.
"""

import shutil
import os
import SimpleITK as sitk
import subprocess

def run_LST_AI(t1_path, flair_path, outfile, is_skull_stripped=True):
    # see https://github.com/CompImg/LST-AI
    print("\n------------------\nrunning LST AI. Assuming that t1 and flair are in the same space")
    print("A number of errors may be printed. These are likely harmless")
    if is_skull_stripped:
        print("WARNING: your images must be skull-stripped or performance will degrade!")
    
    # LST-AI is only working on the cpu, not the gpu
    temp_out = outfile + '.temp'
    command = [
        'source ~/.bashrc && conda activate lstai && lst',
        '--t1', t1_path, '--flair', flair_path, '--output', temp_out,
        '--device', 'cpu', '--fast-mode'
    ]
    
    if is_skull_stripped:
        command.append('--stripped')
    
    print(" ".join(command))
    subprocess.run(" ".join(command), shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL)
    
    ## cleanup
    shutil.move(os.path.join(temp_out, 'space-flair_seg-lst.nii.gz'), outfile)
    shutil.rmtree(temp_out)
    

def run_MS_SAMSEG(in_images, lesion_mask_pattern, out_image, threads=12):
    # see https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg
    """
    DOCUMENTATION FROM FREESURFER:
    only considers voxels as candidate lesions if their (possibly multi-contrast) intensity satisfies certain constraints compared to the intensity of cortical gray matter. The pattern is a sequence of numbers, one number for each input contrast, with the following meaning: a value of "-1" or "1" indicates the voxels should be darker or brighter than cortical gray matter, respectively, whereas a value of "0" means that no intensity constraint is applied to the corresponding input contrast. 
    Typically "0" would be used for T1w or PD scans, and "1" for T2w/FLAIR scans."""
    
    assert out_image.endswith(".nii.gz")
    
    print("\n------------------\nrunning SAMSEG MS.")
    print("A number of errors may be printed. These are likely harmless")
    
    temp = out_image + ".temp"
    
    command = [
        'run_samseg',
        '--input'] + in_images + [
        '--pallidum-separate',
        '--lesion',
        '--lesion-mask-pattern'] + [str(r) for r in lesion_mask_pattern] + [
        '--output', temp,
        '--threads', str(threads),
        '--save-probabilities',
    ]
    
    print(' '.join(command))
    
    _ = subprocess.run(command, stdout=subprocess.DEVNULL)
    
    ## cleanup
    command = [
        'mri_convert', os.path.join(temp, 'seg.mgz'), os.path.join(temp, 'seg.nii.gz')
    ]
    _ = subprocess.run(command, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(temp, 'seg.nii.gz'), out_image)
    shutil.rmtree(temp)
    
    ## create binary image
    seg = sitk.ReadImage(out_image)
    wmh = sitk.Equal(seg, 99)
    wmh.CopyInformation(seg)
    sitk.WriteImage(wmh, out_image.split(".nii.gz")[0] + "_binary_wmh.nii.gz")


def run_WMHSynthSeg(in_image, out_image, device='cpu', threads=-1, csv_vols=False):
    print("\n------------------\nrunning SynthSeg WMH variant.")
    print("A number of errors may be printed. These are likely harmless")
    
    command = [
        'mri_WMHsynthseg',
        '--i', in_image,
        '--o', out_image,
        '--device', device,
        '--threads', str(threads),
        '--crop',
        '--save_lesion_probabilities'
    ]
    
    if csv_vols:
        csv_file = out_image_name + "_csv_vol.csv"
        command += ['--csv_vols', csv_file]
    
    print(' '.join(command))
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    # cleanup
    # put images back into native space
    command = [
        'mri_label2vol', '--seg', out_image,
        '--temp', in_image,
        '--o', out_image,
        '--regheader' 
    ]
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    probs_image = out_image.split(".nii.gz")[0] + ".lesion_probs.nii.gz"
    
    command = [
        'mri_vol2vol', 
        '--mov', probs_image,
        '--targ', in_image,
        '--regheader',
        '--interp', 'nearest',
        '--o', probs_image
    ]
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    # create binarized image
    seg = sitk.ReadImage(out_image)
    wmh = sitk.Equal(seg, 77)
    wmh.CopyInformation(seg)
    sitk.WriteImage(wmh, out_image)
