import SimpleITK as sitk
import shutil
import os
from pathlib import Path
import subprocess
from twaibrain.brainpreprep.utils.dicom2nifti import dicom_nifti_conversion

def save_manipulated_sitk_image_array(source_image, target_array, filepath, new_dtype=None, clip=False, float_to_int=False, useCompression=False):
    """Saves a manipulated nifti image array using the meta data information from a source image"""

    if clip:
        target_array[target_array < 0] = 0
        target_array[target_array > 1] = 1

    if float_to_int:
        target_array = target_array * 255

    if new_dtype is not None:
        target_array = target_array.astype(new_dtype)
    
    target_image = sitk.GetImageFromArray(target_array)
    target_image.SetSpacing(source_image.GetSpacing())
    target_image.SetOrigin(source_image.GetOrigin())
    target_image.SetDirection(source_image.GetDirection())
    target_image.CopyInformation(source_image)
    sitk.WriteImage(target_image, filepath, useCompression=useCompression)

def load_image(filepath):
    return sitk.GetArrayFromImage(sitk.ReadImage(filepath))

def fileending(filepath):
    ending = ".nii" if filepath.endswith(".nii") else ".nii.gz" if filepath.endswith(".nii.gz") else None
    if ending is None:
        raise ValueError("file ending must be .nii or .nii.gz")
    return ending

def copy_nifti_file(inp, out, copy_json_only=False):
    folder = "/".join(out.split("/")[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)
    # if they are both .nii or .nii.gz then just copy the file
    # if copy_json_only then just skips the stage for copying the image
    if not copy_json_only:
    
        if (inp.endswith(".nii") and out.endswith(".nii")) or (inp.endswith(".nii.gz") and out.endswith(".nii.gz")): 
            shutil.copy(inp, out)
        else: # use mri_convert to convert and copy the images
            command = [
                "mri_convert", inp, out
            ]
            _ = subprocess.run(command, stdout=subprocess.DEVNULL)

    else:
        print("copying json only")

    in_name = inp.split(".nii")[0]
    out_name = out.split(".nii")[0]
    
    # copy over json data if dcm2nii(x) has already been run on the provided image
    if os.path.exists(in_name + ".json"):
        shutil.copy(in_name + ".json", out_name+".json")

    if not copy_json_only:
        # copy any bval and bvec files for DWI/DTI data
        if os.path.exists(in_name + ".bval"):
            shutil.copy(in_name + ".bval", out_name+".bval")
        if os.path.exists(in_name + ".bvec"):
            shutil.copy(in_name + ".bvec", out_name+".bvec")
            
def copy_dcm_to_nifti(infile, outfile, dcm_key, force=False):
    splits = infile.split("/")
    if splits[-2] != dcm_key:
        raise ValueError(f"Couldn't match folder to dcm loading key: {dcm_key}")
    infolder = str(Path(infile).parent.absolute())
    print(infolder)
    print(outfile)
    dicom_nifti_conversion(infolder, outfile, gzip=True, rm_dcm=False, force=force)

def copy_analyze_to_nifti(analyze_file, nifti_file):
    assert (analyze_file.endswith(".hdr")) or (analyze_file.endswith(".img"))
    assert (nifti_file.endswith(".nii")) or (nifti_file.endswith(".nii.gz"))

    # print(analyze_file)
    # print(nifti_file)
    # print("...")

    folder = os.path.dirname(nifti_file)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # shutil.copy(analyze_file, folder)
    command = [
        "mri_convert",
        analyze_file,
        nifti_file
    ]
    _ = subprocess.call(command)#, stdout=subprocess.DEVNULL)
    
def copy_image_file(infile, outfile, dcm_key, copy_json_only=False, force_dcm_copy=False):
    if infile.endswith(".hdr") or infile.endswith(".img"):
        copy_analyze_to_nifti(infile, outfile)
    elif ".nii" in infile or dcm_key is None:
        copy_nifti_file(infile, outfile, copy_json_only)
    else:
        copy_dcm_to_nifti(infile, outfile, dcm_key, force=force_dcm_copy)

def convert_mgz_to_niigz(mgz_file):
    assert mgz_file.endswith(".mgz")
    niigz_file = mgz_file.split(".mgz")[0] + ".nii.gz"
    
    command = [
        "mri_convert",
        mgz_file,
        niigz_file
    ]
    
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    return niigz_file
    
def convert_nii_to_mgz(nii_file):
    assert ".nii" in nii_file
    mgz_file = nii_file.split(".nii")[0] + ".mgz"
    
    command = [
        "mri_convert",
        nii_file,
        mgz_file
    ]
    
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    return mgz_file
    
def freesurfer_label_to_orig_space(label_img, regheader=None):
    splits = label_img.split("/")
    label_root = "/".join(splits[:-1])
    filename = splits[-1]
    rawavg = os.path.join(label_root, "rawavg.mgz")
    out_filename = "origspace_" + filename
    out_path = os.path.join(label_root, out_filename)
    
    command = [
        "mri_label2vol",
        "--seg", label_img,
        "--temp", rawavg,
        "--o", out_path,
        "--regheader", label_img if regheader is None else regheader,
    ]
    
    _ = subprocess.call(command)
    
    return out_path
    
def freesurfer_vol_to_orig_space(vol_img):
    splits = vol_img.split("/")
    vol_root = "/".join(splits[:-1])
    filename = splits[-1]
    rawavg = os.path.join(vol_root, "rawavg.mgz")
    out_filename = "origspace_" + filename
    out_path = os.path.join(vol_root, out_filename)
    
    command = [
        "mri_vol2vol",
        "--mov", vol_img,
        "--targ", rawavg,
        "--o", out_path,
        "--regheader",
        "--no-save-reg",
    ]
    
    print(" ".join(command))
    
    _ = subprocess.call(command)
    
    return out_path