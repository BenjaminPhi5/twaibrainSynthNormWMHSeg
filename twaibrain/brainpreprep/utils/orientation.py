import subprocess
import nibabel as nib
import numpy as np
import os

### FREESURFER mri_convert based orient. THIS DOES INTERPOLATION. I wish I knew this when I started.... really quite a pain to be dealing with but there we are.

def orient(image:str, in_orientation:str=None, orientation_if_broken:str='PSL', add_oriented_extension=False, use_image_axcode=False):
    """
    function for orienting a single image. the oriented image is stored in the name folder with '_oriented' appended to the filename (before the .nii(.gz))

    image: path to a nifti image
    in_orientation: (optional) a hard coded input orientation. metadata from the file is ignored in this case, and the image is assumed to be oriented according to in_orientation
    orientation_if_broken: (optional) if in_orientation is None, then the orientation is determined from the metadata of the image. In the event that the metadata is invalid, then
                            the orientation of the image will be assumed to be orientation_if_broken. If, in this case, orientation_if_broken is None, an error will be raised.
                            
    add_oriented_extension: if True, adds _oriented to the end of the filename
    
    returns: the filepath of the oriented image that has been created.
    
    ValueError is raised if the image cannot be oriented.
    """
    
    ending = ".nii" if image.endswith(".nii") else ".nii.gz" if image.endswith(".nii.gz") else None
    if ending is None:
        raise ValueError("file ending must be .nii or .nii.gz")
    
    if add_oriented_extension:
        new_image = image.split(ending)[0] + "_oriented" + ending
    else:
        new_image = image
    command = ['mri_convert']

    if use_image_axcode: # use the orientation ax code in the metadata to orient the image. (e.g if it says LAS in the metadata, assume it is true LAS and ignore the rotation matrix.
        in_orientation = get_image_axcode(image)
        
    if in_orientation is not None:
        command.extend(['--in_orientation', in_orientation])
    
    command.extend(['--out_orientation', 'RAS', image, new_image]) # 'RAS'
    
    result = subprocess.run(command, stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE, text=True)
    
    # check if the orientation failed. if it did, guess that the image is in PSR
    if result.returncode == 1:
        if orientation_if_broken is not None and "WARNING: neither NIfTI-1 qform or sform are valid" in result.stderr:
            orient(image, in_orientation=orientation_if_broken, orientation_if_broken=None, add_oriented_extension=add_oriented_extension)
        else:
            raise ValueError(f"orientation failed for file: {image} with error {result.stderr}")
    
    return new_image



def get_image_axcode(img):
    img = nib.load(img)
    affine = img.affine
    orig_ornt = nib.orientations.aff2axcodes(affine)
    return ''.join(orig_ornt)

    

def get_swapdim_args(affine):
    """
    Determines the fslswapdim arguments needed to convert to RAS orientation.
    """
    # Get orientation codes from affine: e.g., ('L', 'P', 'S')
    orig_ornt = nib.orientations.aff2axcodes(affine)
    desired_ornt = ('R', 'A', 'S')

    # Map from axis codes to fslswapdim options
    axis_map = {
        'R': 'x', 'L': '-x',
        'A': 'y', 'P': '-y',
        'S': 'z', 'I': '-z'
    }

    swapdim_args = []
    for code in desired_ornt:
        # Find which input axis matches this desired direction
        for i, orig_code in enumerate(orig_ornt):
            if axis_map[orig_code] == axis_map[code]:
                swapdim_args.append(axis_map[orig_code])
                break
            elif axis_map[orig_code] == '-' + axis_map[code]:
                swapdim_args.append('-' + axis_map[orig_code].lstrip('-'))
                break
    return swapdim_args

def reorient_label_to_ras(input_path, output_path):
    """
    Reorients a label image to RAS using fslswapdim without interpolation.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    # Load affine to detect current orientation
    img = nib.load(input_path)
    affine = img.affine
    swapdim_args = get_swapdim_args(affine)

    if len(swapdim_args) != 3:
        raise RuntimeError("Failed to determine correct fslswapdim arguments.")

    # Create temporary output
    temp_path = output_path.replace('.nii.gz', '_temp.nii.gz')
    
    # Run fslswapdim to reorient
    subprocess.run(['fslswapdim', input_path] + swapdim_args + [temp_path], check=True)
    subprocess.run(['fslorient', '-copysform2qform', temp_path], check=True)

    # Move or compress to final output
    if output_path.endswith('.nii.gz'):
        subprocess.run(['fslchfiletype', 'NIFTI_GZ', temp_path, output_path], check=True)
        os.remove(temp_path)
    else:
        os.rename(temp_path, output_path)

    print(f"Reoriented label saved to: {output_path}")