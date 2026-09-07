import subprocess
import SimpleITK as sitk
from typing import Tuple
import numpy as np

def resample(image_path, out_path, voxel_sizes, use_nearest_neighbor=False):
    rt = "interpolate"
    if use_nearest_neighbor:
        rt = "nearest"
    
    voxel_sizes = [str(v) for v in voxel_sizes]
    command = [
        "mri_convert", "-vs", *voxel_sizes, "-rt", rt,
        image_path, out_path
    ]
    
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    
    return out_path

def resample_match_if_necessary(fixed, moving, use_nearest_neighbor=False):
    """
    resample moving to fixed if moving and fixed are not in the same space....
    """
    fixed_spacing = sitk.ReadImage(fixed).GetSpacing()
    moving_spacing = sitk.ReadImage(moving).GetSpacing()

    if (abs(fixed_spacing[0] - moving_spacing[0]) <= 0.05) and (abs(fixed_spacing[1] - moving_spacing[1]) <= 0.05) and (abs(fixed_spacing[2] - moving_spacing[2]) <= 0.05):
        return
    
    interp = "trilinear"
    if use_nearest_neighbor:
        interp = "nearest"

    command = [
        "mri_vol2vol", "--mov", moving, "--targ", fixed, "--o", moving, "--regheader", "--interp", interp
    ]
    
    _ = subprocess.call(command, stdout=subprocess.DEVNULL)


def sitk_resample(img: sitk.Image, out_spacing: Tuple[float, float, float], is_label=False, verbose=False, interpolation='bspline') -> sitk.Image:
    """Resample using SimpleITK's ResampleImageFilter."""
    in_spacing = img.GetSpacing()  # (sx, sy, sz)
    in_size = img.GetSize()        # (nx, ny, nz)


    if is_label:
        interpolator = sitk.sitkNearestNeighbor
    else:
        if interpolation == 'linear':
            interpolator = sitk.sitkLinear  
        elif interpolation == 'bspline':
            interpolator = sitk.sitkBSpline
        else:
            raise ValueError(f"interpolation {interpolation} unknown, should be one of : ( linear | bspline )")

    if verbose:
        print("interpolator: ", interpolation, " | is_label: ", is_label)
    
    # Compute new size preserving physical size
    new_size = [
        int(round(in_size[i] * (in_spacing[i] / out_spacing[i])))
        for i in range(3)
    ]

    if verbose:
        print(f"img spacing {in_spacing} | img size: {in_size}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(interpolator)
    # # Use a reasonable default background value (median of image) to avoid bright rims on B-spline
    # arr = sitk.GetArrayViewFromImage(img).astype(np.float32)
    default_val = 0 #float(np.median(arr))
    resampler.SetDefaultPixelValue(default_val)
    out = resampler.Execute(img)

    if verbose:
        print(f"output shape {out.GetSize()}")
    
    return out

    