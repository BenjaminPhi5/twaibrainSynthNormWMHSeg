import ants
from twaibrain.brainpreprep.utils.orientation import orient as orient_func
from twaibrain.brainpreprep.utils.resample import resample
import os
import subprocess

###################################################################################################################
# ANTS registration tools
###################################################################################################################
def run_ants_SyNAggro(fixed, moving, out, outsuffix, label=False, mask=None, moving_mask=None):
    """
    runs the ANTS affine orientation
    fixed, moving, out are the filepaths of the fixed, moving, and desired output location respectively.
    orient: whether to orient the images to standard orientation or not.
    out: the out path to save the registration transform, and optionally the registered image.
    """
    
    print(f"ANTS registering {moving} to {fixed}")
    fixed = ants.image_read(fixed)
    moving = ants.image_read(moving)

    if not label:
        fixed = ants.iMath(fixed, 'Normalize')
        moving = ants.iMath(moving, 'Normalize')
        
        # fixed[fixed==0] = 1e-6
        # moving[moving==0] = 1e-6)
    
    result = ants.registration(
        fixed,
        moving,
        type_of_transform='SyNAggro', #, 'SyNRA',#'SyNRA', #'Affine', #SyNRA #SyNAggro 
        # initial_transform=initial_transform,
        outprefix=out.split(".nii")[0] + "_" + outsuffix + "_",
        grad_step=0.2,
        flow_sigma=3,
        total_sigma=0,
        aff_metric='mattes',
        aff_sampling=64,
        aff_random_sampling_rate=0.3,
        syn_metric='mattes',
        syn_sampling=64,
        reg_iterations=(80, 40, 10),
        aff_iterations=(2100, 1200, 1200, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        write_composite_transform=False,
        random_seed=None,
        verbose=False,
        multivariate_extras=None,
        restrict_transformation=None,
        smoothing_in_mm=False,
        mask=ants.image_read(mask) if mask is not None else None,
        moving_mask=ants.image_read(moving_mask) if moving_mask is not None else None,
    )


def run_ants(fixed, moving, out, orient=False, save=True, normalize=False, rigid=False):
    """
    runs the ANTS affine orientation
    fixed, moving, out are the filepaths of the fixed, moving, and desired output location respectively.
    orient: whether to orient the images to standard orientation or not.
    out: the out path to save the registration transform, and optionally the registered image.
    """
    if orient:
        print("orienting images....")
        fixed = orient_func(fixed)
        moving = orient_func(moving)
    
    print(f"ANTS registering {moving} to {fixed}")
    fixed = ants.image_read(fixed)
    moving = ants.image_read(moving)

    if normalize:
        fixed = ants.iMath(fixed, 'Normalize')
        moving = ants.iMath(moving, 'Normalize')
    
    result = ants.registration(
        fixed,
        moving,
        type_of_transform='Affine' if not rigid else 'Rigid',
        outprefix=out.split(".nii")[0] + "_",
        grad_step=0.2,
        flow_sigma=3,
        total_sigma=0,
        aff_metric='mattes',
        aff_sampling=64,
        aff_random_sampling_rate=0.3,
        syn_metric='mattes',
        syn_sampling=64,
        reg_iterations=(80, 40, 10),
        aff_iterations=(2100, 1200, 1200, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        write_composite_transform=False,
        random_seed=None,
        verbose=False,
        multivariate_extras=None,
        restrict_transformation=None,
        smoothing_in_mm=False,
    )
    
    if save:
        ants.image_write(result['warpedmovout'], out)

def run_ants_standard(fixed, moving, out, save=True):
    """
    runs the ants orientation with default parameters.
    """
    
    print(f"ANTS registering {moving} to {fixed}")
    fixed = ants.image_read(fixed)
    moving = ants.image_read(moving)
    
    result = ants.registration(
        fixed,
        moving,
        type_of_transform='Affine', #'Affine', #SyNRA #SyNAggro 
        outprefix=out.split(".nii")[0] + "_",
        grad_step=0.2,
        flow_sigma=3,
        total_sigma=0,
        aff_metric='mattes',
        aff_sampling=32,
        aff_random_sampling_rate=0.2,
        syn_metric='mattes',
        syn_sampling=32,
        reg_iterations=(40, 20, 0),
        aff_iterations=(2100, 1200, 1200, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        write_composite_transform=False,
        random_seed=None,
        verbose=False,
        multivariate_extras=None,
        restrict_transformation=None,
        smoothing_in_mm=False,
    )
    
    if save:
        ants.image_write(result['warpedmovout'], out)
    
    
def apply_ants_transforms(fixed, moving, out, transforms_list, is_label=False, write=True, whichtoinvert=None, multiimage=False):
    """
    fixed : path to a ANTsImage
        fixed image defining domain into which the moving image is transformed.

    moving : path to a AntsImage
        moving image to be mapped to fixed space.
    
    out : path where the transformed image will be saved
    transforms_list : list of filepaths to ANTS transforms to be applied, in the order of application, i.e [transform1.mat, transform2.mat, ..., transformN.mat].
        To apply a single transform, transforms_list should be of the form: [transform.mat] i.e a single element list
        
    is_label : whether the image is a integer/binary label image, or a brain image

    whichtoinvert: a list of booleans, stating whether each transform should be inverted or not. only works for matrix transforms. for nonlinear, need to pass the invWarp transform to transforms_list. 
    
    multiimage: set to True if transforming a 4D image.
    """
    if not isinstance(transforms_list, list):
        raise ValueError(f"transforms_list must be a list")
        
    if len(transforms_list) == 0:
        raise ValueError("no transforms to apply")
        
    fixed  = ants.image_read(fixed)
    moving = ants.image_read(moving)
    
    transformed_image = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        imagetype=0 if not multiimage else 3,
        transformlist=transforms_list[::-1],
        interpolator=("genericLabel" if is_label else "linear"),
        whichtoinvert=whichtoinvert[::-1] if whichtoinvert is not None else None,
    )
    
    if write:
        ants.image_write(transformed_image, out)
    else:
        return transformed_image

##############################################################################
# FREESURFER LONGITUDINAL BRAIN TEMPLATE
# kept for reference in future
##############################################################################
def create_longitudinal_template(images, out, overwrite_resample=False, orient=False):
    """
    runs the freesurfer mri_robust_template tool
    """
    if orient:
        oriented_images = []
        print("orienting images....")
        for img in images:
            oriented_images.append(orient_func(img))
        images = oriented_images
        
    # first resample all images to median voxel space
    voxel_sizes = []
    for img in images:
        voxel_sizes.append(get_voxel_size(img))
    voxel_sizes = np.array(voxel_sizes)
    target_size = np.median(voxel_sizes, axis=0)
    resampled_imgs = []
    outfolder = "/".join(out.split("/")[:-1])
    if not overwrite_resample:
        resampled_img = os.path.join(outfolder, img.split("/")[-1].split(".nii")[0] + "_resampled.nii.gz")
    else:
        resampled_img = img
    for img in images:
        resampled_img = resample(img,  resampled_img, target_size)
        resampled_imgs.append(resampled_img)
    
    # run freesurfer on the longitudinal images.
    print(f"freesurfer longitudinal registering {images}")
    command = [
        "mri_robust_template",
        "--mov"] + resampled_imgs + [
        "--template", out,
        "--satit", "--floattype", "--epsit", "--affine"
    ]
    
    print("\n\n----")
    print(" ".join(command))
    _ = subprocess.run(command)

    
##############################################################################
# OTHER REGISTRAITION CODE
# kept for reference in future
##############################################################################
from twaibrain.brainpreprep.utils.fsldir import fsldir

def run_flirt(fixed, moving, omat, out, dof=12, cost='mutualinfo'):
    command = [
        os.path.join(fsldir(), 'bin', 'flirt'),
        f'-in {moving}',
        f'-ref {fixed}',
        f'-dof {dof}',
        f'-cost {cost}',
        f'-omat {omat}',
        f'-out {out}',
    ]
    print(' '.join(command))
    os.system(" ".join(command))
    
def flirt_rigid_wrapper1(fixed, moving, out):
    out = out + "flirt_rigid_mi"
    omat = out + "_MAT"
    run_flirt(fixed, moving, omat, out, dof=6, cost='mutualinfo')
    return out + ".nii.gz"
        
def flirt_rigid_wrapper2(fixed, moving, out):
    out = out + "flirt_rigid_normmi_"
    omat = out + "_MAT"
    run_flirt(fixed, moving, omat, out, dof=6, cost='normmi')
    return out + ".nii.gz"    
        
def flirt_affine_wrapper1(fixed, moving, out):
    out = out + "flirt_affine_mi_"
    omat = out + "_MAT"
    run_flirt(fixed, moving, omat, out, dof=12, cost='mutualinfo')
    return out + ".nii.gz"
        
def flirt_affine_wrapper2(fixed, moving, out):
    out = out + "flirt_affine_normi_"
    omat = out + "_MAT"
    run_flirt(fixed, moving, omat, out, dof=12, cost='normmi')
    return out + ".nii.gz"


def run_easyreg(fixed, moving, out):
    print(f"EasyReg Affine registering {moving} to {fixed}")
    
    command = [
        'mri_easyreg',
        '--ref', fixed,
        '--flo', moving,
        '--ref_seg', 'fixed_seg.nii.gz', '--flo_seg', 'moving_seg.nii.gz',
        '--flo_reg', out,
        '--threads', '-1',
        '--fwd_field', 'forward_field.nii.gz', '--bak_field', 'backward_field.nii.gz',
        '--affine_only'
    ]
    
    print(' '.join(command))
    # _ = subprocess.call(command, stdout=subprocess.DEVNULL)
    _ = subprocess.call(command)
    