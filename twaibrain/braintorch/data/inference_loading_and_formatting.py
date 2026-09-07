"""
This set of functions is for loading and formatting images at inference time, and then saving the results back to disk
"""

from twaibrain.braintorch.utils.load_image import torch_load_and_resample, sitk_to_torch, torch_to_sitk
from twaibrain.brainpreprep.utils.resample import sitk_resample
from twaibrain.braintorch.utils.fit_to_mask import get_fit_coords, fit_to_mask
from twaibrain.braintorch.utils.resize import crop_or_pad_dims
from twaibrain.braintorch.utils.fit_to_mask import reverse_fit_to_mask
from twaibrain.braintorch.utils.resample import torch_resample
import torch
import SimpleITK as sitk

def load_and_format_flair_image(image_path, mask_path=None, mask_pad=1, output_spacing=[1,1,2], output_shape=[80, 192, 160], stack_mask=True, is_label=False):
    """
    this function is for loading images at inference time and keeping track of the parameters needed to transform the output from a model
    back to the original space of the image after the model has been run (since our models expect a fixed input spacing and shape etc)

    can use is_label to apply this to other things (e.g synthseg output)
    """
    
    # load image and the mask, as well as resampling them, and get the orig spacing and original image shape to return for later
    img, orig_spatial_dims, orig_spacing, sitk_img = torch_load_and_resample(image_path, output_spacing, is_label=is_label, return_orig_shape=True) 
    if mask_path is not None:
        mask = torch_load_and_resample(mask_path, output_spacing, is_label=True)
    else:
        mask = img != 0

    resampled_spatial_dims = img.shape[1:]

    # fit the image to the mask, record the mask coordinates and the shape of the image after fitting to the mask.
    mask_zs, mask_xs, mask_ys = get_fit_coords(mask, mask_pad)
    img = fit_to_mask(mask_zs, mask_xs, mask_ys, img)
    mask = fit_to_mask(mask_zs, mask_xs, mask_ys, mask)
    
    mask_cropped_spatial_dims = img.shape[-3:]

    # some models take the mask as an input channel...)
    if stack_mask:
        img = torch.cat([img, mask], dim=0)

    img = crop_or_pad_dims(img, [1, 2, 3], output_shape)

    return img, {
        'orig_spacing':orig_spacing,
        'orig_spatial_dims': list(orig_spatial_dims),
        'resampled_spatial_dims':list(resampled_spatial_dims),
        'mask_bounding_coords': (mask_zs, mask_xs, mask_ys),
        'mask_cropped_spatial_dims': list(mask_cropped_spatial_dims),
        'output_spacing': list(output_spacing),
    }, sitk_img

def load_and_format_flair_image_v2(image_path, mask_path=None, mask_pad=1, output_spacing=[1,1,2], output_shape=[80, 192, 160], stack_mask=True, verbose=False, device='cpu'):
    """
    this function is for loading images at inference time and keeping track of the parameters needed to transform the output from a model
    back to the original space of the image after the model has been run (since our models expect a fixed input spacing and shape etc)

    V2 function: uses SimpleITK for the resampling
    also adds a verbose flag
    
    """

    # load image and the mask
    orig_flair_image = sitk.ReadImage(image_path)
    orig_spacing = orig_flair_image.GetSpacing()
    orig_spatial_dims = sitk.GetArrayFromImage(orig_flair_image).shape
    orig_mask_image = sitk.ReadImage(mask_path)
    
    # load image and the mask, as well as resampling them, and get the orig spacing and original image shape to return for later
    flair_image = sitk_resample(orig_flair_image, output_spacing, is_label=False, verbose=verbose)
    mask_image = sitk_resample(orig_mask_image, output_spacing, is_label=True, verbose=verbose)

    img = sitk_to_torch(flair_image, device)
    mask = sitk_to_torch(mask_image, device)
    
    resampled_spatial_dims = img.shape[1:]

    # fit the image to the mask, record the mask coordinates and the shape of the image after fitting to the mask.
    mask_zs, mask_xs, mask_ys = get_fit_coords(mask, mask_pad)
    img = fit_to_mask(mask_zs, mask_xs, mask_ys, img)
    mask = fit_to_mask(mask_zs, mask_xs, mask_ys, mask)
    mask_cropped_spatial_dims = img.shape[-3:]
    if verbose:
        print(f"mask zs, xs, ys: {mask_zs}, {mask_xs}, {mask_ys}")
        print("mask cropped spatial dims: ", mask_cropped_spatial_dims)

    # some models take the mask as an input channel...)
    try:
        if stack_mask:
            img = torch.cat([img, mask], dim=0)
    except Exception as e:
        print("shapes not matching")
        print(img.shape, mask.shape, orig_flair_image.GetSize(), orig_mask_image.GetSize())
        raise e

    img = crop_or_pad_dims(img, [1, 2, 3], output_shape)

    return img, {
        'orig_spacing':orig_spacing,
        'orig_spatial_dims': list(orig_spatial_dims),
        'resampled_spatial_dims':list(resampled_spatial_dims),
        'mask_bounding_coords': (mask_zs, mask_xs, mask_ys),
        'mask_cropped_spatial_dims': list(mask_cropped_spatial_dims),
        'output_spacing': list(output_spacing),
    }, orig_flair_image


def reformat_model_prediction(pred, orig_spacing, orig_spatial_dims, resampled_spatial_dims, mask_bounding_coords, mask_cropped_spatial_dims, output_spacing, is_label=True):
    """
    this function is for taking an output from a model and returning that output to the space of the original image using the parameters returned
    from the `load_and_format_flair_image' function.
    """
    
    if len(pred.shape) != 3:
        raise ValueError(f"input must be a 3D input, but received a {len(pred.shape)}D object")
    
    pred = crop_or_pad_dims(pred, [0, 1, 2], mask_cropped_spatial_dims)

    pred = reverse_fit_to_mask(*mask_bounding_coords, img=pred, orig_spatial_dims=resampled_spatial_dims)

    pred = torch_resample(pred.unsqueeze(0).type(torch.float32), orig_spacing, output_spacing, is_label=is_label).squeeze()
    pred_shape = list(pred.shape)
    
    if pred_shape[0] != orig_spatial_dims[0] or pred_shape[1] != orig_spatial_dims[1] or pred_shape[2] != orig_spatial_dims[2]:
        print(f"output_spacing not an exact match, doing final crop/pad.\noriginal shape: {orig_spatial_dims}\ncurrent shape:{pred.shape}")
        pred = crop_or_pad_dims(pred, [0, 1, 2], orig_spatial_dims)

    return pred

def reformat_model_prediction_v2(pred, orig_spacing, orig_spatial_dims, resampled_spatial_dims, mask_bounding_coords, mask_cropped_spatial_dims, output_spacing, is_label=True, interpolation='linear'):
    """
    this function is for taking an output from a model and returning that output to the space of the original image using the parameters returned
    from the `load_and_format_flair_image' function.

    V2 version uses the sitk resample function
    """
    curr_device = pred.device
    
    if len(pred.shape) != 3:
        raise ValueError(f"input must be a 3D input, but received a {len(pred.shape)}D object")
    
    pred = crop_or_pad_dims(pred, [0, 1, 2], mask_cropped_spatial_dims)

    pred = reverse_fit_to_mask(*mask_bounding_coords, img=pred, orig_spatial_dims=resampled_spatial_dims)

    pred = torch_to_sitk(pred, output_spacing)
    pred = sitk_resample(pred, orig_spacing, is_label=is_label, interpolation=interpolation)
    pred = sitk.GetArrayFromImage(pred)
    pred_shape = pred.shape

    # print("reformatted shape vs orig shape: ", pred_shape, orig_spatial_dims)
    
    if pred_shape[0] != orig_spatial_dims[0] or pred_shape[1] != orig_spatial_dims[1] or pred_shape[2] != orig_spatial_dims[2]:
        print(f"output_spacing not an exact match, doing final crop/pad.\noriginal shape: {orig_spatial_dims}\ncurrent shape:{pred.shape}")
        pred = torch.from_numpy(pred).to(curr_device)
        pred = crop_or_pad_dims(pred, [0, 1, 2], orig_spatial_dims)
        pred = pred.cpu().numpy()
        
    return pred
