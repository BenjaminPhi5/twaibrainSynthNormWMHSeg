import SimpleITK as sitk
import torch
from twaibrain.braintorch.utils.resample import torch_resample
import numpy as np

def torch_load(imgpath):
    sitk_image = sitk.ReadImage(imgpath)
    return torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).type(torch.float32).unsqueeze(0)

def torch_load_with_spacing(imgpath, device='cpu'):
    """
    loads nifti image and returns image spacing as well
    """
    img = sitk.ReadImage(str(imgpath))
    spacing = np.array(img.GetSpacing())
    
    # Convert to numpy then to torch
    arr = sitk.GetArrayFromImage(img)
    tensor = torch.from_numpy(arr).float().to(device)
    
    return tensor, spacing

def sitk_to_torch(sitk_image, device='cpu'):
    return torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).type(torch.float32).unsqueeze(0).to(device)

def torch_to_sitk(arr, spacing):
    arr = arr.cpu().squeeze().numpy()
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img

def torch_load_and_resample(imgpath, out_spacing, orig_spacing=None, is_label=False, return_orig_shape=False):
    sitk_img = sitk.ReadImage(imgpath)
    if out_spacing is None:
        raise ValueError("out spacing must be defined")
    if orig_spacing is None:
        orig_spacing = sitk_img.GetSpacing()

    img = torch.from_numpy(sitk.GetArrayFromImage(sitk_img)).type(torch.float32).unsqueeze(0)

    orig_spatial_dims = img.shape[1:]

    img = torch_resample(img, out_spacing, orig_spacing, is_label=is_label)

    if return_orig_shape:
        return img, orig_spatial_dims, orig_spacing, sitk_img
    
    return img
