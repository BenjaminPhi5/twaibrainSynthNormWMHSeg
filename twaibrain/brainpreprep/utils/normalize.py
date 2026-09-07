import numpy as np
import SimpleITK as sitk
from twaibrain.brainpreprep.utils.image_io import save_manipulated_sitk_image_array, load_image

def torch_normalize_brain(image, mask, lower_percentile=0, upper_percentile=100, verbose=False, type_check=True):
    """
    function to normalize the brain within the mask area for torch tensors only
    """
    
    # image = image.copy()
    mask = (mask==1)
    brain_locs = image[mask]
    
    if lower_percentile > 0 or upper_percentile < 100:
        if verbose:
            print("normalizing with percentiles: ", lower_percentile, upper_percentile)
        
        lower_percentile /= 100
        upper_percentile /= 100
        
        brain_locs = brain_locs.view(-1)
        sorted_indices = brain_locs.argsort()
        num_brain_voxels = len(sorted_indices)
        #print(num_brain_voxels)

        lower_index = int(lower_percentile*num_brain_voxels)
        upper_index = int(upper_percentile*num_brain_voxels)

        retained_indices = sorted_indices[lower_index:upper_index]
        #print(len(retained_indices)/num_brain_voxels)
        
        brain_locs = brain_locs[retained_indices]
    else:
        if verbose:
            print("no percentiles used for normalization")

    mean = brain_locs.mean()
    std = brain_locs.std()
    
    if verbose:
        print("mean: ", mean.item(), "SD: ", std.item())

    image[mask] = (image[mask] - mean) / std
    image[~mask] = 0

    return image

def normalize_brain(image, mask, lower_percentile=0, upper_percentile=100, verbose=False, type_check=True):
    """
    function to normalize the brain within the mask area
    """
    # convert to float
    if type_check:
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
    
    # image = image.copy()
    mask = (mask==1)
    brain_locs = image[mask]
    
    if lower_percentile > 0 or upper_percentile < 100:
        if verbose:
            print("normalizing with percentiles: ", lower_percentile, upper_percentile)
        
        lower_percentile /= 100
        upper_percentile /= 100
        
        brain_locs = brain_locs.flatten()
        sorted_indices = np.argsort(brain_locs)
        num_brain_voxels = len(sorted_indices)
        #print(num_brain_voxels)

        lower_index = int(lower_percentile*num_brain_voxels)
        upper_index = int(upper_percentile*num_brain_voxels)

        retained_indices = sorted_indices[lower_index:upper_index]
        #print(len(retained_indices)/num_brain_voxels)
        
        brain_locs = brain_locs[retained_indices]
    else:
        if verbose:
            print("no percentiles used for normalization")

    mean = np.mean(brain_locs)
    std = np.std(brain_locs)
    
    if verbose:
        print(mean, std)

    image[mask] = (image[mask] - mean) / std
    image[~mask] = 0

    return image

def normalize_brain_file(image_path, mask_path, out_path, lower_percentile=0, upper_percentile=100):
    img = sitk.ReadImage(image_path)
    img_arr = sitk.GetArrayFromImage(img)
    mask_arr = load_image(mask_path)
    
    img_arr = normalize_brain(img_arr, mask_arr, lower_percentile, upper_percentile)
    
    save_manipulated_sitk_image_array(img, img_arr, out_path)
