import torch

def get_fit_coords(mask, pad):
    mask = mask.squeeze()

    wheres = torch.where(mask !=0)
    zs = wheres[-3].min().item(), wheres[-3].max().item()
    xs = wheres[-2].min().item(), wheres[-2].max().item()
    ys = wheres[-1].min().item(), wheres[-1].max().item()
    
    shape = mask.shape
    
    zs = max(0, zs[0] - pad), min(shape[0], zs[1] + pad)
    xs = max(0, xs[0] - pad), min(shape[1], xs[1] + pad)
    ys = max(0, ys[0] - pad), min(shape[2], ys[1] + pad)

    return zs, xs, ys

def fit_to_mask(zs, xs, ys, img):
    if len(img.shape) == 4: # my presumption for the dataloader
        return img[:, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
    elif len(img.shape == 5):
        return img[:, :, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
    elif len(img.shape) == 3:
        return img[zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
    else:
        raise ValueError(f"img should be 3D, 4D or 5D, not {len(img.shape)}D")


def reverse_fit_to_mask(zs, xs, ys, img, orig_spatial_dims):
    """
    reverses the fit to mask function by taking an image and placing it in a zero valued image of the same size of the pre mask extracted image
    at the position where the mask appears.
    """
    new_shape = list(img.shape)
    new_shape[-3] = orig_spatial_dims[0]
    new_shape[-2] = orig_spatial_dims[1]
    new_shape[-1] = orig_spatial_dims[2]
    
    new_image = torch.zeros(new_shape, dtype=img.dtype)

    if len(img.shape) == 4:
        new_image[:, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1] = img
    elif len(img.shape) == 5:
        new_image[:, :, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1] = img
    elif len(img.shape) == 3:
        new_image[zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1] = img
    else:
        raise ValueError(f"img should be 3D, 4D or 5D, not {len(img.shape)}D")

    return new_image
    
