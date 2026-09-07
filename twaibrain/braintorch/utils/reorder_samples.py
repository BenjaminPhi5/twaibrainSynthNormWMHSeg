import torch

def reorder_samples(sample):
    """
    reorder samples by slice from lowest volume to highest volume, sample should either be 2 class or 1 class
    and shape S,N,C,<dims> (S is number of samples, not batch size here, batch is effectively 1).
    given that this is 2D resampling.
    """
    if sample.shape[1] > 1:
        slice_volumes = sample.argmax(dim=1).sum(dim=(-1, -2))
    else:
        slice_volumes = (sample > 0.5).sum(dim=(-1, -2))
    slice_volume_orders = torch.sort(slice_volumes.T, dim=1)[1]
    
    # rearrange the samples into one...
    new_sample = torch.zeros(sample.shape).to(sample.device)
    for i, slice_volumes_orders in enumerate(slice_volume_orders):
        for j, sample_index in enumerate(slice_volumes_orders):
            new_sample[j][i] = sample[sample_index][i]
            
    return new_sample
