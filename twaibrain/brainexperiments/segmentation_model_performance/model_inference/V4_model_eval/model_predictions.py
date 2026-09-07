import torch

def ensemble_single_image_pred_standard(inp, ensemble, device):
    inp = inp.unsqueeze(0)
    outs = []
    with torch.no_grad():
        for model in ensemble:
            outs.append(model.to(device).eval()(inp.to(device))[0].squeeze(0))

    outs = torch.stack(outs)
    mean = outs.mean(dim=0)
    samples = outs.cpu()
    mean = mean.cpu()
    return mean, samples

def ensemble_single_image_pred_ssn(inp, ensemble, device, samples_per_element=2):
    inp = inp.unsqueeze(0)
    means = []
    samples = []
    with torch.no_grad():
        for model in ensemble:
            o = model.to(device).eval().mean_and_sample(inp.to(device), num_samples=samples_per_element)
            means.append(o[0].squeeze().cpu())
            samples.append(o[1].squeeze().cpu())
    # return the mean segmentation and the samples

    mean = torch.stack(means).to(device).mean(dim=0).cpu()
    samples = torch.cat(samples, dim=0).cpu()

    return mean, samples
