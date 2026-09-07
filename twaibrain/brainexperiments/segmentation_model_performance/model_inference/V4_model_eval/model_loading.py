from pathlib import Path

from twaibrain.braintorch.models.nnUNet import nnUNetV2_model_loader
from twaibrain.braintorch.models.nnUNet.nnUNetV2_model_loader import get_network_from_plans
from twaibrain.braintorch.models.ssn import SSN_Wrapped_Deep_Supervision, SSN_Wrapped_Deep_Supervision_LLO, Hierarchical_SSN_with_ConvRefine, Hierarchical_SSN_with_ConvSpatialAttention
from twaibrain.braintorch.fitting_and_inference.lightning_fitter import StandardLitModelWrapper
import json


def _get_model_config_path(model_type):
    if model_type == "large":
        config_name = "nnUNetResEncUNetMPlans.json"
    elif model_type == "small":
        config_name = "nnUNetResEncUNetMPlans-small.json"
    else:
        raise ValueError("model type unknown")

    config_dir = Path(nnUNetV2_model_loader.__file__).resolve().parent / "cvd_configs"
    return config_dir / config_name


def _load_model_config(model_type):
    with _get_model_config_path(model_type).open() as f:
        return json.load(f)


def load_standard_model(model_type='small'):
    model_config = _load_model_config(model_type)

    dims = "3d_fullres"
    config = model_config['configurations'][dims]['architecture']
    network_name = config['network_class_name']
    kw_requires_import = config['_kw_requires_import']

    model = get_network_from_plans(
        arch_class_name=network_name,
        arch_kwargs=config['arch_kwargs'],
        arch_kwargs_req_import=kw_requires_import,
        input_channels=2,
        output_channels=1,
        allow_init=True,
        deep_supervision=True,
    )

    return model


def load_ssn_model(model_type = 'small', ssn_pre_head_layers=32, ssn_rank=25):
    ssn_config = {
        'intermediate_channels':ssn_pre_head_layers,
        'out_channels':2,
        'dims':3,
        'rank':ssn_rank,
        'diagonal':False,
    }

    model_config = _load_model_config(model_type)

    dims = "3d_fullres"
    config = model_config['configurations'][dims]['architecture']
    network_name = config['network_class_name']
    kw_requires_import = config['_kw_requires_import']

    model = get_network_from_plans(
        arch_class_name=network_name,
        arch_kwargs=config['arch_kwargs'],
        arch_kwargs_req_import=kw_requires_import,
        input_channels=2,
        output_channels=ssn_pre_head_layers,
        allow_init=True,
        deep_supervision=True,
    )

    ssn_model = SSN_Wrapped_Deep_Supervision(model, 5, ssn_config)

    return ssn_model

def load_checkpoint(base_model, ckpt_path):
    litmodel = StandardLitModelWrapper.load_from_checkpoint(
        ckpt_path,
        model=base_model,
        loss=None,
        optimizer_configurator=None,
    )
    model = litmodel.model
    model = model.eval()
    return model


def load_model_ensemble(model_type, ssn, checkpoints, vprint, ssn_pre_head_layers=32, ssn_rank=25, device='cpu'):
    vprint(f"loading models of size: {model_type}")
    vprint(f"is model an SSN model: {ssn}")

    models = []

    for i, checkpoint in enumerate(checkpoints):
        vprint("loading checkpoint: ", i)
        if ssn:
            base_model = load_ssn_model(model_type, ssn_pre_head_layers=ssn_pre_head_layers, ssn_rank=ssn_rank)
        else:
            base_model = load_standard_model(model_type)

        models.append(load_checkpoint(base_model, checkpoint).to(device))

    return models
