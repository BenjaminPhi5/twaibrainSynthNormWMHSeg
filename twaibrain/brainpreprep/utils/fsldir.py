import os

def fsldir():
    """
    get the fsl directory
    """
    FSLDIR = os.getenv('FSLDIR')
    print("fsl: ", FSLDIR)
    if FSLDIR == "" or FSLDIR == None:
        raise ValueError("FSL is not installed. Install FSL to complete bias correction")
    if os.getenv('FSLOUTPUTTYPE') != 'NIFTI_GZ':
        raise ValueError("FSL output type must be configured to NIFTI_GZ")
    return FSLDIR
