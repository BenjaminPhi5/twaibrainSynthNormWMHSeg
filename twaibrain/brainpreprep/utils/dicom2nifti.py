import SimpleITK as sitk
import shutil
import os
import subprocess

def dicom_nifti_conversion(in_folder:str, out_file:str, gzip:bool=False, rm_dcm:bool=False, force:bool=False):
    """
    USING DCM2NIIX: requires dcm2niix is installed
    in_folder: path to the folder containing the dcms for a single series
    out_file: path to the desired output filename
    gzip: whether to use gzip compression in savedd image
    rm_dcm: whether to delete the dicom folder after doing the conversion
    force: whether to continue if the file already exists
    """
    
    # parse output filename
    out_splits = out_file.split("/")
    out_folder = "/".join(out_splits[:-1])
    out_filename = out_splits[-1]
    out_filename = out_filename.split(".nii")[0] # dcm2niix adds any fileendings
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # check if file exists already and follow force param
    existing_outfiles = os.listdir(out_folder)
    exists = False
    for f in existing_outfiles:
        if f.startswith(out_filename):
            exists = True
            break
    if exists:
        if force:
            print("nifti image already exists, deleting existing nifti")
            command = f"rm {os.path.join(out_folder,out_filename)}*"
            os.system(command)
        else:
            print("nifti image already exists and force is set to False, doing nothing")
            return       
    
    # run dcm2niix
    command = f"dcm2niix -o {out_folder} -f {out_filename} -z {'y' if gzip else 'n'} {in_folder}"
    subprocess.run(command.split(" "), stdout=subprocess.DEVNULL)
    
    # check if image is stored as second echo (occurs for the pd/double modality...)
    # or if the image is the real part of a complex image (not sure why this occurs)
    failed = False
    fileending = ".nii" + (".gz" if gzip else "")
    filepathname = os.path.join(out_folder, out_filename)
    
    e2_filepath = filepathname + "_e2" + fileending
    real_filepath = filepathname + "_real" + fileending
    e2_real_filepath = filepathname + "_e2_real" + fileending
    ph_filepath =  filepathname + "_ph" + fileending
    standard_filepath = filepathname + fileending
    if not os.path.exists(standard_filepath):
        if os.path.exists(e2_filepath):
            os.rename(e2_filepath, standard_filepath)
        elif os.path.exists(real_filepath):
            os.rename(real_filepath, standard_filepath)
        elif os.path.exists(e2_real_filepath):
            os.rename(e2_real_filepath, standard_filepath)
        elif os.path.exists(ph_filepath):
            print("warning: phase file, not brain image, ignoring.")
        else:
            failed = True
            
            
    # delete any files created by dcm2niix that we do not want
    # note that thsi will not delete bval and bvec files needed for DTI (I don't think they are nifti files)
    files = os.listdir(out_folder)
    for file in files:
        if file.startswith(filepathname):
            if file.endswith(".json") or (".nii" in file and file != standard_filepath):
                os.remove(os.path.join(out_folder, file))
    
    # remove the dcm folder
    if rm_dcm:
        files = os.listdir(in_folder)
        for f in files:
            if not f.endswith(".dcm"):
                raise ValueError(f"found a non dcm file/folder within {in_folder}, hence not safe to delete, aborting: {f}")
        shutil.rmtree(in_folder)
                
    # raise error if converting this file failed
    if failed:
        raise ValueError(f"error: dcm2niix did not provide desired output file: {standard_filepath}")
    


def dicom2nifti(infolder, outfile, rm_dcm=False):
    """
    USING SIMPLEITK
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(infolder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    image = sitk.DICOMOrient(image, 'RAS')
    
    if not outfile.endswith(".nii.gz") and not outfile.endswith(".nii"):
        outfile = outfile + ".nii.gz"
    sitk.WriteImage(image, outfile)
    
    if rm_dcm:
        files = os.listdir(infolder)
        for f in files:
            if not f.endswith(".dcm"):
                raise ValueError(f"found a non dcm file/folder within {infolder}, hence not safe to delete, aborting: {f}")
        shutil.rmtree(infolder)
