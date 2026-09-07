# twaibrainSynthNormWMHSeg
WMH segmentation model trained with SynthNorm. Instructions:

1) create conda environment (other environment managers can be used instead). For more information in installing (mini)conda see here: https://continuumio-docs.readthedocs-hosted.com/miniconda/
```
conda create env -n segmodel python=3.12

```

2) install package
```
conda activate segmodel
git clone git@github.com:BenjaminPhi5/twaibrainSynthNormWMHSeg.git
cd twaibrainSynthNormWMHSeg
pip install -e .
```

3) download model weights from [here](https://datasync.ed.ac.uk/index.php/s/LDqLVfv2RXEdRVO) using password `weights`. Unzip the model weights. large_ssnens and small_ssnens refer to two models of different sizes. The small model is quicker to run and the weights are much smaller in size, but it has not been extensively tested. Nonetheless, the segmentations appear consistent visually. 


4) requirements for the images:

- FLAIR image is skull stripped (no bias correction is necessary)
- RAS orientation
- FLAIR and brain tissue mask (e.g. output of SynthStrip) provided
- .nii.gz format


5) to run the script you will need to provide a comma separated csv file containing the following columns: <image name/id>,<flair image path.nii.gz>,<brain mask path.nii.gz> . The csv file should have no headings (see eval_test_filenames.csv as an example). You can optionally provide the parent folder of all the images under the --imgfolder flag instead of storing the full filepath in the csv. This csv is provided as input to the model with the `-i` flag. 

Specify where you want the resulting segmentations saved using the `-o` flag.


```
python run_model.py -i <image_filepaths.csv> --imgfolder optional/folder/where/images/are -o output/folder/ --verbose --model_name small_ssnens --models_folder <e.g path/to/small_ssnens_weights/>

```

The model will output 4 files:
- <image_name>_<model_name>_wmhseg.nii.gz  : binarised WMH segmentation
- <image_name>_<model_name>_wmhprob.nii.gz : soft WMH probabilities
- <image_name>_<model_name>_samples.nii.gz : 12 plausible binarised segmentations, reflects the uncertainty in the segmentation
- <image_name>_<model_name>_uqimg.nii.gz   : uncertianty map (predictive entropy of the model ensemble)

example run command on my machine:

```
python run_model.py -i eval_test_filenames.csv --imgfolder /run/media/benphilps/NVMEBenSpare/thesis_work/testing_images/ -o /run/media/benphilps/NVMEBenSpare/thesis_work/test_out --verbose --model_name large_ssnens -z --models_folder /run/media/benphilps/NVMEBenSpare/thesis_work/chapter_5_checkpoints/V4_training_UQ_scripts_model_checkpoints

```