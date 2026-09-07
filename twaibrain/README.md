# TwaiBrain (trustworthy AI Brain)

This is the new library for preprocessing brain mri data for the trustworthai project. Subject frankly to being renamed....

## preprocessing steps:

- dicom to nifti conversion - maybe don't save the nifti in future but create the niftis as we go?
- analyze to nifti conversion
- image loading (may be extracting from a zip file, or a .nii or a .nii.gz or from a tar.gz)
- orientation fixing (make all the images the same orientation - clinical orientatoin apparently)
- registration (rigid or flexible?)
- brain extraction (SynthStrip) - first applied to an anatomical image

### augmentations for later
- i have these written down somewhere for my plan for the pretrained model.
- when doing super resoltuion (i.e downsample first) only downsample in dimensions that are already high quality, the axial apperance can be misleading when some images look amazing in axal but the slice thickness is super big.

### anatomical

- bias correction (we are now NOT going to do this, as per discussion with maria)
- brain extraction
- normalization (z-score not min max)
- running of synthseg and calculation of the 'distance to the ventricles' map


T2_SE; DeFaced <- Axial T2 TSE with Fat Sat REPEAT
T2_SE; DeFaced <- Axial T2 TSE with Fat Sat     straight axial do not oblique

### diffusion

- normalization

### fMRI

- normalization

### PET

- normalization?

## preprocessed data filestructure:

dataset_name/
    readme.md
    files_spreadsheet
    domain-x/ # can be omitted if the dataset does not have domains
        sub-0x/
            ses-0x/ # can be omitted if the dataset is not longitudinal
                anat/
                func/
                diffusion/
                ct/
                pet/
                derivatives/
                
information that I need to know about each participant:

participant_ID : (sub)
imageing_domain: (domain)
longitudinal imaging session: (ses)
image aquisition date : (date)
format (dcm, nii, nii.gz, load from zip or load from tar.gz)
modality name (e.g t1, t1_with_contrast, flair, dti, dwi, fmri_resting_state or derived data e.g WMH seg etc)
image or derivative data (it's category can be anat, diffusion, func, ct, pet, derivative)
label or not (only applies for derivatives)
main or not (each session for each patient must have a 'main' image. This is the image that is used to skull strip the image and also to get the anatomy segmentations via synthseg)
bias_corr or not (whether bias correction is to be applied to this file)
the image modality - i.e is it t1, t1 with contrast, flair etc. I should keep an enum of allowed (i.e known) modalities.
I need to separate image type into the generic type and the specific name for the image.
to save space, I should store all the outputs of the model in a single file, one file per individual? but then at training time will I be loading files I'm not using?
well maybe if I am loading the derived data unnessearily....

so the idea is we just build a spreadsheet for each dataset like this and in this way figure out how to preprocess each dataset nicely. Good.

at preprocessing time, images will be grouped by participant and by session.

for longitudinal studies, all images need to be registered to session 0. Nice.

I shall preprocess all images of all type, but say there are multiple t1 images then at training time I will pick a random one. Nice.

I should instead run the preprocessing across all the images at once for speed?

for the zippped datasets, if it is .nii then we can extract one file at a time from the zip, but if it is dcm then we need to unzip
the whole thing first.

images that are named gre, I need to be carefull about whether they are a t1 or a t2star so if it says gre in anything, I need to go back and check
explicitly what kind of image it is. I am now making three distinctions, a GRE that looks visibily t1 weighted to me, is GREt1, a GRE that is stated
as t2 weighted is T2star and a GRE that looks t2 to me but is just stated as t1 is now a standard GRE, so we have three categories, T2star, GRE and GREt1. see this for trying to understand it: https://mriquestions.com/spoiled-gre-parameters.html

anthing that is pddouble or t2 from a loni database needs to have it's weighting checked before we preprocess it.

if there is cor or tra in a flair then see if they have a normal flair first I think.

in oasis there will be duplicates of the same image type for the same subject (so pick the one with best resolution maybe?)

## steps to undertake for preprocessing the images

- [ ] figure out exactly what modalities I have across each dataset
    - [x] update the structure I want to have for the fileparser maps
    - [x] create an enum dictionary of all the possible modalities that I have
    - [ ] for each dataset, go through and create a list of each filetype that we have (and save as a csv)
    - [ ] finish creating the fileparser maps for each dataset
    - [ ] organise the download for 4rtni dataset
    - [x] organise the download of every flair + t1 timepoint for all individuals across all of adni who have a maximum time difference more than 3 years
    - [ ] understand when we have DTI / DWI and when we have derived data (Trace, ADC etc)
    - [ ] do the same for the SWI and perfusion scans.
- [ ] dicom to nifti (required for only some datasets). One of the datasets requires some further reconstruction I think. Use my existing code.
    - [ ] find my existing code and use that
    - [ ] for the images stored on the loni database, I can use the image id's that I have saved in order to know what the image datatype is
    - [ ] for the step above, I need to store a mapping from image id to the description and then from the description to a specific image type
- [ ] Brain Extraction: SynthStrip only
- [ ] registration. Thing is, the registration wants to occur after brain extraction. Okay simple, if registration is required, do brain extraction first.
    - [ ] compare the examples of the SITK vs the FSL registration vs the recommended tool: niftyreg through TracTor or nonlinear via SPM12 or SyN algorithm for NATS
    - [ ] try using easy-reg. It is built into freesurfer and uses the same ideas as synthseg and synthstrip. Nice. However, I need to see how well it performs with other images.
    - [ ] record the reference and target image. so there should be a single reference image that every other image is registered too.
- [ ] Bias Correction:
    - [ ] time N4ITK vs FSL FAST vs the simpleITK N4ITK with downsampling and masking. it should be faster. Nice.
    - [ ] actually, the synthseg paper points out the fact that augmenting beyond realism tends to improve performance in the long run. So, instead, what we should do is train on a large quantity of data and implement a bias correction augmentation, like they do in synthseg. I should see what effect this has on performance when trying to generalize the model, while reducing pre-processing time.
- [ ] Intensity Scaling
    - [ ] switching to MinMax scaling now
    - [ ] along with restricting the percentiles that are used for the normalization
    - [ ] have the option to do this on cuda or not
- [ ] Anatomy and Pathology outputs (SynthSeg, SamSeg). This is done at the end on the final output. What does SamSeg run on? For consistency, I should run it on the same modality, maybe I run it on FLAIR only every time?
- [ ] so: does DTI, SWI, perfusion and anything else I come across need preprocessing
- [ ] so: each time I come across a modality, do I understand eactly what it is? that is an important question here, particularly in terms of, how do I preprocess it?


### list of all the modalities I come accross and whether I understand them:
- [ ] t1
- [ ] t1 with contrast
- [ ] t2
- [ ] t2*
- [ ] t2 with contrast
- [ ] flair
- [ ] gre
- [ ] dti
- [ ] dwi
- [ ] swi
- [ ] perfusion
- [ ] trace
- [ ] adc
- [ ] ct: for now we are ignoring
- [ ] pet: for now we are ignoring
- [x] fmri: for now we are ignoring

#### T1

#### T2

#### FLAIR

#### DTI

#### SWI

#### DWI




## postprocessing:

optional resampling, cropping and file collation

## notes on experiments to run

### bias correction

problems:
it is slow
it prevents adoption
it can go wrong because you need to tune the parameters
it can go wrong because it gets affected by presence of skull, okay I skull strip but could this generalize to tumors and other issues
there are multiple aglorithms and some suggest other approaches (based on pixel map?)

emperical issue:
it is unclear whether model performance is affected by this or not because the size of the training data is small
unclear whether parameters need to be changes for other modality types

how to investigate the issue:
train an ensemble on wmh segmentation
train the model with t1 and flair that has been bias field corrected
train the model on a mix of CVD and WMH Challenge Dataset
then extract explicitly the epistemic uncertainty

then manipulate the bias field, or remove the bias field entirely
assess the differences in the segmentation

then repeat the whole process, training the model on just the t1 image
(all in all, train a model on bias corrected, not bias corrected, not biascorr t1 only, biascorr t1 only)

see whether where the model degrades in performance is highlighted by the epistemic uncertainty? I could also add to this - movement artefacts, ghosting. Nice.

then - pretrain a model on a large number of other images from other datasets, do one for t1 and flair, one for t1 only
train as an ensemble to get epistemic uncertainty estimates
finetune that model on the wmh segmentation task (again the 4 models and again an ensemble)

see if the pretrained model without the bias correction is able to cope with the issue or not (assuming that there is, in general, an issue).
finally, train the pretraining model with random bias correction augmentation applied to the input image. Nice, I like that.

### initial investigation of wmh progression over time in longitudinal datasets
get a model for segmenting wmh in flair and t1. 
For literally every individual I have in all datasets, for t1 and flair, segment WMH amongst all the images.
adjust for ICV volume
do plots for each dataset of WMH over time vs age, then adjust for CN, MCI, AD, sex etc.
then look at individuals, see how the WMH progresses for indivduals, across each cohort. Nice.

### for the rest, see the main paper ideas document. Nice.