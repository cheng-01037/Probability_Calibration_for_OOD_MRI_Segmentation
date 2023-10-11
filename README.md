# Post-hoc probability calibration for cardiac MRI segmentation under domain shift
[MICCAI UNSURE 2022](https://link.springer.com/chapter/10.1007/978-3-031-16749-2_6)

----

**NOTE: under construction, may still be buggy**

### 0. Dependencies
```
dcm2nii
jupyter==1.0.0
nibabel==2.5.1
notebook==6.0.2
numpy==1.15.1
opencv-python==4.1.1.26
Pillow==5.3.0
sacred==0.7.5
scikit-image==0.17.2
scipy==1.1.0
segmentation-models-pytorch==0.1.3
SimpleITK==1.2.3
tensorboardX==1.4
torch==1.3.0
torchvision==0.4.1
tqdm==4.32.2
```

### 1. Tunning the calibration model

Running the tunning of the calibration model on the frozen segmentation networks: 

`bash exp_scripts/exp_tune_proposed.sh <gpu_id> <note>`

### 2. Evaluating the calibration result

`bash exp_scripts/exp_test_proposed.sh <gpu_id> <note>`

### 3. Pre-trained frozen segmentation models and calibration models

[link](TBD)

### 4. Artifact-corrupted cardiac MRI

This is inspired by a [related work](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_14) and the data is curated from ACDC dataset. To get access to the curated version, please drop me an email with the data access certificate you obtained from the original owner of ACDC and we can share it with you upon reasonable request.

### Acknowledgments

[Local temperature scaling](https://github.com/uncbiag/LTS), [Calibration visualization](https://github.com/hollance/reliability-diagrams), [Insights on curated data for cardiac MRI](https://github.com/cherise215/Cooperative_Training_and_Latent_Space_Data_Augmentation)




