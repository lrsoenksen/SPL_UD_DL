# Deep Learning for Dermatologist-Level Detection of Ugly-Duckling (UD) and Suspicious Pigmented Skin Lesions (SPL) from Wide-Field Images
Code to reproduce [Soenksen, LR. et al 2021, on Science Translational Medicine](https://static1.squarespace.com/static/5c264953620b850c9fb03732/t/602d85e4d46da532404689f2/1613596138740/stm_luis.pdf).

![Image description](src/notebook_imgs/SPL_UD_DL_Fig1.jpg)


#### CODE STRUCTURE (NOTEBOOKS / INPUTS / OUTPUTS)

> Samples of data Preparation, model training, testing and integrated analysis system according to the methods of Soenksen, LR. et al 2020, can be executed through the included Jupyter notebooks in the following order:

* 00_A_DL_Image_Patch_generation.ipynb
* 00_B_DL_Image_Database_CLAHE_PreProcessing.ipynb
* 00_C_DL_Image_Database_Randomization.ipynb
* 00_D_DL_Image_Augmentation_of_Randomized_CLAHE_Database.ipynb
* 01_DL_SPL_Detection_Basic_Model_Creator.ipynb
* 02_DL_SPL_Detection_Augmented_Model_Creator.ipynb
* 03_DL_SPL_Detection_Augmented_TL_VGG16_Bottleneck_Model_Creator.ipynb
* 03_DL_SPL_Detection_Augmented_TL_VGG16_Fine_Tuning_Model_Creator.ipynb
* 04_DL_SPL_Detection_Augmented_TL_XCEPTION_Bottleneck_Model_Creator.ipynb
* 04_DL_SPL_Detection_Augmented_TL_XCEPTION_Fine_Tuning_Model_Creator.ipynb
* 05_DL_SPL_A_Wide_Field_Feature_Extractor_UglyDucking_Ranking_and_T-SNE.ipynb

#### PROBLEM/SOLUTION DEFINITION
> Wide-field imaging and deep neural networks are used to facilitate the accurate detection of suspicious and salient pigmented lesions to allow for convenient skin screenings at the primary care level.

#### DATASET (Direct Download)
> All data has been de-identified and randomized to comply with MIT data sharing policies. Due to egress limits on GIT, this repo requires that you download the "Wide-field" and "close-up" Dataset directly from this link: https://www.dropbox.com/sh/wf4cqy5odiborbq/AAAkgUHeaYELYbq-93gkOWZca?dl=0. Data and code are provided ONLY for non-commertial purposes. After download place in the main project folder to use with the provided code.
 
#### MODELS (Direct Download)
> Due to egress limits on GIT, this repo requires that you download the following "Outputs" folder, which includes the trained Deep Convolutional Neural Network (DCNN) model weight files directly from this link: https://www.dropbox.com/s/n0aj1ezzh99uyjd/Models.zip?dl=0. After download place in the main project folder and unzip.
