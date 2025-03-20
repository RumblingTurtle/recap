# Downloading datasets

Run ```./prepare_data.sh``` script to prepare the folder structure for the datasets. It will additionally download the CMU dataset and extract it into ```./data/allasfamc``` subfolder.

## AMASS dataset

1) Register on the [AMASS website](https://amass.is.tue.mpg.de/) and download the datasets you want to use. Make sure to download SMPL-H data (gender doesn't matter). Put the downloaded files into ```./data/amass/datasets``` subfolder.

2) Register and download the [SPML-H model](https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz) and extract it into ```./data/amass/body_models``` subfolder

3) Register and download the [DMPLs](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=dmpls.tar.xz) and extract it into ```./data/amass/body_models``` subfolder


