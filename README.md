# Roof Segmentation from Aerial Imagery

This repository contains code for project "Roof Segmentation from Aerial Imagery", submitted as final project for CSCI-B657 course.


## Dataset 
[Aerial Imagery for Roof Segmentation](https://www.airs-dataset.com)

## Usage details

Install dependencies 
```
pip3 install -r roof_segmentation.txt
```

Run the notebooks in the following order

- Training.ipynb : Contains data preprocessing and training code. 
- Inference.ipynb : Contains code to perform inference on the test data, calculate dice score, and overlay prediction masks on the image.
- metrics.ipynb  : This notebook plots loss curves and dice score curves for training and validation set.


Parts of codes in this repository are inspired from a previous project one of our team member (Sripad) worked on. The said project is not submitted as assignment/project to any of the courses at Indiana University or elsewhere and was for an assessment test the team member gave for a research assistantship at CNS at IU . The repository of the said project is - https://github.com/j-sripad/colon_crypt_segmentation .
