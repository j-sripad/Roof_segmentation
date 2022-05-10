# Roof Segmentation from Aerial Imagery

This repository contains code for project "Roof Segmentation from Aerial Imagery", submitted as final project for CSCI-B657(Computer Vision) course.

Team members :
 - Sripad Joshi (joshisri)
 - Srimanth Agastyraju (sragas)
 - Himani Shah (shahhi)
 - Karan Acharya (karachar)

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


Some parts of codes in this repository are inspired from a previous project one of our team members (joshisri: Sripad Joshi) worked on. The said project was his personal project and was not submitted as any assignment/project to any of the courses at Indiana University or elsewhere. It can be found at - https://github.com/j-sripad/colon_crypt_segmentation .
