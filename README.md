## Concrete Crack Detection

This repository contains code -work in progress- for a use case with the objective of inspecting concrete surfaces from drone images.
The goal is to facilitate the inspection of concrete surfaces on real-world structures, such as buildings, bridges etc, to investigate possible degradation of concrete.

### Table of contents

* [Intro](#intro)
* [Packages used](#packages-used)
* [Datasets](#datasets)
* [Quick 'n dirty Keras CNN](#quick-n-dirty-keras-cnn)
* [Image preprocessing](#image-preprocessing)
* [Further work](#further-work)

## Intro

Steps:
- detect cracks present or not
- provide information on cracks present, i.e. size

Because of the extreme difference in images ize between available training data and the drone images provided it proved very diificult to get a good accuracy with a simple CNN. I tried to preprocess the images for classification and crack dimension estimation but didn't get satisfying results. Cutting the drone images and calculating a compound classification score might be an addition to the work done. But image segmentation would be a better approach to get a clear crack rendering and to proceed in estimating the crack dimensions.

## Packages used

- Numpy: a scientific computation package
- Pandas: a data analysis/manipulation tool using dataframes
- OpenCV: open source computer vision and machine learning software library
- Tensorflow Keras: Deep Learning API

## Datasets

* Kaggle Surface Crack Detection data: 458 high-res images (4032x3024) split into 40k (227x227)

|crack | no crack |
|-----|-----|
|20k | 20k |

Fairly clean and evenly distributed

* SDNet2018 data: 230 bridge/walls/pavement images split into +56k (256x256)

|  |crack | no crack |
|---------|-------| ------- |
|Walls| 3851 | 14278 |
|Pavement|  2608| 21726 |
|Bridge decks|  2025| 11595 |

More dirty/unclear and unevenly distributed

* Drone images: 685 + 154 (5456x3632) positive crack images

Very high resolution and challenging images

## Quick 'n dirty Keras CNN

First thing I did was to build a simple Keras CNN, trained on Kaggle concrete surface crack dataset. Validation accuracy was very good but when trying this on other datasets and the drone images the results were varying a lot depending on the image qualtity, crack visibility and presence of artifacts.

## Image preprocessing

Next thing I did was preprcess images to isolate the cracks better. I focused on Min-Max Gray Level Discrimination or M2GLD. 
* Apply extended Otsu tresholding to create binary image
* Remove shapes by contour area and size ratio axes

This worked fine for certain images but didn't scale to the very high resolution drone images.

Training a Keras CNN with these preprocessed images gave almost worse results then the original setup.

## Further work

Because of the lack of relevant training data with regard to image size, image segmentation would be an interesting route to investigate. This would also make it possibly easier to estimate crack dimension. 

Tiling the images and calculating a compound score for crack detection would also prove to be a fast and easy addition to work done.

