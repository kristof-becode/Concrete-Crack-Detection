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

## Packages used

- Numpy: a scientific computation package
- Pandas: a data analysis/manipulation tool using dataframes
- OpenCV: open source computer vision and machine learning software library
- Tensorflow Keras: Deep Learning API

## Datasets

|  | |  |
|-------|-------| ------- |
||  |  |

## Quick 'n dirty Keras CNN

## Image preprocessing

## Further work

Because of the lack of relevant training data with regard to image size, image segmentation would be an interesting route to investigate. This would also make it possibly easier to estimate crack dimension. 

Tiling the images and calculating a compound score for crack detection would also prove to be a fast and easy addition to my work.

