
for_training - v5 2024-05-31 12:09am
==============================

This dataset was exported via roboflow.com on June 26, 2024 at 5:44 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1430 images.
No_lesions-Ze7P are annotated in Tensorflow Object Detection format.

The following pre-processing was applied to each image:

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -11 and +11 degrees
* Random brigthness adjustment of between -24 and +24 percent
* Random Gaussian blur of between 0 and 1.1 pixels

The following transformations were applied to the bounding boxes of each image:
* Random shear of between -3° to +3° horizontally and -3° to +3° vertically
* Random exposure adjustment of between -3 and +3 percent


