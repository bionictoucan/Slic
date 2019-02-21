# Slic: Solar Image Classification using Convolutional Neural Networks
Slic is a fast image classification tool for identifying the large-scale features in a solar dataset. The initial training of Slic is done on H&alpha; &lambda;6563&#8491; data from Hinode/SOT (Solar Optical Telescope) and the training set consists of 13175 images split 90% to 10% for training to validation consisting of five features: filaments, flare ribbons, prominences, sunspots and the quiet Sun (i.e. the lack of any of the other four features).

**Note: filaments and prominences are the same physical feature but are geometrically different so are treated as different classes here.**

We provide a pre-trained model trained for 73 epochs and a learning rate of &eta;=0.0005 that reaches an accuracy of 99.24%. This will be good enough for identifying these features in visible wavelengths. We provide an example notebook of how to use this tool for dataset traversal and give instructions on how to train from scratch if needed.

## Requirements
For training:

* NVIDIA GPU
* CUDA 9.0+
* `Python 3`+
* `numpy`
* `matplotlib`
* `scipy`
* `astropy`
* `scikit-image`
* `pandas`
* `tqdm`
* `PyTorch 0.4+`

For prediction:

* `Python 3`+
* `PyTorch 0.4+`
* `numpy`
* `matplotlib`
* `tqdm`

Optional:

* `sunpy`
* `palettable`

## Usage

## Publications
