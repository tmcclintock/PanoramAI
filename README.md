# PanoramAI [![Build Status](https://travis-ci.com/tmcclintock/PanoramAI.svg?branch=master)](https://travis-ci.com/tmcclintock/PanoramAI) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Coverage Status](https://coveralls.io/repos/github/tmcclintock/PanoramAI/badge.svg?branch=master)](https://coveralls.io/github/tmcclintock/PanoramAI?branch=master)

Panoramic images using generative networks, and (in the future) panoramic images conditional on smaller field-of-view images.

Much of the training data for this project was generated using the synthetic panoramic image tool called the [LandscapeGenerator](https://github.com/tmcclintock/LandscapeGenerator); a stand-alone tool for making synthetic landscape images.

## Installation

Install the requirements and then this repository

```bash
pip install -r requirements.txt
python setup.py install
```

The requirements are:

* [`numpy` and `scipy`](https://scipy.org/install.html)
* [`scikit-learn`](https://scikit-learn.org/stable/install.html)
* [`tensorflow`](https://www.tensorflow.org/install) >= version 2.0.0
* [`notebook`](https://jupyter.readthedocs.io/en/latest/install.html) (for running the example notebooks)
* [`matplotlib`](https://matplotlib.org/users/installing.html) (for notebooks)
* [`pytest`](https://docs.pytest.org/en/latest/getting-started.html) (for testing)

These can all be installed together using the `requirements.txt` file as shown above (assuming you have `pip`).

## Usage

PanoramAI contains models to generate panoramic images. Models provided in this package are untrained. This section documents how to instantiate and train a model, and then how to generate new panoramic images.

First, import and create a model. As an example here, we consider a convolutional variational autoencoder (`VAEorama`).

```python
import PanoramAI

model = PanoramAI.VAEorama(dataset)
```

In this example, `dataset` is a `numpy.ndarray` containing the input panoramic images for training. It should have `N` RGB images in total, meaning its shape must be (`N`,`Height`,`Width`,`3`) where the height and width are in pixels.

Once created, we train the model for some number of epochs.
```python
epochs = 100
model.train(epochs)
```

Predictions can be made in batches
```python
#Create 10 sample panoramas
samples - model.generate_samples(10)
```

### Models

At present, PanoramAI includes two fully generative models: a DCGAN and a convolutional variational autoencoder (VAE). A conditional VAE is in development.