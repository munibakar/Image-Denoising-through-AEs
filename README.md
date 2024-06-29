# Image-Denoising-through-AEs

## Overview
This project focuses on denoising digit images using a Multi-Layer Perceptron (MLP) regressor. 
The dataset used is the handwritten digits dataset (`load_digits`) from the `sklearn` library.
The project consists of adding noise to the images and then training a neural network to remove the noise.

## Requirements
The following Python packages are required to run the project:
- numpy
- matplotlib
- scikit-learn
- argparse
- json

  
You can install these packages using the following command:
```bash
pip install numpy matplotlib scikit-learn argparse

```
## Files
`main.py`: Main script that performs data preprocessing, adds noise to the images, trains the MLP regressor, and evaluates the model.

`best_hyperparameters.json`: JSON file to save/load the best hyperparameters for the MLP regressor.

## Usage
Clone the repository:
```bash
git clone https://github.com/munibakar/Image-Denoising-through-AEs.git
```


### Command Line Arguments
- `--hyperparameters`: Specify whether to perform hyperparameter optimization. Options: `on` or `off`. Default: `off`.

Example usage:
```bash
python main.py --hyperparameters on

```
## Results
The following plots and visualizations will be generated:

- Best Hyperparameters: If hyperparameter optimization is performed, the best hyperparameters are printed and saved in `best_hyperparameters.json`.

- Loss Plot: A plot of training and validation loss over epochs.
  
- Denoised Images: Visualization of original, noisy, and denoised images.



