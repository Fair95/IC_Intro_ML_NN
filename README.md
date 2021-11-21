# Neural_Networks_71
#### The Coursework 2 of `Introduction to Machine Learning`.
#### By He Liu, Kejian Shi, Qianyi Li, Shitian Jin
## I. Instruction for Program
#### This program is written based on Python, simply run `part1_nn_lib.py` and `part2_house_value_regression.py` directly to start the script of Part 1 & Part 2 respectively.

## II. Files in this project:
### `part1_nn_lib.py` 
### Part 1 of the coursework where we implemented a mini-library of the neural network. Check this file for imformation about
#### Classes:
- MSELossLayer
- CrossEntropyLossLayer
- SigmoidLayer
- ReluLayer
- LinearLayer
- MultiLayerNetwork
- Trainer
#### Functions:
- \__init__()
- forward()
- backward()
- update_params()
- shuffle()
- ...
### `part2_house_value_regression.py`
### Part 2 of the coursework where we created and trained a neural network for regression. Check this file for imformation about
#### Classes:
- LinearRegressorModel
- Regressor
#### Functions:
- \__init__()
- _preprocessor()
- fit()
- predict()
- score()
- RegressorHyperParameterSearch()
- ...

## III. Implementation of the Neural Network (If you are interested...)
### Part 1: Create a neural network mini-library
* Implemented in `part1_nn_lib.py`
#### 1.1 Implement a linear layer.
* In class LinearLayer:
- \__init__(): Constructor
- forward(): Forward pass method
- backward(): Backward pass method
- update_params(): Parameter update method
#### 1.2 Implement activation function classes.
* In class SigmoidLayer and ReluLayer:
- forward(): Forward pass method
- backward(): Backward pass method
#### 1.3 Implement a multi-layer network.
* In class MultiLayerNetwork:
- \__init__(): Constructor
- forward(): Forward pass method
- backward(): Backward pass method
- update_params(): Parameter update method
#### 1.4 Implement a trainer.
* In class Trainer:
- \__init__(): Constructor
- shuffle(): Data shuffling
- train(): Main training loop
- eval_loss(): Computing evaluation loss
#### 1.5 Implement a preprocessor.
* In class Preprocessor:
- \__init__(): Constructor
- apply(): Apply method
- revert(): Revert method

### Part 2: Create and train a neural network for regression
* In this part we aim to infer the median house value from all other attributes in a given dataset.
* The dataset `housing.csv` is given, it covers all the block groups in California from the 1990 Census, contains 20,640 observations on ten variables:
1. longitude: longitude of the block group
2. latitude: latitude of the block group
3. housing median age: median age of the individuals living in the block group
4. total rooms: total number of rooms in the block group
5. total bedrooms: total number of bedrooms in the block group
6. population: total population of the block group
7. households: number of households in the block group
8. median income: median income of the households comprise in the block group
9. ocean proximity: proximity to the ocean of the block group
10. median house value: median value of the houses of the block group

#### 2.1 Implement an architecture for regression
- _preprocessor(): Preprocessor method
- \__init__(): Constructor method
- fit(): Model-training method
#### 2.2 Set up model evaluation
- predict(): Prediction method
- score(): Evaluation method
#### 2.3 Perform hyperparameter tuning
- RegressorHyperParameterSearch(): Perform a hyperparameter search

## IV. Final Report
* The final report of this coursework has been located in `\report` directory, it contains `intro_to_ML_cw2_report_final.pdf` and its LaTeX file compressed within zip.

