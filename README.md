# Park Rating Prediction with Deep Learning

## Overview

This project uses deep learning techniques to predict park ratings based on textual reviews. It's a practical application of natural language processing (NLP) and machine learning to extract insights from textual data.

## Prerequisites

Before running the code, ensure you have the necessary libraries and dependencies installed. You can install them using the `requirements.txt` file included in the project:

```bash
pip install -r requirements.txt
```
## Libraries Utilized

This project leverages various libraries, including:

- **TensorFlow and Keras** for deep learning.
- **Pandas and NumPy** for data manipulation.
- **Scikit-learn** for model evaluation and hyperparameter tuning.

## Dataset

The dataset used for this project can be downloaded from the following URL:

[Download Dataset](https://github.com/lorenzouttini/Exam-Deep-Learning/raw/main/parkReviews.zip)

The dataset is a collection of movie reviews, and you can customize the dataset as needed for your analysis.

## Data Preprocessing

The code includes data preprocessing steps to clean and prepare the dataset for model training. This includes:

- Loading the dataset from a CSV file.
- Sampling a subset of the data.
- Removing reviews with extreme lengths.
- Performing undersampling to balance the classes.

## Model Architecture

The deep learning model used for rating prediction is a recurrent neural network (RNN) with LSTM or GRU layers. You can find the model architecture defined in the code, along with various hyperparameters that can be customized.

## Training the Model

The model is trained on the preprocessed data, and you can adjust the training parameters as needed. The code includes options for grid search to find the best hyperparameters.

## Model Evaluation

The performance of the model is evaluated using metrics such as accuracy. The code includes options for model evaluation on both the training and test datasets.

## Usage

To use this project, follow these steps:

1. Download the dataset from the provided URL and unzip it.

2. Install the required libraries using the `requirements.txt` file.

3. Customize the data preprocessing and model architecture as needed for your specific dataset and use case.

4. Train and evaluate the model based on your dataset.

## License

This project is available under the MIT License. You are free to use and modify the code as per the terms of the license.

## Acknowledgments

This project was created by [Your Name] and can be found on GitHub at [Your GitHub Repository URL]. Feel free to reach out with any questions or feedback.

Enjoy exploring the world of movie rating prediction with deep learning!


The deep learning model used for rating prediction is a recurrent neural network (RNN) with LSTM or GRU layers. You can find the model architecture defined in the code, along with various hyperparameters that can be customized.
