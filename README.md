# Sentiment Analysis using Naive Bayes and Logistic Regression

## Problem Definition
Performing text classification into five distinct classes using the Stanford Sentiment Treebank (SST) dataset. The task involves employing two different classification approaches: Naive Bayes and Logistic Regression. Additionally, there is a third part dedicated to evaluation, where we will generate a confusion matrix and derive performance metrics from it.

## Dataset Description
The dataset is sourced from the Stanford Sentiment Treebank (SST), a collection of movie reviews labelled with sentiment scores ranging from 0 to 1. We will preprocess the dataset by categorizing the sentiment scores into five distinct classes.
### Dataset Link : https://huggingface.co/datasets/sst

## Preprocessing
We will map the sentiment scores as follows:
- 0 to 0.2 (inclusive): "very negative"
- 0.2 to 0.4 (inclusive): "negative"
- 0.4 to 0.6 (inclusive): "neutral"
- 0.6 to 0.8 (inclusive): "positive"
- 0.8 to 1.0 (inclusive): "very positive"

## Part 1: Naive Bayes
### Algorithm Implementation
Implemented the Naive Bayes algorithm from scratch using NumPy only, following the algorithm outlined in the provided figure.
![image](https://github.com/elmahygurl/NLP_sst_Classification/assets/97133077/650a0523-4529-4930-8c3b-8531b63f9279)

### Comparison with scikit-learn
After implementing the algorithm, we reproduced the same results using scikit-learn's MultinomialNB classifier and compare the results.

## Part 2: Logistic Regression
### Feature Representation
Generated word bi-gram features for each sentence. Each sentence will be represented with a vector of length equal to the number of unique word bi-grams in the dataset.

### Algorithm Implementation
Logistic Regression is implemented from scratch using NumPy only, following the details from the readings in https://web.stanford.edu/~jurafsky/slp3/5.pdf .

### Comparison with scikit-learn
Compared our implementation with scikit-learn's LogisticRegression classifier, exploring the differences and determining which implementation is better suited for our task.

## Part 3: Confusion Matrix & Evaluation Metrics
### Confusion Matrix and Evaluation Metrics
Implemented a function to generate the confusion matrix given the predictions and the ground truth labels. We also computed precision, recall, and F1 score per class and macro-averaged using the generated confusion matrix.

### Comparison with scikit-learn
Finally, we compared our evaluation metrics with those obtained using scikit-learn's functions.

## Extra Notes
- Memory Optimization: We will take care of data types and memory usage to ensure efficient execution, especially in Colab where memory is limited.
- Stochasticity: Data will be shuffled every epoch to enforce the stochasticity of the SGD.
- Vectorized Calculations: Implementations is batched and vectorized for efficient calculations.

## Authors 
- [Nada Zayed](https://github.com/nadaz10)
- [Salma ElMahy](https://github.com/elmahygurl)
