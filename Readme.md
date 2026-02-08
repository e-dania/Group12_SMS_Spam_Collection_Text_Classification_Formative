# SMS Spam Classification: Comparative Analysis of Embeddings and Models

This repository contains the implementation and experimental evaluation for a **comparative study of text classification models using multiple word embedding techniques**, conducted as part of a group formative assignment.

The project focuses on **SMS spam detection** using both **traditional machine learning** and **sequence-based deep learning models**, evaluated across consistent embedding strategies.

---

## Project Objective

The objective of this project is to **systematically compare the performance of different model–embedding combinations** for text classification.  
Specifically, we investigate how various **word embeddings** interact with different **model architectures** and analyze their impact on classification performance.

---

## Dataset

- **Dataset:** SMS Spam Collection  
- **Source:** UCI Machine Learning Repository  
- **Task:** Binary classification (Spam vs Ham)

Each SMS message is labeled as:
- `ham` → non-spam  
- `spam` → spam  

The dataset is split into **training**, **validation**, and **test** sets to ensure fair evaluation and reproducibility.

---

## Preprocessing (Shared Across All Models)

All team members follow a **consistent preprocessing pipeline**, including:

- Lowercasing text
- Tokenization
- Removal of missing values
- Label encoding (`ham = 0`, `spam = 1`)
- Train / validation / test split

Embedding-specific preprocessing is applied where necessary (e.g., sequence padding for neural networks).

---

## Models Implemented

Each team member is responsible for one model architecture and evaluates it across multiple embeddings.

### Implemented Architectures

- **Traditional Machine Learning**
  - Logistic Regression (TF-IDF)

- **Sequence Models**
  - RNN (SimpleRNN)
  - LSTM
  - GRU

---

## Embedding Techniques

The following embeddings are used consistently across models for fair comparison:

- **TF-IDF**
- **Word2Vec Skip-gram**
- **Word2Vec CBOW**

Word2Vec embeddings are **trained only on the training split** to avoid data leakage.

---

## Run Model Training examples

python RNN/rnn_tfidf.py
python RNN/rnn_skipgram.py
python RNN/rnn_cbow.py

Each script:

Trains the model

Performs basic hyperparameter tuning

Saves the trained model (.h5)

Exports evaluation metrics to CSV

## Evaluation Metrics

All models are evaluated on the test set using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Metrics are stored in the results/ directory for easy aggregation and comparison.
