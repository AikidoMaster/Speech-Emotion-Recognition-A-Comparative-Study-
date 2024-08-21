Speech Emotion Recognition using RAVDESS Dataset
This project implements speech emotion recognition (SER) using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The system analyzes audio recordings to classify emotions expressed in speech.
Table of Contents
Introduction
Dataset
Feature Extraction
Model Architectures
Implementation
Results
Future Work
Installation
Usage
Contributing
License
Introduction
Speech Emotion Recognition (SER) is the process of automatically identifying human emotions from speech signals. This project explores various techniques for SER using machine learning and deep learning approaches.
Objectives
Implement and compare different SER models
Analyze the effectiveness of various audio features
Evaluate model performance on the RAVDESS dataset
Dataset
The RAVDESS dataset contains 7,356 audio-visual recordings of 24 professional actors (12 female, 12 male) vocalizing two lexically-matched statements. The dataset includes 8 emotions:
Neutral
Calm
Happy
Sad
Angry
Fearful
Disgust
Surprised
Feature Extraction
We extract the following features from the audio signals:
Mel Spectrograms
Mel-frequency Cepstral Coefficients (MFCCs)
Chroma Features
Spectral Features (Centroid, Bandwidth, Contrast, Flatness)
Model Architectures
The project implements and compares three main model architectures:
Multi-Layer Perceptron (MLP)
Long Short-Term Memory (LSTM)
Convolutional Neural Network (CNN)
Implementation
Key implementation details include:
Data preprocessing and augmentation
Feature extraction pipeline
Model training and evaluation
Hyperparameter tuning
Results
The project provides a comparative analysis of model performances, including:
Accuracy, F1-score, and other relevant metrics
Confusion matrices
ROC curves and AUC scores
Future Work
Potential areas for future enhancement include:
Exploring advanced architectures (e.g., Transformers)
Multi-modal emotion recognition
Real-time emotion detection
Cross-cultural emotion recognition
Installation
bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt

Usage
bash
python train_model.py --model cnn --features mfcc
python evaluate_model.py --model cnn --test_data path/to/test/data

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
