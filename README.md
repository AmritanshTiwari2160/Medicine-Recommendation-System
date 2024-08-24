# 🩺 Medicine Recommendation System

Welcome to the **Medicine Recommendation System** project! This innovative system leverages machine learning to predict diseases based on user-input symptoms and provides a comprehensive set of recommendations, including medication, precautions, diet, and workouts. Perfect for aiding healthcare professionals or serving as an intelligent assistant for personal health management.

## 🚀 Features

- **Symptom-Based Disease Prediction**: Input your symptoms and get a real-time prediction of potential diseases.
- **Comprehensive Recommendations**: The system provides a detailed description of the disease, suggested medications, precautions, diet plans, and recommended workouts.
- **Multiple Machine Learning Models**: The system uses top-performing models like SVC, RandomForest, and GradientBoosting for accurate predictions.
- **Efficient Model Training**: Trained on a robust dataset, the models are optimized for high accuracy and reliability.
- **Scalable and Extendable**: Easy to extend with additional symptoms, diseases, and recommendation types.

## 🛠️ Technologies Used

- **Programming Language**: Python 🐍
- **Libraries**: Scikit-learn, Pandas, NumPy, Pickle
- **Models Used**: Support Vector Classifier (SVC), RandomForest, GradientBoosting, K-Nearest Neighbors, Naive Bayes
- **Dataset**: Custom CSV files containing disease symptoms, descriptions, medications, and more.

## 📂 Project Structure

- **Medical Recommendation.py**: The main Python script that handles the disease prediction and recommendation system.
- **README.md**: This file, containing an overview and instructions for the project.
- **Training.csv**: The dataset used for training the machine learning models.
- **description.csv**: Contains descriptions of the diseases predicted by the system.
- **diets.csv**: A dataset containing recommended diets for various diseases.
- **medications.csv**: A dataset with medication recommendations for each disease.
- **precautions_df.csv**: Data on precautions that should be taken for each disease.
- **svc.pkl**: The serialized model file used for making predictions.
- **symptoms_df.csv**: Contains data on symptoms associated with different diseases.
- **workout_df.csv**: A dataset that provides workout recommendations based on the predicted disease.

## 🧠 How It Works

1. **Data Preprocessing**: 
   - The dataset is loaded and preprocessed.
   - Symptoms are encoded as features, and diseases are used as labels.
  
2. **Model Training**: 
   - Various machine learning models are trained on the dataset.
   - The best-performing model is selected for predictions.
  
3. **Disease Prediction**: 
   - The model predicts the disease based on user-input symptoms.
  
4. **Recommendation System**: 
   - The system provides a description of the predicted disease.
   - It also suggests recommended medications, precautions, diets, and workouts.
