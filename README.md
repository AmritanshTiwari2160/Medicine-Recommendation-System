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

```bash
medicine-recommendation-system/
├── datasets/
│   ├── Trainingcsv/Training.csv          # Training data with symptoms and disease labels
│   ├── symtoms_df.csv                    # Symptoms and their encoded values
│   ├── precautions_df.csv                # Precautions related to each disease
│   ├── workout_df.csv                    # Workouts recommended for each disease
│   ├── description.csv                   # Disease descriptions
│   ├── medications.csv                   # Medication recommendations
│   └── diets.csv                         # Diet plans for each disease
├── models/
│   └── svc.pkl                            # Serialized trained SVC model
├── main.py                               # Main script for training models and making predictions
└── README.md                             # Project documentation

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
