# 🌾 AI Crop Disease Detection System

An AI-powered web application that detects crop diseases using Deep Learning and provides treatment recommendations to help farmers take early preventive action.

---

## 📌 Project Overview

This project uses Transfer Learning with MobileNetV2 to classify crop leaf images into different disease categories. The system is designed to assist farmers in identifying plant diseases quickly and accurately.

The model was trained on a balanced dataset derived from the PlantVillage dataset.

---

## 🚀 Features

- 7-class crop disease classification
- Transfer Learning using MobileNetV2
- 95% validation accuracy
- Real-time image upload via Flask web app
- Automatic treatment recommendation
- Balanced dataset training
- Data augmentation for better generalization

---

## 🧠 Classes Covered

- Pepper__bell___Bacterial_spot
- Pepper__bell___healthy
- Potato___Early_blight
- Potato___healthy
- Potato___Late_blight
- Tomato_Bacterial_spot
- Tomato_Early_blight

---

## 🏗️ Tech Stack

- Python 3.12
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Flask
- NumPy
- Matplotlib
- PIL

---

## 📊 Model Performance

- Training Accuracy: ~94-95%
- Validation Accuracy: ~95-96%
- Balanced dataset (147 images per class)
- Confidence Range: 75% - 98%

---

## 🖼️ System Architecture

1. User uploads leaf image
2. Image is resized to 224x224
3. Model performs prediction using MobileNetV2
4. Disease class is identified
5. Treatment recommendation is displayed

---

## 🛠️ Installation Guide

git clone https://github.com/kashyapkakadiya/AI-Crop-Disease-Detection-System.git

cd AI-Crop-Disease-Detection-System


### Step 2: Install Dependencies

pip install tensorflow flask numpy matplotlib pillow


### Step 3: Run Application

python app.py

Open browser:

http://127.0.0.1:5000/

---

## 📂 Project Structure

AI-Crop-Disease-Detection-System/
│
├── dataset/
├── templates/
│ └── index.html
├── app.py
├── train_model.py
├── crop_disease_model.h5
├── class_names.json
├── accuracy_graph.png
└── README.md

---

## 🎯 Future Improvements

- Add more crop varieties
- Deploy on cloud platform
- Add fertilizer recommendation system
- Mobile app integration
- Real-time field detection using camera

---

## 👨‍💻 Author

**Kakadiya Kashyap**  
BE - Computer Engineering  
LDRP-ITR  

Email: kashyapkakadiya149@gmail.com

---

## 📜 License

This project is developed for academic and internship purposes.
### Step 1: Clone Repository

