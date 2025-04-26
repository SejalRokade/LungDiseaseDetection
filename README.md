
# Explainable AI Model for Lung Disease Detection using Deep Learning

**Kaggle Dataset:** [https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types]
---

**EfficientNet-V2-s** [https://www.kaggle.com/code/tycod253pranitsarode/lung-efficient-net-final]
---

**VIT** [https://www.kaggle.com/code/pranit9/lung-disease-new-vit]






---

## 📚 Project Description

Diagnosing lung diseases like **pneumonia**, **tuberculosis**, and **COVID-19** using chest X-rays can be highly challenging, especially in rural areas with limited radiology experts.  
This project leverages **Deep Learning** and **Explainable AI (XAI)** techniques to build an **interpretable** and **highly accurate** model to assist healthcare professionals in detecting lung diseases.

---

## 🎯 Objectives

- Compare multiple deep learning models for chest X-ray classification.
- Develop an optimized predictive model with high accuracy.
- Enhance transparency and interpretability using **XAI techniques** such as **Grad-CAM** and **LIME**.
- Deploy the model via a web interface for user-friendly access.

---

## 🏗️ Methodologies & Algorithms

- **Deep Learning Models:**
  - ResNet-50
  - DenseNet-121
  - VGG-16
  - Inception V3
  - EfficientNet (Ensemble)
  - Vision Transformers (ViT)

- **XAI Techniques:**
  - **LIME** (Local Interpretable Model-Agnostic Explanations)
  - **Grad-CAM** (Gradient-weighted Class Activation Mapping)

- **Preprocessing:**
  - Data Augmentation (Rotation, Flip, Color Jitter)
  - Normalization & Resizing

- **Hyperparameter Tuning:**
  - Automated search using **Optuna**
  - Cross-validation

- **Deployment:**
  - Web interface using **Streamlit**

- **Training Environment:**
  - Google Colab / Kaggle GPU
  - PyTorch Framework

---

## 🗂️ Dataset Structure

- **Classes:**
  - Bacterial Pneumonia
  - Coronavirus Disease
  - Normal
  - Tuberculosis
  - Viral Pneumonia

- **Total Images:** 10,095 Chest X-ray Images

---

## 📈 Results

| Model                | Test Accuracy |
|----------------------|---------------|
| EfficientNet-V2-S     | 94.79%         |
| Vision Transformers  | 92.00%         |

- **XAI Visualization:**  
  - **Grad-CAM** was used to highlight critical regions in X-rays.
  - **LIME** explained feature contributions for each prediction.

---

## ⚡ Advantages

- **Improved diagnostic accuracy** over manual interpretation.
- **Transparent decision-making** via XAI techniques.
- **Scalability** for large datasets.
- **Faster diagnosis** aiding clinical decision support.

---

## 🚀 Future Scope

- Integration with **hospital real-time databases**.
- Use **multi-modal data** (X-rays + patient history) for richer diagnosis.
- Develop a **mobile/web app** version for faster accessibility.
- Improve XAI techniques like **Grad-CAM++** for better clarity.

---

## 📜 References

- Literature Survey from latest IEEE, Elsevier, MDPI, and arXiv publications.
- Project published in:
  - **International Conference on Advances in Information Technology and Mobile Communication – AIM 2025**
  - **Tech Summit 2025 (Baby Conference)**

---

## ✨ Team Members

- **Sejal Rokade** (TYCOD245)
- **Pranav Sakpal** (TYCOD248)
- **Avanti Sangai** (TYCOD252)
- **Pranit Sarode** (TYCOD253)

Under the guidance of **Ms. Kavita Kolpe**,  
Department of Computer Engineering,  
PCET’s Pimpri Chinchwad College of Engineering.

---
