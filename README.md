# 🫁 Pneumonia Detection Using Deep Learning

An AI-powered Chest X-ray Screening Application that uses a **Convolutional Neural Network (CNN)** built with PyTorch to classify chest X-ray images into two categories: **Normal** and **Pneumonia**.

The project provides an interactive web interface built using Streamlit, allowing users to upload chest X-ray images and view model predictions along with visual explanations using **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

## 🚀 Live Demo

⭐ If you find this project interesting, consider giving the repository a star!
**[Try the Pneumonia Detection Application](https://pneumonia-xrayscreening.streamlit.app/)**

## 📌 Project Overview

Pneumonia is a respiratory infection that can affect one or both lungs. Early identification can help support timely medical evaluation.

This project demonstrates how Deep Learning and Computer Vision can be applied to medical image classification.

The application analyzes uploaded chest X-ray images and predicts whether the image belongs to:

- **Normal:** No pneumonia detected by the model.
- **Pneumonia:** The model identifies patterns associated with pneumonia.

Along with the prediction, the application displays class probability scores and a Grad-CAM heatmap to help visualize the image regions influencing the model's prediction.

## ✨ Key Features

- **CNN-Based Classification:** Uses a custom Convolutional Neural Network implemented in PyTorch.
- **Two-Class Prediction:** Classifies chest X-ray images as Normal or Pneumonia.
- **Image Upload:** Allows users to upload chest X-ray images for analysis.
- **Prediction Confidence:** Displays probability scores for both classification categories.
- **Grad-CAM Visualization:** Generates heatmaps highlighting image regions that contribute to the model's prediction.
- **Interactive Dashboard:** Provides a user-friendly interface developed using Streamlit.
- **Model Caching:** Uses Streamlit caching to avoid repeatedly loading the trained model.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep Learning framework |
| Torchvision | Image preprocessing and transformations |
| CNN | Image classification architecture |
| NumPy | Numerical computations |
| Pandas | Data handling |
| Pillow (PIL) | Image processing |
| Streamlit | Interactive web application |
| Grad-CAM | Model prediction visualization |

## 🧠 Model Architecture

The project uses a custom-built Convolutional Neural Network (CNN) for binary image classification.

The architecture consists of:

- Convolutional Layers for extracting image features.
- ReLU Activation Functions for introducing non-linearity.
- Batch Normalization for stabilizing training.
- Max Pooling Layers for spatial dimension reduction.
- Global Average Pooling for feature aggregation.
- Final Classification Layer for generating two class scores.

The model produces two output logits corresponding to the Normal and Pneumonia classes. Softmax is applied during inference to obtain class probabilities.

## 🔍 Image Processing Workflow

1. The user uploads a chest X-ray image.
2. The image is loaded and converted into RGB format.
3. Image resizing and center cropping are performed to obtain a 224 × 224 image.
4. The image is converted into a PyTorch tensor.
5. Normalization is applied using ImageNet mean and standard deviation.
6. The pre-trained CNN model processes the image.
7. The predicted class and probability scores are calculated.
8. Grad-CAM generates a heatmap to visualize the regions influencing the prediction.
9. The results are displayed through the Streamlit interface.

## 📂 Project Structure

```text
PneumoniaDetection-streamlit_app/
│
├── .streamlit/
│   └── config.toml
│
├── arch.py                  # Custom CNN architecture
├── streamlit_app.py         # Streamlit application
├── xray_model.pth           # Trained PyTorch model weights
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/JasdeepSingh14/PneumoniaDetection-streamlit_app.git
```

### 2. Navigate to the Project Directory

```bash
cd PneumoniaDetection-streamlit_app
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default browser.

## 📊 Output and Visualization

The application provides:

- Predicted classification (Normal or Pneumonia).
- Probability distribution for both classes.
- Grad-CAM heatmap visualization.
- Visual representation of the image regions influencing the prediction.

## 🔮 Future Enhancements

- Improve model performance through additional training and validation.
- Evaluate the model using accuracy, precision, recall, F1-score and ROC-AUC.
- Explore transfer learning using architectures such as ResNet and DenseNet.
- Add support for additional chest conditions.
- Improve model explainability and visualization.
- Introduce a more comprehensive medical image analysis dashboard.

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The predictions generated by this application are not medical diagnoses and should not be used as a substitute for professional medical advice, clinical evaluation, or radiologist interpretation.

Always consult a qualified healthcare professional for medical assessment.

## 👨‍💻 Author

**Jasdeep Singh**

- GitHub: [@JasdeepSingh14](https://github.com/JasdeepSingh14)

---
