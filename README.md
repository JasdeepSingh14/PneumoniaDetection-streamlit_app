# 🫁 Pneumonia Detection from Chest X-rays Using Deep Learning

An AI-powered Chest X-ray Screening Application built using **Python, PyTorch, Convolutional Neural Networks (CNN), Grad-CAM and Streamlit**.

The application analyzes chest X-ray images and classifies them into two categories: **Normal** and **Pneumonia**. It also provides visual explanations, severity estimation, DICOM image support, patient information management and downloadable PDF reports.

<image_placeholder></image_placeholder>

## 🚀 Live Demo

⭐ If you find this project interesting, consider giving the repository a star!
**[Try the Pneumonia Detection Application](https://pneumonia-xrayscreening.streamlit.app/)**
---

## 📌 Project Overview

Pneumonia is a respiratory infection that affects the lungs and can cause serious complications if not identified and treated appropriately.

This project demonstrates the application of Deep Learning and Computer Vision in medical image classification.

A custom-built Convolutional Neural Network (CNN), implemented using PyTorch, is trained on chest X-ray images to identify patterns associated with pneumonia.

The application goes beyond basic image classification by integrating Grad-CAM explainability, heatmap-based severity estimation, DICOM processing and automated PDF report generation.

## 📸 Application Screenshots

### 1. Home / Image Upload

<img width="1365" height="605" alt="image" src="https://github.com/user-attachments/assets/ebc41231-3fee-4ac7-a0c1-e0deb9998152" />

### 2. Prediction Results

<img width="1366" height="607" alt="image" src="https://github.com/user-attachments/assets/e48cad98-3206-4ddf-b8c2-08f059c32a74" />

### 3. Grad-CAM Visualization

<img width="1459" height="540" alt="merged-image-2026-11-12T07-43-35" src="https://github.com/user-attachments/assets/caae79e7-51a4-4c9e-9fb8-db7aad932b83" />

### 4. Dashboard

<img width="2037" height="541" alt="merged-image-2026-11-12T07-49-58" src="https://github.com/user-attachments/assets/d5a45e61-38e7-4f45-a898-ada9627e1046" />

## ✨ Key Features

### 1. AI-Based Pneumonia Classification
- Custom CNN architecture developed using PyTorch.
- Binary classification into Normal and Pneumonia.
- Displays prediction probabilities for both classes.
- Optimized for CPU-based inference.

### 2. Grad-CAM Explainability
- Generates heatmaps showing image regions influencing model predictions.
- Displays original X-ray, heatmap and overlay visualizations.
- Adjustable heatmap intensity.
- Multiple heatmap colour schemes.
- Helps interpret the model's classification behaviour.

### 3. Multiple Image Upload
- Supports uploading multiple chest X-ray images.
- Individual prediction and visualization for each image.
- Batch prediction summary for multiple uploaded images.
- Includes sample X-ray images for demonstration.

### 4. DICOM Image Support
- Supports JPEG, PNG and DICOM (`.dcm`, `.dicom`) formats.
- Processes DICOM pixel data using Pydicom.
- Handles DICOM windowing and MONOCHROME1 inversion.
- Extracts relevant metadata for display and PDF reporting.

### 5. Heatmap-Based Severity Estimation
- Estimates the percentage of image area highlighted by Grad-CAM.
- Categorizes the estimated highlighted area into:
  - Mild
  - Moderate
  - Severe
- Displays affected regions across five image sections:
  - Upper Left
  - Upper Right
  - Central
  - Lower Left
  - Lower Right

*Note: Severity estimation is a heatmap-based heuristic, not a clinically validated severity assessment.*

### 6. Patient Information Management
- Patient name
- Patient ID
- Age
- Sex
- Referring physician

Patient information can be included in generated PDF reports.

### 7. Automated PDF Report Generation
- Generates downloadable one-page PDF reports.
- Includes patient information and image details.
- Displays prediction and confidence probabilities.
- Includes original X-ray and Grad-CAM overlay.
- Includes severity estimation and DICOM metadata when applicable.

### 8. Interactive Dashboard
- Total number of analyzed images.
- Normal and Pneumonia prediction counts.
- Average prediction confidence.
- Prediction distribution charts.
- Confidence comparison per image.
- Severity distribution.
- Session-based prediction timeline.

### 9. Session History
- Maintains prediction history during the current session.
- Displays image names, predictions and probabilities.
- Provides CSV export functionality.
- Allows clearing session history.

---

## 🧠 CNN Model Architecture

The project uses a custom-built Convolutional Neural Network implemented using PyTorch.

The architecture consists of:

- **9 Convolutional Blocks:** Extract important visual features from chest X-ray images.
- **ReLU Activation:** Introduces non-linearity.
- **Batch Normalization:** Stabilizes the learning process.
- **3 Max-Pooling Stages:** Reduce spatial dimensions.
- **Global Average Pooling:** Aggregates extracted feature maps.
- **Final 4 × 4 Convolution:** Produces two classification scores.

The model accepts RGB images of size 224 × 224 and produces two output logits corresponding to Normal and Pneumonia.

Softmax is applied during inference to calculate the class probabilities.

## 🔄 Project Workflow

1. **Image Upload:** Upload one or multiple chest X-ray images in JPEG, PNG or DICOM format.
2. **Image Loading:** Read the uploaded image and convert it into a compatible RGB representation.
3. **Preprocessing:** Resize and center-crop the image to 224 × 224 pixels.
4. **Normalization:** Apply ImageNet mean and standard deviation normalization.
5. **Model Inference:** Pass the processed image through the trained CNN.
6. **Classification:** Generate Normal and Pneumonia prediction scores.
7. **Probability Calculation:** Apply Softmax to obtain classification probabilities.
8. **Explainability:** Generate Grad-CAM heatmaps using gradients from the last convolution block.
9. **Severity Estimation:** Estimate the highlighted image area for Pneumonia predictions.
10. **Result Visualization:** Display predictions, probabilities and heatmap visualizations.
11. **Report Generation:** Generate downloadable PDF reports and maintain session-based analysis history.

---

## 📊 Dataset and Model Performance

The model was trained and evaluated using a dataset containing **5,856 chest X-ray images** belonging to two categories: Normal and Pneumonia.

### Dataset Distribution

| Dataset | Number of Images |
|---|---:|
| Training | 3,982 |
| Validation | 702 |
| Testing | 1,172 |
| **Total** | **5,856** |

The validation set was held out from the training folder. The test set was kept separate for final evaluation.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Deep Learning Framework | PyTorch |
| Model Architecture | Custom CNN |
| Input Image Size | 224 × 224 |
| Loss Function | Weighted Cross-Entropy Loss |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Momentum | 0.9 |
| Batch Size | 32 |
| Training Epochs | 12 |
| Learning Rate Scheduler | StepLR |
| Data Augmentation | Colour changes, horizontal flips and rotations |

Class weights were used to address class imbalance in the training dataset. The best model checkpoint was selected using validation balanced accuracy.

### Test Results

<text_placeholder></text_placeholder>

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 95.14% |
| Balanced Accuracy | 95.28% |
| Normal Recall | 95.6% |
| Pneumonia Recall | 95.0% |

### Confusion Matrix

| Actual / Predicted | Normal | Pneumonia |
|---|---:|---:|
| Normal | 303 | 14 |
| Pneumonia | 43 | 812 |

The model correctly classified 1,115 out of 1,172 test images.

**Important:** 43 pneumonia cases were classified as Normal in the test set. Therefore, a Normal prediction must never be interpreted as proof that pneumonia is absent.

---

## 🔥 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to understand which image regions influence the model's prediction.

The implementation:
1. Captures feature activations from the last convolution block.
2. Calculates gradients of the predicted class score.
3. Uses gradient-based channel weights.
4. Combines the feature maps to generate a heatmap.
5. Resizes the heatmap and overlays it on the original X-ray.

The application provides:
- Original image.
- Grad-CAM heatmap.
- Heatmap overlay.
- Adjustable visualization settings.

Bright regions indicate areas that contributed more strongly to the model's prediction. They do not establish the presence or precise location of disease.

## 🩺 Severity Estimation

The application uses the Grad-CAM heatmap to estimate the percentage of image area highlighted.

| Highlighted Area | Estimated Level |
|---|---|
| Below 20% | Mild |
| 20%–40% | Moderate |
| 40% or more | Severe |

The highlighted area is also analyzed across five image regions.

This is a simple visualization-based heuristic, not a clinical severity grade or measurement of pneumonia progression.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep Learning and model inference |
| Torchvision | Image transformations and preprocessing |
| CNN | Chest X-ray classification |
| Grad-CAM | Model explainability |
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Pillow (PIL) | Image processing |
| Pydicom | DICOM image handling |
| ReportLab | PDF report generation |
| Streamlit | Interactive web application |

## 📂 Project Structure

```text
PneumoniaDetection-streamlit_app/
│
├── .streamlit/
│   └── config.toml
│
├── sample_images/
│   └── Sample chest X-ray images
│
├── arch.py
│   └── Custom CNN architecture
│
├── streamlit_app.py
│   └── Main Streamlit application
│
├── severity.py
│   └── Grad-CAM-based severity estimation
│
├── dicom_handler.py
│   └── DICOM image processing and metadata extraction
│
├── report_generator.py
│   └── PDF report generation
│
├── xray_model.pth
│   └── Trained PyTorch model weights
│
├── requirements.txt
│   └── Project dependencies
│
└── README.md
    └── Project documentation
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

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

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

The application will be available at:

```text
http://localhost:8501
```

---

## ⚠️ Limitations and Disclaimer

- This project is developed for educational, demonstration and research purposes.
- It is not a certified medical device and must not be used to diagnose or rule out pneumonia.
- The model performs binary classification and does not detect other lung conditions.
- The model has been trained and tested on a single dataset and has not been externally validated across hospitals, scanners or patient populations.
- Grad-CAM visualizations represent model attention and are not clinical evidence.
- Severity estimation is based on highlighted heatmap area and is not clinically validated.
- A Normal prediction does not rule out pneumonia.
- Every X-ray should be reviewed by a qualified radiologist or healthcare professional.
- Avoid uploading identifiable patient information to a public deployment.

## 🚀 Future Enhancements

- Explore transfer learning using ResNet and DenseNet.
- Improve model generalization using larger and more diverse datasets.
- Introduce multi-class classification for additional chest conditions.
- Improve explainability and visualization techniques.
- Perform external validation using independent datasets.
- Enhance reporting and patient analysis capabilities.

---

## 👨‍💻 Author

**Jasdeep Singh**

- GitHub: [@JasdeepSingh14](https://github.com/JasdeepSingh14)

---

⭐ If you find this project useful, consider giving the repository a star!



