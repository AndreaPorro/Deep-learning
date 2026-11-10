# 🍕 Food Image Classification: Transfer Learning on a Small, Imbalanced Dataset

This repository contains a group project developed for the **Foundations of Deep Learning** course as part of the MSc in Data Science at the **University of Milan-Bicocca (UNIMIB)**.

The project tackles a **9-class food image classification** problem with fewer than 1,000 images and a strong class imbalance, and asks whether transfer learning from ImageNet-pretrained networks can compensate for the lack of data. It covers the full pipeline: data cleaning with leakage prevention, a CNN baseline, three transfer learning backbones, classical ML on deep features, Explainable AI and a live **Gradio** demo.

📓 **Notebook:** [`Project_Cesaratto_Porro.ipynb`](./Project_Cesaratto_Porro.ipynb)
📊 **Slides:** [`FDL_project_Cesaratto_Porro.pdf`](./Result presentation FDL.pptx)
🚀 **Live demo:** [Hugging Face Space](https://huggingface.co/spaces/IlariaCesa/food_FDL)

---

## 🎯 Project Objectives & Research Questions

The central question is: **can transfer learning from ImageNet-pretrained networks compensate for a small, imbalanced training set, and which adaptation strategy works best?** It is broken down into three comparisons:

* **Transfer learning vs. training from scratch:** does a pretrained backbone clearly outperform a CNN trained on the same limited data?
* **Feature extraction vs. fine-tuning:** does unfreezing layers add value, or does it only increase the risk of overfitting?
* **End-to-end vs. hybrid:** can classical ML classifiers (SVM, Random Forest, XGBoost) on frozen deep features compete with neural networks?

---

## 🛠️ Tech Stack & Workflow

* **Python & Google Colab:** Development environment, with checkpoints persisted on Google Drive.
* **TensorFlow / Keras:** CNN from scratch and the three pretrained backbones (EfficientNetV2S, ResNet50, MobileNetV3Large).
* **scikit-learn & XGBoost:** PCA, t-SNE, and SVM, Random Forest and XGBoost classifiers with hyperparameter search.
* **imagehash:** Perceptual hashing to detect near-duplicate images.
* **Grad-CAM & LIME:** Explainability on correct and incorrect predictions.
* **Gradio & Hugging Face Spaces:** Interactive demo with live Grad-CAM overlays.

---

## 🔬 Methodology

* **Dataset:** 978 images across 9 classes (pizza, hamburger, french fries, sushi, sashimi, hot dog and three salad types) from a public MathWorks example repository, from 15 to 299 images per class (about 20 to 1).
* **Cleaning:** 35 problematic images removed, 4 relabeled and 46 exact duplicates deleted, for **897 final images**. Near-duplicates (29 groups) are not deleted but kept together in the same split.
* **Leakage-free split:** 70/15/15, stratified by class and group-aware, so a near-identical image never appears in both training and test.
* **Preprocessing:** 224×224 resize, a backbone-specific normalisation and conservative augmentation (flip, ±10% rotation/zoom, ±15% brightness/contrast), with no hue changes because colour matters for food.
* **Imbalance:** class weights inversely proportional to class frequency, capped at 5.
* **Models:** 10 in total, compared under identical conditions: a CNN from scratch, three backbones each in two phases (feature extraction and fine-tuning), and three classical classifiers on EfficientNetV2S deep features.
* **Selection:** by **macro F1** on the validation set, since accuracy alone would be dominated by the majority classes. The test set is used once, at the end.

---

## 📈 Key Insights

* **Transfer learning beats training from scratch:** the CNN from scratch reaches only **39%** validation accuracy, against **92–98%** for the transfer learning models.
* **The winner is EfficientNetV2S with feature extraction:** **97.66%** validation accuracy and macro F1 of 0.946.
* **On the held-out test set:** **96.27%** accuracy, **98.51%** top-3 accuracy and 0.960 weighted F1.
* **Fine-tuning did not improve accuracy on any backbone:** EfficientNetV2S drops from 97.66% to 96.88%. With only 633 training images, unfreezing layers adds overfitting risk, and classes like pizza, hot dog and hamburger already exist in ImageNet.
* **Classical ML is competitive:** SVM, Random Forest and XGBoost on frozen features exceed 90% validation accuracy. t-SNE shows that the frozen features are already class-structured.
* **Macro F1 drops on the test set (0.883)** because of tiny minority classes: only 1 of the 3 Greek salad images is classified correctly.
* **Errors are not random:** Grad-CAM and LIME show that a Greek salad misclassified as caprese salad is driven by visually plausible cues.

---

## 📂 Repository Structure

* `Project_Cesaratto_Porro.ipynb` — Complete notebook: loading, EDA, cleaning, split, preprocessing, models, comparison, test evaluation, XAI and Gradio interface
* `FDL_project_Cesaratto_Porro.pdf` — Presentation slides

---

## ⚠️ Limitations

* **Small dataset:** minority-class metrics are noisy and unstable.
* **Compute constraints:** Colab prevented k-fold cross-validation, a deeper CNN and formal significance tests such as McNemar's test.
* **Manual hyperparameters:** neural networks were tuned by hand, while classical ML models used grid and random search.

---

## 👥 Contributors

- [@ilaria-cesa](https://github.com/ilaria-cesa)

---
*Developed by Andrea Porro and Ilaria Cesaratto.*
