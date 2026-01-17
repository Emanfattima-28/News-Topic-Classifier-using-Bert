# News-Topic-Classifier-using-Bert

## Problem Statement
With the massive growth of online news platforms, organizing and categorizing news articles automatically has become essential. Manual classification is time-consuming and inefficient.  
The goal of this project is to build a **News Topic Classification system** using a **pre-trained BERT (Bidirectional Encoder Representations from Transformers)** model to accurately classify news headlines into predefined topic categories.

---

## Objective
- Fine-tune a transformer-based model (`bert-base-uncased`) on a news dataset.
- Classify news headlines into meaningful categories.
- Evaluate model performance using accuracy and F1-score.
- Deploy the trained model using **Streamlit** for real-time prediction.

---

## Dataset Description

- **Dataset Name:** AG News Dataset
- **Source:** Hugging Face Datasets (`sh0416/ag_news`)
- **Type:** Text-based dataset
- **Splits:**
  - Training Set
  - Test Set
- **Classes (4):**
  - World
  - Sports
  - Business
  - Sci/Tech

### Dataset Fields:
- `title` – News headline
- `description` – Short description of the news
- `label` – Category index (0–3)

---

## Tools & Libraries Used

- **Python**
- **Hugging Face Datasets**
- **Transformers (BERT)**
- **PyTorch**
- **NumPy**
- **Scikit-learn** – Evaluation metrics
- **Streamlit** – Web application deployment

---

## Project Workflow & Explanation

### 1️. Dataset Loading
- Dataset loaded using Hugging Face `load_dataset()` API.
- Training and testing splits accessed separately.
- Labels normalized to ensure four valid classes.

---

### 2️. Text Preprocessing & Tokenization
- Combined `title` and `description` into a single text input.
- Tokenization performed using `BertTokenizer`.
- Applied:
  - Padding to fixed length
  - Truncation to `max_length = 128`
- Converted dataset into PyTorch tensor format.

---

### 3️. Model Selection
- Pre-trained **BERT Base Uncased** model selected.
- Used `BertForSequenceClassification` with:
  - `num_labels = 4`
- Leveraged transfer learning for better accuracy with limited training epochs.

---

### 4️. Model Training
- Training handled using Hugging Face `Trainer` API.
- **Training configuration:**
  - Learning rate: `2e-5`
  - Batch size: `16`
  - Epochs: `3`
  - Weight decay: `0.01`
- Training logs stored for monitoring.

---

### 5️. Model Evaluation
- Evaluated on test dataset using:
  - **Accuracy**
  - **Weighted F1-score**
- Custom evaluation function implemented using Scikit-learn metrics.

---

### 6️. Model Saving
---

### 7️. Deployment using Streamlit
- Built a simple web interface for live classification.
- User enters a news headline.
- Model predicts and displays the corresponding topic instantly.

**Predicted Categories:**
- World
- Sports
- Business
- Sci/Tech

---

## Evaluation Metrics

- **Accuracy:** Measures overall classification correctness.
- **F1-score:** Balances precision and recall across all classes.

The fine-tuned BERT model achieves strong performance due to contextual understanding of text.

---

## Key Insights
- Transformer-based models significantly outperform traditional ML models for text classification.
- Combining title and description improves contextual understanding.
- Fine-tuning pre-trained BERT requires fewer epochs while achieving high accuracy.
- Streamlit enables quick and effective deployment of NLP models.

---

## Conclusion
This project demonstrates the effectiveness of **BERT-based transformer models** for multi-class text classification tasks. By leveraging transfer learning, the model accurately classifies news topics with minimal preprocessing and training time. The Streamlit deployment makes the solution interactive, user-friendly, and production-ready.

---

## How to Run the Project

1. Install dependencies:
```bash
pip install transformers datasets torch streamlit scikit-learn

- Fine-tuned BERT model saved locally.
- Tokenizer saved along with the model for reuse.
- Enables easy deployment and inference.

**Saved Directory:**
