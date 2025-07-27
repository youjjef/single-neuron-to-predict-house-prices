# AI Therapist - Free LLM Fine-tuning & RAG Implementation

A complete AI therapist application using LLM fine-tuning and Retrieval Augmented Generation (RAG), designed for free deployment.

## 🚀 Features

- **Fine-tuned LLM**: Custom therapy model using free resources (Colab, Kaggle)
- **RAG Implementation**: Knowledge base with therapeutic techniques and resources
- **Free Deployment**: Streamlit Cloud + Hugging Face Hub
- **Privacy-Focused**: Local processing options and secure data handling
- **Multi-Modal**: Text-based therapy sessions with emotional analysis

## 🛠️ Tech Stack

- **Model**: Mistral-7B-Instruct (fine-tuned on therapy data)
- **RAG**: ChromaDB + Sentence Transformers
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud (Free)
- **Model Storage**: Hugging Face Hub (Free)
- **Training**: Google Colab / Kaggle (Free GPUs)

## 📁 Project Structure

```
ai-therapist/
├── app/                    # Streamlit web application
├── data/                   # Training and RAG data
├── models/                 # Model training scripts
├── rag/                    # RAG implementation
├── notebooks/              # Training notebooks
├── deployment/             # Deployment configs
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `streamlit run app/main.py`
4. For training: Use notebooks in Google Colab

## 🔧 Free Resources Used

- Google Colab Pro (Free tier available)
- Hugging Face Hub (Free model hosting)
- Streamlit Cloud (Free web app hosting)
- ChromaDB (Open source vector database)
- Kaggle Notebooks (Free GPU access)

## ⚠️ Disclaimer

This is an educational project. For actual mental health support, please consult licensed professionals.
