# 🚀 Quick Start Guide - AI Therapist

Get your AI therapist running in 5 minutes!

## 📋 Prerequisites

- Python 3.8+
- Git
- Internet connection

## ⚡ Super Quick Start (Free Cloud Deploy)

### Option 1: Streamlit Cloud (Recommended)

1. **Fork/Clone this repo to your GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Click "New app"**
5. **Select this repository**
6. **Set main file path: `app/main.py`**
7. **Click "Deploy"**

**That's it! Your AI therapist will be live in 2-3 minutes.**

### Option 2: Hugging Face Spaces

1. **Go to [huggingface.co/spaces](https://huggingface.co/spaces)**
2. **Create new Space with Streamlit SDK**
3. **Upload the files or connect Git repo**
4. **Space auto-deploys!**

## 💻 Local Development

### 1. Setup (One-time)

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-therapist

# Run setup (installs dependencies, creates directories)
python3 setup.py

# Or minimal setup only
python3 setup.py --minimal
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
# Method 1: Using Streamlit directly
streamlit run app/main.py

# Method 2: Using the demo script
python3 run_demo.py
```

## 🏋️ Custom Model Training (Optional)

### Quick Training on Free GPU

1. **Open `notebooks/fine_tune_therapist.ipynb` in Google Colab**
2. **Enable GPU runtime (Runtime → Change runtime type)**
3. **Run all cells**
4. **Model auto-uploads to Hugging Face Hub**

### Local Training

```bash
# Generate training data
python3 data/prepare_training_data.py

# Start training
python3 models/train.py
```

## 🔧 Configuration

### Environment Variables (Optional)

Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your API keys (optional)
```

### Config File

Edit `config.json` for customization:

```json
{
  "model": {
    "default_model": "microsoft/DialoGPT-medium",
    "temperature": 0.7
  },
  "rag": {
    "n_results": 3
  }
}
```

## 🌍 Free Deployment Options

| Platform | Cost | Setup Time | Performance |
|----------|------|------------|-------------|
| **Streamlit Cloud** | Free | 2 min | Good |
| **HF Spaces** | Free | 3 min | Good |
| **Railway** | Free tier | 5 min | Better |
| **Render** | Free tier | 5 min | Good |

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
export PYTHONPATH="${PYTHONPATH}:."
```

**2. Memory Issues**
- Use smaller model: `microsoft/DialoGPT-small`
- Enable quantization in config

**3. Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Quick Fixes

```bash
# Reset everything
python3 setup.py --skip-test

# Test specific components
python3 -c "from rag.rag_system import RAGTherapist; print('RAG OK')"
python3 -c "from models.inference import TherapistModel; print('Model OK')"
```

## 📱 Access Your App

### Local
- **Streamlit**: http://localhost:8501
- **Demo**: Terminal interface

### Cloud
- **Streamlit Cloud**: `https://yourapp.streamlit.app`
- **HF Spaces**: `https://huggingface.co/spaces/username/appname`

## 🔐 Security Notes

- App includes safety disclaimers
- No data is stored by default
- All processing can be done locally
- Use environment variables for API keys

## 📞 Need Help?

1. **Check the logs** in terminal/browser console
2. **Review documentation** in `deployment/deploy_guide.md`
3. **Run the demo** with `python3 run_demo.py`
4. **Create an issue** on GitHub

## 🎯 Next Steps

1. **Customize the therapy knowledge base** in `rag/rag_system.py`
2. **Add more training data** in `data/prepare_training_data.py`
3. **Fine-tune your model** using the notebooks
4. **Deploy to production** following security best practices

---

## ⚠️ Important Disclaimers

- **Educational purposes only**
- **Not a replacement for professional therapy**
- **Users should consult licensed professionals for mental health issues**
- **Include crisis resources and safety information**

---

**🎉 Congratulations! Your AI therapist is ready to help people in a supportive, ethical way.**