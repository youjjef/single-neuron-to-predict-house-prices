# Free Deployment Guide for AI Therapist

This guide covers multiple free deployment options for your AI therapist application.

## 🚀 Option 1: Streamlit Cloud (Recommended)

**Cost:** 100% Free
**Requirements:** GitHub repository

### Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial AI therapist commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app/main.py`
   - Click "Deploy"

3. **Environment Variables** (if needed)
   - Add any API keys in the Advanced settings
   - Set `HUGGING_FACE_TOKEN` if using private models

### Pros:
- Completely free
- Easy deployment
- Automatic updates from GitHub
- Built-in SSL
- Custom domain support

### Cons:
- Limited to 1GB RAM
- 1 CPU core
- Apps sleep after 7 days of inactivity

---

## 🤗 Option 2: Hugging Face Spaces

**Cost:** 100% Free
**Requirements:** Hugging Face account

### Steps:

1. **Create a Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Name your space (e.g., "ai-therapist")

2. **Upload Files**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-therapist
   cd ai-therapist
   
   # Copy your app files
   cp -r ../app/* .
   cp ../requirements.txt .
   
   # Create app.py (entry point)
   echo "import sys; sys.path.append('.'); from main import main; main()" > app.py
   
   git add .
   git commit -m "Add AI therapist app"
   git push
   ```

3. **Configure Space**
   - Add description and tags
   - Set visibility (public/private)
   - Space will auto-build and deploy

### Pros:
- Free hosting
- Integrated with HF ecosystem
- Easy model loading
- Built-in gradio option

### Cons:
- Less customization
- Tied to HF ecosystem

---

## 🚂 Option 3: Railway

**Cost:** $5/month after free trial
**Free Tier:** 500 execution hours/month

### Steps:

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Initialize Project**
   ```bash
   railway init
   railway add
   ```

3. **Deploy**
   ```bash
   railway up
   ```

4. **Environment Variables**
   ```bash
   railway variables set PYTHONPATH=/app
   ```

### Pros:
- Good free tier
- PostgreSQL included
- Easy scaling
- GitHub integration

### Cons:
- Limited free tier
- Requires credit card

---

## 🎨 Option 4: Render

**Cost:** Free tier available
**Limitations:** Apps sleep after 15 minutes of inactivity

### Steps:

1. **Create Web Service**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Choose "Web Service"

2. **Configuration**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0`

3. **Environment Variables**
   - Add any required API keys

### Pros:
- Free tier available
- Auto-deploy from GitHub
- Easy setup

### Cons:
- Apps sleep after inactivity
- Slower cold starts

---

## 📱 Option 5: Replit

**Cost:** Free tier available
**Good for:** Quick testing and demos

### Steps:

1. **Create Repl**
   - Go to [replit.com](https://replit.com)
   - Import from GitHub or create new Python repl

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run**
   ```bash
   streamlit run app/main.py
   ```

### Pros:
- Very easy setup
- In-browser development
- Good for prototypes

### Cons:
- Performance limitations
- Not ideal for production

---

## 🔧 Optimization for Free Hosting

### Memory Optimization

1. **Use Quantized Models**
   ```python
   # In models/inference.py
   use_quantization = True  # Enable 4-bit quantization
   ```

2. **Lazy Loading**
   ```python
   # Load model only when needed
   @st.cache_resource
   def load_model():
       return TherapistModel()
   ```

3. **Reduce Model Size**
   - Use smaller base models (DialoGPT-small instead of medium)
   - Use distilled models
   - Consider model pruning

### Performance Tips

1. **Caching**
   ```python
   @st.cache_data
   def expensive_function():
       # Cache expensive operations
       pass
   ```

2. **Async Loading**
   ```python
   # Load components asynchronously
   with st.spinner("Loading..."):
       model = load_model()
   ```

3. **Optimize Dependencies**
   ```txt
   # requirements.txt - use minimal versions
   streamlit==1.28.0
   torch==2.0.0
   transformers==4.35.0
   # Remove unused packages
   ```

---

## 🔒 Security Considerations

### For Free Hosting:

1. **Environment Variables**
   - Never commit API keys
   - Use platform-specific environment variable systems

2. **Rate Limiting**
   ```python
   # Simple rate limiting
   if st.session_state.get('request_count', 0) > 100:
       st.error("Rate limit exceeded")
   ```

3. **Input Validation**
   ```python
   def validate_input(text):
       if len(text) > 1000:
           return False
       return True
   ```

---

## 📊 Monitoring (Free Options)

1. **Streamlit Analytics**
   - Built-in visitor analytics
   - Performance metrics

2. **Google Analytics**
   ```python
   # Add GA tracking code to your app
   st.markdown("""
   <!-- Global site tag (gtag.js) - Google Analytics -->
   <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
   """, unsafe_allow_html=True)
   ```

3. **Simple Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   
   def log_interaction(user_input, response):
       logging.info(f"User: {user_input[:50]}... | Response: {response[:50]}...")
   ```

---

## 🚀 Quick Deploy Commands

### Streamlit Cloud
```bash
# Push to GitHub
git add . && git commit -m "Deploy" && git push

# Then deploy via web interface
```

### Hugging Face Spaces
```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
cd SPACE_NAME

# Add files and push
cp -r ../app/* .
git add . && git commit -m "Deploy app" && git push
```

### Railway
```bash
railway login
railway init
railway up
```

---

## 📞 Support and Community

- **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Hugging Face Discord**: [hf.co/join/discord](https://hf.co/join/discord)
- **Railway Discord**: [railway.app/discord](https://railway.app/discord)

---

## 🎯 Recommended Free Stack

For the best free experience:

1. **Model Hosting**: Hugging Face Hub
2. **App Hosting**: Streamlit Cloud
3. **Code Hosting**: GitHub
4. **Analytics**: Google Analytics
5. **Monitoring**: Streamlit built-in analytics

This combination provides a completely free, scalable solution for your AI therapist!