# Streamlit Deployment Guide

## Local Testing

To test the app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Deploy to Streamlit Cloud (Free)

### Prerequisites

1. GitHub account
2. Your code pushed to a GitHub repository

### Steps:

1. **Push your code to GitHub**

   ```bash
   git init
   git add .
   git commit -m "Add Streamlit app"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Sign up/Login to Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy!"

4. **Your app will be live at**: `https://[your-app-name].streamlit.app`

### Important Files for Deployment

- `streamlit_app.py` - Main application file
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `app/model.joblib` - Trained model (must be in repo)

## Alternative: Deploy to Other Platforms

### Heroku

1. Create `Procfile`:

   ```
   web: sh setup.sh && streamlit run streamlit_app.py
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

### Docker (for any cloud provider)

Use the existing Dockerfile or modify it to run Streamlit:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Model not found

Make sure `app/model.joblib` exists. If not, run:

```bash
python train_model.py
```

### Port issues

If port 8501 is busy, specify a different port:

```bash
streamlit run streamlit_app.py --server.port=8502
```

### Memory issues on Streamlit Cloud

Streamlit Cloud has 1GB RAM limit. If your model is large, consider:

- Using model compression
- Upgrading to Streamlit Cloud paid plan
- Deploying to another platform

## Security Notes

- Don't commit sensitive data or API keys
- Use Streamlit secrets for sensitive configuration
- The model is publicly accessible once deployed
