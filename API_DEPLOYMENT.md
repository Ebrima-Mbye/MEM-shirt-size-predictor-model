# API Deployment Guide for my-exam-mate Integration

This guide shows how to deploy the FastAPI server so your my-exam-mate project can make API requests over the internet.

## Why Not Streamlit?

Streamlit is for **interactive web UIs**, not API endpoints. Your [app/server.py](app/server.py) FastAPI server is what you need for API requests.

## Current API Endpoints

Your FastAPI server provides:
- `GET /` - API info
- `GET /health` - Health check
- `GET /model-info` - Model metadata
- `POST /predict` - **Main prediction endpoint** (what my-exam-mate needs)

### Example Request (from your client.py):

```python
import requests

url = "https://your-api.onrender.com/predict"  # Will be your deployed URL

payload = {
    "height_cm": 175,
    "weight_kg": 75,
    "age": 24,
    "gender": "male",
    "fit_preference": "regular",
    "build": "average"
}

response = requests.post(url, json=payload)
print(response.json())
```

### Example Response:

```json
{
  "recommended_size": "M",
  "confidence": 0.62,
  "probabilities": {
    "S": 0.1,
    "M": 0.62,
    "L": 0.22,
    "XL": 0.05,
    "XXL": 0.01
  },
  "model_version": "shirt-size-v1"
}
```

## Option 1: Deploy to Render (Recommended - Easy & Free)

### Steps:

1. **Go to [render.com](https://render.com)** and sign up with GitHub

2. **Click "New +" → "Web Service"**

3. **Connect your GitHub repository**: `Ebrima-Mbye/MEM-shirt-size-predictor-model`

4. **Configure the service**:
   - **Name**: `mem-shirt-size-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`

5. **Click "Create Web Service"**

6. **Wait for deployment** (5-10 minutes)

7. **Your API URL will be**: `https://mem-shirt-size-api.onrender.com`

### Test Your Deployed API:

```bash
curl -X POST "https://mem-shirt-size-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "height_cm": 175,
    "weight_kg": 75,
    "age": 24,
    "gender": "male",
    "fit_preference": "regular",
    "build": "average"
  }'
```

## Option 2: Deploy to Railway.app (Also Free)

### Steps:

1. **Go to [railway.app](https://railway.app)** and sign up with GitHub

2. **Click "New Project" → "Deploy from GitHub repo"**

3. **Select**: `Ebrima-Mbye/MEM-shirt-size-predictor-model`

4. **Railway auto-detects Python**

5. **Add Start Command** in Settings:
   ```
   uvicorn app.server:app --host 0.0.0.0 --port $PORT
   ```

6. **Generate Domain** in Settings → Networking

7. **Your API URL will be**: `https://your-app.up.railway.app`

## Option 3: Deploy with Docker (Any Platform)

Your existing [Dockerfile](Dockerfile) already works! Just push to any Docker-supporting service:

- **Fly.io**
- **Google Cloud Run**
- **AWS ECS**
- **Azure Container Apps**

### Quick Deploy to Fly.io:

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Deploy
fly launch --dockerfile Dockerfile
```

## Option 4: Keep Both Streamlit UI + FastAPI

You can have **both**:

1. **Streamlit Cloud** - For the interactive UI at `mem-shirt-size-predictor-model.streamlit.app`
2. **Render/Railway** - For the API at `mem-shirt-size-api.onrender.com`

This gives you:
- A public web interface for manual testing
- A REST API for programmatic access from my-exam-mate

## Update my-exam-mate to Use Your API

Once deployed, update your my-exam-mate code:

```python
# Before (localhost)
API_URL = "http://localhost:8000/predict"

# After (deployed)
API_URL = "https://mem-shirt-size-api.onrender.com/predict"

# Make request
response = requests.post(API_URL, json={
    "height_cm": 175,
    "weight_kg": 75,
    "age": 24,
    "gender": "male",
    "fit_preference": "regular",
    "build": "average"
})

result = response.json()
print(f"Recommended size: {result['recommended_size']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## Important Notes

### Free Tier Limitations:

- **Render**: Spins down after 15 min inactivity (cold start ~30s)
- **Railway**: 500 hours/month free
- **Streamlit Cloud**: Can't handle API requests

### Production Considerations:

For production use, consider:
- Paid plans for always-on instances
- Add API authentication (API keys)
- Rate limiting
- Monitoring and logging
- HTTPS (automatically handled by Render/Railway)

## Testing

After deployment, test all endpoints:

```bash
# Health check
curl https://your-api-url.com/health

# Model info
curl https://your-api-url.com/model-info

# Prediction
curl -X POST "https://your-api-url.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"height_cm": 175, "weight_kg": 75, "age": 24, "gender": "male", "fit_preference": "regular", "build": "average"}'
```

## Next Steps

1. Deploy to Render (easiest option)
2. Test the API with curl or Postman
3. Update my-exam-mate with the new API URL
4. Keep Streamlit for the UI if you want both!
