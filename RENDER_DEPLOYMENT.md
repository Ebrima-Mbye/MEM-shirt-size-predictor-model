# Deploy FastAPI to Render.com

Your FastAPI server at [`app/server.py`](app/server.py) is production-ready with the `/predict` endpoint that your my-exam-mate app needs.

## Quick Deploy to Render (Free)

### Option 1: Using Render Dashboard (Easiest)

1. **Go to Render**: https://render.com/
2. **Sign up/Login** with your GitHub account
3. **Click "New +"** → **"Web Service"**
4. **Connect your repository**: `Ebrima-Mbye/MEM-shirt-size-predictor-model`
5. **Configure the service**:
   - **Name**: `mem-shirt-size-predictor-api`
   - **Region**: Choose closest to your users (e.g., Frankfurt for Europe, Oregon for US)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

6. **Click "Create Web Service"**

### Option 2: Using render.yaml (Blueprint)

The `render.yaml` file is already configured. Just:

1. Go to https://render.com/
2. Click **"New +"** → **"Blueprint"**
3. Connect your repository
4. Render will auto-detect the `render.yaml` and deploy

## After Deployment

You'll get a URL like: `https://mem-shirt-size-predictor-api.onrender.com`

### Update my-exam-mate Environment Variable

In your **my-exam-mate** project, add this environment variable:

**For Vercel/Production:**

```bash
MEM_PREDICTOR_BASE_URL=https://mem-shirt-size-predictor-api.onrender.com
```

**For Local Development:**

```bash
MEM_PREDICTOR_BASE_URL=http://localhost:8000
```

Add to `.env.local`:

```env
MEM_PREDICTOR_BASE_URL=https://mem-shirt-size-predictor-api.onrender.com
```

## Test Your API

Once deployed, test it:

```bash
curl -X POST https://mem-shirt-size-predictor-api.onrender.com/predict \
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

Expected response:

```json
{
  "recommended_size": "L",
  "confidence": 0.85,
  "probabilities": {
    "S": 0.01,
    "M": 0.09,
    "L": 0.85,
    "XL": 0.04,
    "XXL": 0.01
  },
  "model_version": "1.0"
}
```

## API Endpoints

Your FastAPI server provides:

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /model-info` - Model metadata
- `POST /predict` - Shirt size prediction
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

Visit `https://your-api-url.onrender.com/docs` for interactive testing!

## Important Notes

### Free Tier Limitations

- **Cold starts**: Free instances sleep after 15 minutes of inactivity
- **First request** after sleep takes ~30-60 seconds to wake up
- **Solution**: Use a cron job or uptime monitor to ping every 10 minutes

### Keep It Awake (Optional)

Use a service like:

- **UptimeRobot** (free): https://uptimerobot.com/
- **Cron-job.org** (free): https://cron-job.org/

Set it to ping: `https://your-api-url.onrender.com/health` every 10 minutes

### Upgrade Options

If you need:

- No cold starts
- More RAM/CPU
- Custom domain

Consider upgrading to Render's **Starter plan** ($7/month)

## Troubleshooting

### Build Fails

- Check that `app/model.joblib` is in your repository
- Verify `requirements.txt` has correct dependencies

### 502 Bad Gateway

- Service might be starting (wait 1-2 minutes)
- Check Render logs for errors

### Model Not Found Error

- Make sure `app/model.joblib` is committed to git:
  ```bash
  git add app/model.joblib
  git commit -m "Add trained model"
  git push
  ```

## Alternative Platforms

If Render doesn't work for you:

### Railway.app

```bash
railway login
railway init
railway up
```

### Fly.io

```bash
fly auth login
fly launch
fly deploy
```

### Heroku

```bash
heroku login
heroku create mem-shirt-predictor
git push heroku main
```

All provide similar free tiers with automatic deployments from GitHub!
