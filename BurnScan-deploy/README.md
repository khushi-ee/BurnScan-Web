# 🔥 BurnScan — AIIMS Paediatric Burns Analysis

AI-assisted burn wound analysis tool for AIIMS Paediatric Surgery.  
Extracts colour, depth, and texture features from clinical photographs and renders annotated block-average grid overlays.

---

## Project Structure

```
BurnScan/
├── backend/
│   └── main.py           # FastAPI app — all HTTP endpoints
├── core/
│   └── pipeline.py       # Self-contained ML pipeline (feature extraction + classification)
├── frontend/
│   └── index.html        # Single-page app (no build step needed)
├── Procfile              # Railway / Heroku start command
├── railway.toml          # Railway deployment config
├── nixpacks.toml         # Nixpacks build config (Python + system libs)
├── runtime.txt           # Python version pin
├── requirements.txt      # Python dependencies
└── .gitignore
```

---

## Deploy to Railway (one-click)

1. Push this folder to a GitHub repository.
2. Go to [railway.com](https://railway.com) → **New Project** → **Deploy from GitHub Repo**.
3. Select your repo. Railway auto-detects `nixpacks.toml` and `Procfile`.
4. Once deployed, click **Generate Domain** in the service settings.
5. Your app is live — the frontend and API are served from the same URL.

---

## Local Development

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server (from project root)
uvicorn backend.main:app --reload --port 8000

# 4. Open http://localhost:8000 in your browser
```

---

## API Endpoints

| Method | Path            | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/api/health`   | Liveness probe — returns `{"status":"ok"}` |
| POST   | `/api/analyse`  | Upload image → JSON + base64 grid PNGs |
| GET    | `/`             | Serves the frontend UI               |

### POST `/api/analyse` — form fields

| Field        | Type    | Required | Default | Notes                        |
|--------------|---------|----------|---------|------------------------------|
| `file`       | File    | ✅       | —       | JPG or PNG, max 20 MB        |
| `k`          | int     | ❌       | 10      | Block size in pixels (5–30)  |
| `patient_id` | string  | ❌       | —       | Free-text patient identifier |
| `patient_age`| int     | ❌       | —       | Age in years                 |
| `burn_cause` | string  | ❌       | —       | Aetiology free-text          |

---

## Demo Login Credentials

| Username    | Password   | Role     |
|-------------|------------|----------|
| `dr.sharma` | `aiims2024`| Doctor   |
| `dr.mehta`  | `burns123` | Resident |
| `admin`     | `admin123` | Admin    |

> ⚠️ **Research prototype only.** Not clinically validated. All outputs must be reviewed by a qualified burns specialist.
