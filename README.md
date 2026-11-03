# ReFind AI — real CLIP image-matching backend

A working FastAPI + SQLite + pretrained CLIP image-embedding prototype connected to the existing ReFind AI HTML/CSS/JS website.

## Requirements
- Python 3.10–3.12 recommended, 64-bit
- Internet access **the first time** CLIP is used, to download model weights from Hugging Face
- Roughly 1–2 GB of disk space for Python packages and model cache (varies by platform)

## Run on Windows (PowerShell)

```powershell
cd refind-ai-clip
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

If `py -3.11` is unavailable, use `python -m venv .venv` with a supported Python version. If PowerShell blocks activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m uvicorn backend.main:app --reload` without activating.

## Run on macOS / Linux

```bash
cd refind-ai-clip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000/** (not the HTML file directly). API documentation: **http://127.0.0.1:8000/docs**. The first **photo upload** may take a while while the model downloads and initializes.

## Test the model
1. Submit a **lost** report with a photo of an object.
2. Submit a **found** report with a different photo of the same or a similar object.
3. Click **Find AI matches**. The results show the actual CLIP cosine similarity and a combined ranking score.
4. Try unrelated images to compare rankings.

Data persists in `data/refind.sqlite3`, with photos in `data/uploads/`. Deleting the `data` folder resets the demo. Do not commit `data/` or `.venv/` to Git.

## API
- `GET /api/health` — health and model status
- `GET /api/items` — list reports
- `POST /api/items` — multipart form fields `type`, `name`, `category`, `color`, `location`, `date`, `description`, optional `photo`
- `GET /api/matches/{item_id}` — rank opposite-type reports
- `GET /docs` — interactive Swagger UI

The image embedding is a 512-dimensional normalized vector for the default CLIP ViT-B/32 model. The ranking uses image cosine similarity, text keyword overlap, location, dates, category and color. Missing photos cause the score weights to be renormalized. **The combined score is not a calibrated ownership probability.**

## Important limits before public deployment
This is a **local hackathon prototype, not a secure public service**. It does not include authentication, private ownership claims, moderation, rate limiting, or cloud storage. Anyone with access to the server can view uploaded reports and images. Do not submit identity documents, personal addresses, phone numbers or confidential photos. For production, add user accounts, authorization, private image access, consent/privacy policies, upload scanning, moderation, rate limiting, PostgreSQL and object storage. Serve through HTTPS. The claim button intentionally does not save sensitive evidence.

Model choice: `REFIND_CLIP_MODEL=openai/clip-vit-base-patch32` (default). Switching to another model can invalidate previously stored embeddings; rebuild embeddings when changing model versions.
