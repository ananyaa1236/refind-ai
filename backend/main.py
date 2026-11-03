"""ReFind AI API. Run from project root: uvicorn backend.main:app --reload"""
from __future__ import annotations
import io, json, os, re, sqlite3, threading, uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv('REFIND_DATA_DIR', str(ROOT / 'data'))).resolve()
UPLOADS = DATA / 'uploads'
DATA.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
DB = DATA / 'refind.sqlite3'
MAX_BYTES = 8 * 1024 * 1024
MODEL_ID = os.getenv('REFIND_CLIP_MODEL', 'openai/clip-vit-base-patch32')
model_lock = threading.Lock()
_model = None
_processor = None

@contextmanager
def db():
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()

with db() as con:
    con.execute('''CREATE TABLE IF NOT EXISTS items (
        id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
        category TEXT NOT NULL, color TEXT NOT NULL, location TEXT NOT NULL,
        date TEXT NOT NULL, description TEXT NOT NULL, image_name TEXT,
        embedding TEXT, created_at TEXT NOT NULL)''')
    con.execute('CREATE INDEX IF NOT EXISTS items_type_idx ON items(type)')

app = FastAPI(title='ReFind AI — CLIP Matching API', version='1.0.0')
app.mount('/uploads', StaticFiles(directory=UPLOADS), name='uploads')
@app.get('/app.js')
def javascript():
    return FileResponse(ROOT/'frontend'/'app.js', media_type='application/javascript')

def load_model():
    global _model, _processor
    with model_lock:
        if _model is None:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
                _processor = CLIPProcessor.from_pretrained(MODEL_ID)
                _model = CLIPModel.from_pretrained(MODEL_ID)
                _model.eval()
            except Exception as exc:
                raise HTTPException(503, f'CLIP unavailable. Install model dependencies and allow first model download: {type(exc).__name__}') from exc
    return _model, _processor

def embed_image(image: Image.Image) -> list[float]:
    import torch
    model, processor = load_model()
    inputs = processor(images=image, return_tensors='pt')
    with torch.inference_mode():
        vector = model.get_image_features(**inputs)
        # Transformers versions may return a tensor or a model output.
        if not isinstance(vector, torch.Tensor):
            vector = vector.pooler_output
        vector = vector[0].float().cpu().numpy()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise HTTPException(500, 'Model returned an invalid embedding')
    return (vector / norm).tolist()

def public_item(row):
    d = dict(row)
    d.pop('embedding', None)
    image_name = d.pop('image_name')
    d['photo'] = f'/uploads/{image_name}' if image_name else None
    return d

def tokens(text):
    return set(re.findall(r'[a-z0-9]{3,}', text.lower())) - {'the','and','with','found','lost','item','near'}

def compare(a, b):
    reasons = []
    # Image similarity is a cosine similarity, mapped into [0, 1] for ranking.
    image_score = None
    if a['embedding'] and b['embedding']:
        va = np.array(json.loads(a['embedding']), dtype=np.float32)
        vb = np.array(json.loads(b['embedding']), dtype=np.float32)
        if va.shape == vb.shape:
            cosine = float(np.clip(np.dot(va, vb), -1, 1))
            image_score = (cosine + 1) / 2
            reasons.append(f'CLIP image cosine: {cosine:.3f}')
    common = tokens(a['name']+' '+a['description']) & tokens(b['name']+' '+b['description'])
    all_words = tokens(a['name']+' '+a['description']) | tokens(b['name']+' '+b['description'])
    text_score = len(common)/len(all_words) if all_words else 0
    location_score = 1.0 if a['location'].casefold().strip() == b['location'].casefold().strip() else (0.5 if tokens(a['location']) & tokens(b['location']) else 0)
    day_delta = abs((date.fromisoformat(a['date']) - date.fromisoformat(b['date'])).days)
    time_score = max(0, 1 - day_delta/7)
    category_score = float(a['category'] == b['category'])
    color_score = float(a['color'].casefold().strip() == b['color'].casefold().strip())
    features = [('image',image_score,.45),('text',text_score,.20),('location',location_score,.10),('date',time_score,.10),('category',category_score,.10),('color',color_score,.05)]
    # Reweight when either item lacks a photo; do not pretend image matching occurred.
    present = [(v,w) for _,v,w in features if v is not None]
    weighted = sum(v*w for v,w in present)/sum(w for _,w in present)
    if category_score: reasons.append('Same category')
    if color_score: reasons.append('Same color')
    if location_score: reasons.append('Similar location')
    if day_delta <= 1: reasons.append('Dates within one day')
    if common: reasons.append('Shared keywords: '+', '.join(sorted(common)[:4]))
    return {'score':round(weighted*100,1),'image_cosine':round(image_score*2-1,4) if image_score is not None else None,'reasons':reasons}

@app.get('/api/health')
def health():
    return {'status':'ok','model':MODEL_ID,'model_loaded':_model is not None}

@app.get('/api/items')
def list_items():
    with db() as con:
        rows = con.execute('SELECT * FROM items ORDER BY created_at DESC').fetchall()
    return [public_item(row) for row in rows]

@app.post('/api/items', status_code=201)
async def create_item(type: Literal['lost','found']=Form(...), name: str=Form(...), category: str=Form(...), color: str=Form(...), location: str=Form(...), date_value: str=Form(...,alias='date'), description: str=Form(...), photo: UploadFile|None=File(None)):
    fields = [name,category,color,location,description]
    if any(not x.strip() for x in fields) or any(len(x)>2000 for x in fields) or len(name)>80:
        raise HTTPException(422,'Please provide valid item details')
    try:
        date.fromisoformat(date_value)
    except ValueError:
        raise HTTPException(422,'Date must be YYYY-MM-DD')
    image_name, embedding = None, None
    if photo and photo.filename:
        raw = await photo.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES: raise HTTPException(413,'Image exceeds 8 MB')
        try:
            image = Image.open(io.BytesIO(raw))
            image.verify()
            image = Image.open(io.BytesIO(raw))
            image = ImageOps.exif_transpose(image).convert('RGB')
            if image.width*image.height>24_000_000:
                raise HTTPException(413,'Image dimensions too large')
            image.thumbnail((1200,1200))
        except (UnidentifiedImageError,OSError,ValueError) as exc:
            raise HTTPException(422,'Invalid image file') from exc
        # Compute embedding before writing record, so failures do not leave broken reports.
        embedding = embed_image(image)
        image_name = uuid.uuid4().hex+'.jpg'
        image.save(UPLOADS/image_name, 'JPEG', quality=84)
    item_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc).isoformat()
    with db() as con:
        con.execute('INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?)', (item_id,type,name.strip(),category.strip(),color.strip(),location.strip(),date_value,description.strip(),image_name,json.dumps(embedding) if embedding else None,created))
        row = con.execute('SELECT * FROM items WHERE id=?',(item_id,)).fetchone()
    return public_item(row)

@app.get('/api/matches/{item_id}')
def matches(item_id: str, limit: int=10):
    with db() as con:
        target = con.execute('SELECT * FROM items WHERE id=?',(item_id,)).fetchone()
        if target is None: raise HTTPException(404,'Item not found')
        other = con.execute('SELECT * FROM items WHERE type != ?',(target['type'],)).fetchall()
    results = [{'item':public_item(row),**compare(target,row)} for row in other]
    results.sort(key=lambda x:x['score'],reverse=True)
    return {'query':public_item(target),'matches':results[:max(1,min(limit,50))], 'note':'Scores are heuristic rankings, not ownership probabilities.'}

@app.get('/')
def homepage():
    return FileResponse(ROOT/'frontend'/'index.html')
