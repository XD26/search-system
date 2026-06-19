import os
import tempfile
import ast
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import faiss
import torch
import clip
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from sqlmodel import SQLModel, Field, Session, create_engine, select

# Конфигурация
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "Food Images"
DB_PATH = BASE_DIR / "recipes.db"
CSV_PATH = DATA_DIR / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True, parents=True)

MODEL_NAME = "ViT-B/32"
VECTOR_DIM = 512
DEFAULT_LIMIT = 6
THRESHOLD = 0.12

# Имена файлов индексов (без пути - сохраняются в текущей директории)
IMAGE_INDEX_FILE = "image_index.bin"
TEXT_INDEX_FILE = "text_index.bin"

# Модели
class Recipe(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    ingredients: str
    cleaned_ingredients: str
    instructions: str
    image_name: str
    image_path: str
    image_vector_id: Optional[int] = Field(default=None)
    text_vector_id: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)

class VectorMapping(SQLModel, table=True):
    __tablename__ = "vector_mappings"
    id: Optional[int] = Field(default=None, primary_key=True)
    faiss_id: int = Field(index=True)
    recipe_id: int = Field(foreign_key="recipe.id")
    vector_type: str

# CLIP энкодер
class CLIPEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Устройство: {self.device}")
        print(f"Загрузка модели {MODEL_NAME}...")
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)
        self.model.eval()
        print("Модель загружена")
    
    def encode_image(self, path: str) -> np.ndarray:
        try:
            img = Image.open(path).convert("RGB")
            inp = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(inp)
            return (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype('float32').flatten()
        except:
            return np.zeros(VECTOR_DIM, dtype='float32')
    
    def encode_text(self, text: str) -> np.ndarray:
        try:
            tokens = clip.tokenize([text[:70]], truncate=True).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_text(tokens)
            return (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype('float32').flatten()
        except:
            return np.zeros(VECTOR_DIM, dtype='float32')

# FAISS менеджер
class FAISSManager:
    def __init__(self):
        self.image_index = self._load(IMAGE_INDEX_FILE)
        self.text_index = self._load(TEXT_INDEX_FILE)
        print(f"Индексы: изображений={self.image_index.ntotal}, текстов={self.text_index.ntotal}")
    
    def _load(self, filename: str):
        path = Path(filename)
        if path.exists() and path.stat().st_size > 0:
            try:
                return faiss.read_index(str(path))
            except Exception as e:
                print(f"Ошибка загрузки {filename}: {e}")
        return faiss.IndexIDMap(faiss.IndexFlatIP(VECTOR_DIM))
    
    def add_image(self, vec: np.ndarray) -> int:
        if np.all(vec == 0): return -1
        v = vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(v)
        vid = self.image_index.ntotal
        self.image_index.add_with_ids(v, np.array([vid]))
        return vid
    
    def add_text(self, vec: np.ndarray) -> int:
        if np.all(vec == 0): return -1
        v = vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(v)
        vid = self.text_index.ntotal
        self.text_index.add_with_ids(v, np.array([vid]))
        return vid
    
    def search_images(self, vec: np.ndarray, k: int):
        if self.image_index.ntotal == 0: return np.array([[]]), np.array([[]])
        v = vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(v)
        return self.image_index.search(v, min(k, self.image_index.ntotal))
    
    def search_texts(self, vec: np.ndarray, k: int):
        if self.text_index.ntotal == 0: return np.array([[]]), np.array([[]])
        v = vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(v)
        return self.text_index.search(v, min(k, self.text_index.ntotal))
    
    def save(self):
        try:
            faiss.write_index(self.image_index, IMAGE_INDEX_FILE)
            faiss.write_index(self.text_index, TEXT_INDEX_FILE)
            print(f"Индексы сохранены: img={self.image_index.ntotal}, txt={self.text_index.ntotal}")
        except Exception as e:
            print(f"Ошибка сохранения: {e}")

# Основной движок
class RecipeSearchEngine:
    def __init__(self):
        print("Инициализация...")
        self.encoder = CLIPEncoder()
        self.faiss = FAISSManager()
        self.engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        SQLModel.metadata.create_all(self.engine)
        
        db_count = self.count()
        print(f"БД: {db_count} рецептов, Индекс: {self.faiss.text_index.ntotal} векторов")
        print("Готово")
    
    def clean_text(self, text: str) -> str:
        return re.sub(r'[^\w\s]', ' ', str(text)).strip()[:60]
    
    def rebuild_indexes_from_db(self):
        """Перестроить индексы из рецептов в БД"""
        print("Перестроение индексов...")
        
        with Session(self.engine) as s:
            for m in s.exec(select(VectorMapping)).all():
                s.delete(m)
            s.commit()
        
        self.faiss = FAISSManager()
        
        with Session(self.engine) as s:
            recipes = s.exec(select(Recipe)).all()
            total = len(recipes)
            print(f"Обработка {total} рецептов...")
            
            for i, r in enumerate(recipes):
                try:
                    if not Path(r.image_path).exists():
                        continue
                    
                    img_vec = self.encoder.encode_image(r.image_path)
                    if np.all(img_vec == 0): continue
                    
                    txt_vec = self.encoder.encode_text(self.clean_text(r.title))
                    if np.all(txt_vec == 0): continue
                    
                    img_id = self.faiss.add_image(img_vec)
                    txt_id = self.faiss.add_text(txt_vec)
                    
                    r.image_vector_id = img_id
                    r.text_vector_id = txt_id
                    s.add(r)
                    s.flush()
                    
                    s.add(VectorMapping(faiss_id=img_id, recipe_id=r.id, vector_type="image"))
                    s.add(VectorMapping(faiss_id=txt_id, recipe_id=r.id, vector_type="text"))
                    
                    if (i + 1) % 100 == 0:
                        print(f"  {i + 1}/{total}")
                        s.commit()
                except Exception as e:
                    print(f"  Ошибка: {e}")
            
            s.commit()
        
        self.faiss.save()
        print(f"Готово: {self.faiss.text_index.ntotal} векторов")
    
    def search(self, vec: np.ndarray, vtype: str, limit: int, threshold: float) -> List[Dict]:
        dists, ids = (self.faiss.search_images(vec, limit*3) if vtype == "image" 
                      else self.faiss.search_texts(vec, limit*3))
        
        if ids.size == 0: return []
        
        res, seen = [], set()
        with Session(self.engine) as s:
            for score, fid in zip(dists[0], ids[0]):
                if score < threshold: continue
                
                m = s.exec(select(VectorMapping).where(
                    VectorMapping.faiss_id == int(fid),
                    VectorMapping.vector_type == vtype
                )).first()
                
                if m and (r := s.get(Recipe, m.recipe_id)) and r.id not in seen:
                    seen.add(r.id)
                    res.append({
                        'id': r.id, 'title': r.title,
                        'ingredients': r.ingredients, 'instructions': r.instructions,
                        'image_name': r.image_name,
                        'similarity': float(score),
                        'similarity_percent': round(float(score) * 100, 1)
                    })
                    if len(res) >= limit: break
        return res
    
    def search_by_text(self, q: str, limit=DEFAULT_LIMIT, threshold=THRESHOLD):
        if not q.strip(): return []
        query = self.clean_text(q)
        print(f"Поиск: '{query}'")
        return self.search(self.encoder.encode_text(query), "text", limit, threshold)
    
    def search_by_image(self, path: str, limit=DEFAULT_LIMIT, threshold=THRESHOLD):
        return self.search(self.encoder.encode_image(path), "image", limit, threshold)
    
    def count(self) -> int:
        with Session(self.engine) as s:
            return len(s.exec(select(Recipe)).all())
    
    def import_csv(self):
        """Импорт из CSV с построением индексов"""
        if not CSV_PATH.exists():
            print(f"CSV не найден")
            return 0
        
        print("Импорт из CSV с построением индексов...")
        df = pd.read_csv(CSV_PATH)
        processed = 0
        
        # Очищаем старые данные
        with Session(self.engine) as s:
            for m in s.exec(select(VectorMapping)).all():
                s.delete(m)
            for r in s.exec(select(Recipe)).all():
                s.delete(r)
            s.commit()
        
        # Новые индексы
        self.faiss = FAISSManager()
        
        for i, row in df.iterrows():
            try:
                # Поиск изображения
                img_name = str(row['Image_Name'])
                img_path = None
                for e in ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    p = IMAGES_DIR / (img_name + e)
                    if p.exists():
                        img_path = p
                        break
                if not img_path: continue
                
                # Парсинг ингредиентов
                ing = str(row['Cleaned_Ingredients'])
                ing = ', '.join(ast.literal_eval(ing)) if ing.startswith('[') else ing.strip("[]'\"")
                
                title = str(row['Title']).strip()
                
                # Создаем векторы
                img_vec = self.encoder.encode_image(str(img_path))
                if np.all(img_vec == 0): continue
                
                txt_vec = self.encoder.encode_text(self.clean_text(title))
                if np.all(txt_vec == 0): continue
                
                # Добавляем в FAISS
                img_id = self.faiss.add_image(img_vec)
                txt_id = self.faiss.add_text(txt_vec)
                
                # Сохраняем в БД
                with Session(self.engine) as s:
                    rec = Recipe(
                        title=title,
                        ingredients=str(row['Ingredients']),
                        cleaned_ingredients=ing,
                        instructions=str(row['Instructions']),
                        image_name=img_path.name,
                        image_path=str(img_path),
                        image_vector_id=img_id,
                        text_vector_id=txt_id
                    )
                    s.add(rec)
                    s.flush()
                    
                    s.add(VectorMapping(faiss_id=img_id, recipe_id=rec.id, vector_type="image"))
                    s.add(VectorMapping(faiss_id=txt_id, recipe_id=rec.id, vector_type="text"))
                    s.commit()
                    
                    processed += 1
                
                if processed % 50 == 0:
                    print(f"  Обработано: {processed}")
                    self.faiss.save()
                    
            except Exception as e:
                print(f"  Ошибка в строке {i}: {e}")
        
        self.faiss.save()
        print(f"Импорт завершен: {processed} рецептов, {self.faiss.text_index.ntotal} векторов")
        return processed

# FastAPI
search_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine
    print("\n" + "="*50)
    print("ПОИСК РЕЦЕПТОВ")
    print("="*50)
    
    search_engine = RecipeSearchEngine()
    
    if search_engine.count() == 0 and CSV_PATH.exists():
        print("\nБаза пуста, запуск импорта...")
        search_engine.import_csv()
    
    print("\nСервер: http://localhost:8000")
    print("="*50 + "\n")
    yield

app = FastAPI(title="Recipe Search", lifespan=lifespan)

@app.get("/")
async def home():
    html = BASE_DIR / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8") if html.exists() else "<h1>Recipe Search API</h1>")

@app.get("/recipe-image/{name}")
async def get_image(name: str):
    base = name.split('.')[0]
    for e in ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        if (p := IMAGES_DIR / (base + e)).exists():
            return FileResponse(p)
    return FileResponse(BASE_DIR / "no_image.jpg")

@app.get("/search/text")
async def search_text(q: str, limit: int = DEFAULT_LIMIT, threshold: float = THRESHOLD):
    return {"results": search_engine.search_by_text(q, limit, threshold)}

@app.post("/search/image")
async def search_image(file: UploadFile = File(...), limit: int = DEFAULT_LIMIT, threshold: float = THRESHOLD):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return {"results": search_engine.search_by_image(tmp_path, limit, threshold)}
    finally:
        os.unlink(tmp_path)

@app.get("/stats")
async def stats():
    return {"total": search_engine.count(), "indexed": search_engine.faiss.text_index.ntotal}

@app.post("/admin/rebuild")
async def rebuild():
    search_engine.rebuild_indexes_from_db()
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
