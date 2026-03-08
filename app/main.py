from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from starlette.responses import FileResponse
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

app = FastAPI(title="Trademarkia Semantic Search Demo")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: float
    result: str
    dominant_cluster: int

class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    similarity_threshold: float

# Simple in-memory mock cache
MOCK_CACHE = {}
MOCK_STATS = {
    "hits": 0,
    "misses": 0
}

MOCK_RESULTS = [
    "1. [sim=0.912] [comp.sys.ibm.pc.hardware] Are there any known issues with the new motherboard chipsets? I'm looking to upgrade my rig soon...",
    "2. [sim=0.875] [comp.sys.mac.hardware] I've been comparing the memory bandwidth on the latest models and the specifications look solid...",
    "3. [sim=0.850] [sci.electronics] Can someone explain the voltage specifications for this specific component? The manual is unclear...",
    "4. [sim=0.812] [comp.graphics] What kind of hardware specs do I need to run the latest 3D rendering software efficiently?...",
    "5. [sim=0.790] [misc.forsale] I am selling my old PC hardware, including a graphics card and 16GB of RAM. Serious offers only..."
]

@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest):
    q = payload.query.strip().lower()
    if not q:
        raise HTTPException(status_code=400, detail="Query must be non-empty.")

    # Extremely rudimentary mock caching
    cache_hit = False
    similarity_score = 1.0
    matched_query = q
    result_str = "\n".join([f"Query: {payload.query}", "", "Top matches:"] + MOCK_RESULTS)
    dominant_cluster = random.randint(0, 19)

    for cached_q, data in MOCK_CACHE.items():
        if cached_q in q or q in cached_q or len(set(q.split()) & set(cached_q.split())) > 1:
            cache_hit = True
            similarity_score = random.uniform(0.86, 0.99)
            matched_query = cached_q
            result_str = data["result"]
            dominant_cluster = data["cluster"]
            break

    if cache_hit:
        MOCK_STATS["hits"] += 1
    else:
        MOCK_STATS["misses"] += 1
        MOCK_CACHE[q] = {"result": result_str, "cluster": dominant_cluster}

    return QueryResponse(
        query=payload.query,
        cache_hit=cache_hit,
        matched_query=matched_query if cache_hit else payload.query,
        similarity_score=similarity_score,
        result=result_str,
        dominant_cluster=dominant_cluster,
    )

@app.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats_endpoint():
    total = MOCK_STATS["hits"] + MOCK_STATS["misses"]
    rate = (MOCK_STATS["hits"] / total) if total > 0 else 0.0
    return CacheStatsResponse(
        total_entries=len(MOCK_CACHE),
        hit_count=MOCK_STATS["hits"],
        miss_count=MOCK_STATS["misses"],
        hit_rate=rate,
        similarity_threshold=0.85,
    )

@app.delete("/cache")
def cache_delete_endpoint():
    MOCK_CACHE.clear()
    MOCK_STATS["hits"] = 0
    MOCK_STATS["misses"] = 0
    return {"status": "ok", "message": "Semantic cache flushed."}


# --------- Static frontend (optional single-origin deployment) ---------

if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        index_path = FRONTEND_DIST / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not built.")
