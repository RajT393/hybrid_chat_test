import json
import time
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
from google import genai
import config

DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32
COLLECTION = config.QDRANT_COLLECTION

# Gemini + Qdrant clients
client = genai.Client()
qdrant = QdrantClient(url=config.QDRANT_URL)

def ensure_collection(size=768):
    try:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
        )
        print(f"✅ Collection '{COLLECTION}' created successfully.")
    except Exception as e:
        print("⚠️ Collection exists or error:", e)

def get_embeddings(texts):
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def chunked(data, n):
    for i in range(0, len(data), n):
        yield data[i:i+n]

def main():
    ensure_collection()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        txt = node.get("semantic_text") or node.get("description", "")
        if not txt.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", ""),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], txt[:1000], meta))

    print(f"Preparing {len(items)} vectors for upload...")
    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading"):
        ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        metas = [b[2] for b in batch]
        embeds = get_embeddings(texts)
        points = [
            models.PointStruct(id=i, vector=vec, payload=meta)
            for i, vec, meta in zip(ids, embeds, metas)
        ]
        qdrant.upsert(collection_name=COLLECTION, points=points)
        time.sleep(0.2)
    print("✅ Upload complete.")

if __name__ == "__main__":
    main()
