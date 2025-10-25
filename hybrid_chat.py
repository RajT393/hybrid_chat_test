import json
from typing import List
from google import genai
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import config

TOP_K = 5

# Clients
llm = genai.Client()
qdrant = QdrantClient(url=config.QDRANT_URL)
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

# -----------------------------
# Embedding helper
# -----------------------------
def embed_text(text: str) -> List[float]:
    resp = llm.embeddings.create(model=config.EMBED_MODEL, input=text)
    return resp.data[0].embedding

# -----------------------------
# Vector search
# -----------------------------
def qdrant_query(query_text: str, top_k=TOP_K):
    vec = embed_text(query_text)
    hits = qdrant.search(collection_name=config.QDRANT_COLLECTION, query_vector=vec, limit=top_k)
    print(f"🔍 Found {len(hits)} vector matches")
    return hits

# -----------------------------
# Graph context
# -----------------------------
def fetch_graph_context(names):
    facts = []
    with driver.session() as session:
        for name in names:
            cypher = (
                "MATCH (n:Entity {id:$id})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, m.id AS id, m.name AS name, m.description AS desc LIMIT 5"
            )
            for r in session.run(cypher, id=name):
                facts.append({
                    "rel": r["rel"],
                    "target": r["name"],
                    "desc": (r["desc"] or "")[:200]
                })
    print(f"🧩 Retrieved {len(facts)} graph facts")
    return facts

# -----------------------------
# Build prompt
# -----------------------------
def build_prompt(user_q, hits, facts):
    sem = "\n".join([f"- {h.payload.get('name')} ({h.payload.get('city')})" for h in hits])
    rels = "\n".join([f"- {f['rel']} → {f['target']}: {f['desc']}" for f in facts])
    return (
        f"You are a helpful travel planner.\n\n"
        f"User question: {user_q}\n\n"
        f"Semantic results:\n{sem}\n\n"
        f"Graph context:\n{rels}\n\n"
        "Give a short, realistic travel answer, suggest an itinerary, and mention at least one location."
    )

# -----------------------------
# Chat completion
# -----------------------------
def chat(prompt):
    res = llm.models.generate_content(model=config.CHAT_MODEL, contents=prompt)
    return res.text

# -----------------------------
# Interactive loop
# -----------------------------
def main():
    print("🌏 Hybrid AI Travel Assistant — type 'exit' to quit.")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in ("exit","quit"):
            break
        hits = qdrant_query(q)
        ids = [h.payload.get("id") for h in hits if h.payload.get("id")]
        facts = fetch_graph_context(ids)
        prompt = build_prompt(q, hits, facts)
        ans = chat(prompt)
        print("\n🤖 Assistant:\n", ans, "\n")

if __name__ == "__main__":
    main()
