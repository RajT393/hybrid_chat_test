import json
from neo4j import GraphDatabase
from tqdm import tqdm
import config

DATA_FILE = "vietnam_travel_dataset.json"

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

def upsert_node(tx, node):
    labels = [node.get("type","Unknown"), "Entity"]
    label_cypher = ":" + ":".join(labels)
    props = {k:v for k,v in node.items() if k not in ("connections",)}
    tx.run(
        f"MERGE (n{label_cypher} {{id: $id}}) SET n += $props",
        id=node["id"], props=props
    )

def create_relationship(tx, src, rel):
    rel_type = rel.get("relation", "RELATED_TO")
    tgt = rel.get("target")
    if not tgt:
        return
    cypher = (
        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
        f"MERGE (a)-[r:{rel_type}]->(b)"
    )
    tx.run(cypher, src=src, tgt=tgt)

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    with driver.session() as s:
        s.execute_write(create_constraints)
        for node in tqdm(nodes, desc="Nodes"):
            s.execute_write(upsert_node, node)
        for node in tqdm(nodes, desc="Relations"):
            for rel in node.get("connections", []):
                s.execute_write(create_relationship, node["id"], rel)
    print("✅ Neo4j load complete.")

if __name__ == "__main__":
    main()
