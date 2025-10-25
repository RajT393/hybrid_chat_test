# config_example.py — copy to config.py and fill with real values.
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

OPENAI_API_KEY = "sk-proj-......" # your OpenAI API key

PINECONE_API_KEY = "pcsk_311....." # your Pinecone API key
PINECONE_ENV = "us-east1-gcp"   # example
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 1536       # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
