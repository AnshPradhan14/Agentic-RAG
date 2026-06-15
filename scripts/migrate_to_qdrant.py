import os
import sys
import uuid
import logging
import sqlite3
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qdrant_client.http import models

from src.core.config import DB_PATH, QDRANT_COLLECTION
from src.tools.rag_tools import _get_embedding_model, _get_qdrant_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate_to_qdrant")

def migrate():
    logger.info("Starting FAISS to Qdrant migration...")
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    try:
        # Check if Qdrant column exists
        cur = conn.execute("PRAGMA table_info(sentences)")
        columns = [c["name"] for c in cur.fetchall()]
        if "qdrant_id" not in columns:
            logger.error("qdrant_id column not found in sentences table. Did you run the updated server first?")
            return
            
        rows = conn.execute('''
            SELECT s.sentence_id, s.chunk_id, s.sentence_text, s.qdrant_id, c.doc_id, d.source 
            FROM sentences s
            JOIN chunks c ON s.chunk_id = c.chunk_id
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE s.qdrant_id IS NULL
        ''').fetchall()
        
        if not rows:
            logger.info("No sentences found requiring migration.")
            return
            
        logger.info(f"Found {len(rows)} sentences to migrate.")
        
        # Load embedding model
        model = _get_embedding_model()
        client = _get_qdrant_client()
        
        # Ensure collection exists
        texts = [row["sentence_text"] for row in rows]
        
        # Encode a single batch to get dimensions
        logger.info("Encoding single sentence to get dimensions...")
        sample_emb = model.encode([texts[0]], convert_to_numpy=True)
        dim = sample_emb.shape[1]
        
        collections = client.get_collections().collections
        if not any(c.name == QDRANT_COLLECTION for c in collections):
            logger.info(f"Creating Qdrant collection '{QDRANT_COLLECTION}' with dim={dim}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            
        # Process in batches
        BATCH_SIZE = 100
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i+BATCH_SIZE]
            batch_texts = [r["sentence_text"] for r in batch]
            
            logger.info(f"Encoding batch {i//BATCH_SIZE + 1}/{(len(rows) + BATCH_SIZE - 1)//BATCH_SIZE}...")
            embeddings = model.encode(batch_texts, convert_to_numpy=True).astype(np.float32)
            
            qdrant_points = []
            updates = []
            
            for j, row in enumerate(batch):
                qdrant_id = uuid.uuid4().hex
                qdrant_points.append(
                    models.PointStruct(
                        id=qdrant_id,
                        vector=embeddings[j].tolist(),
                        payload={
                            "chunk_id": row["chunk_id"],
                            "doc_id": row["doc_id"],
                            "doc_name": row["source"],
                            "sentence_text": row["sentence_text"]
                        }
                    )
                )
                updates.append((qdrant_id, row["sentence_id"]))
                
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=qdrant_points
            )
            
            conn.executemany("UPDATE sentences SET qdrant_id = ? WHERE sentence_id = ?", updates)
            conn.commit()
            logger.info(f"Successfully migrated {len(batch)} sentences.")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
