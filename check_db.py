import sqlite3
from pathlib import Path

def run_diagnostics():
    db_path = Path(r"c:\RAG1\data\rag.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # We will print repr() to avoid carriage returns hiding text
    cursor.execute("SELECT chunk_id, markdown_text FROM chunks WHERE markdown_text LIKE ?", ("%vijaykumar%N%",))
    res = cursor.fetchall()
    
    for row in res:
        chunk_id = row[0]
        text = row[1]
        print(f"\n--- FULL RAW CHUNK TEXT FOR CHUNK ID: {chunk_id} ---")
        # Print string representation safely to avoid console overwrite 
        safe_text = repr(text).replace('\\n', '\n')
        print(safe_text)
        print("---------------------------------------------------------")

    conn.close()

if __name__ == '__main__':
    run_diagnostics()
