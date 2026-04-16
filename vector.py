import faiss

# Load the file
index = faiss.read_index(r"c:\RAG1\index\sentences.index")

print(f"Total Vectors in Index: {index.ntotal}")
print(f"Vector Dimensions: {index.d}")

# Extract and print the actual embedding vector for Sentence 0
vector_zero = index.reconstruct(0)
print("\nFirst Vector (Sentence 0):")
print(vector_zero)
