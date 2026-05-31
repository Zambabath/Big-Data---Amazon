import chromadb
import pandas as pd
import time

#test parameters
#batch_size cannot be more than 5461 since its done on the disk
number = 11
test_user_id = "AFSKPY37N3C43SOI5IEXEK5JSIYA" 
batch_size = 5000

#get the vectorised data
master_user_data = pd.read_parquet("master_user_vectors_TFIDF.parquet")

#chroma initialisation
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection(name="user_sim")
    print("Deleted old 'user_sim' collection.")
except:
    pass

collection = client.get_or_create_collection(
    name="user_sim", 
    metadata={"hnsw:space": "cosine",
              "hnsw:construction_ef": 100,
              "hnsw:M": 16
              }
)

#batch the data otherwise cannot do it since its too big
total_records = len(master_user_data)
start_time = time.time()

for i in range(0, total_records, batch_size):
    batch = master_user_data[i : i + batch_size]
    ids=batch['id'].astype(str).tolist()
    collection.add(
        ids=ids,
        embeddings=batch['vector'].tolist(),
        metadatas=[{"user_id": uid} for uid in ids]
    )

#query
print(f"\nSearching for users similar to: {test_user_id}")

#Find the vector for our test_user_id
target_user = master_user_data[master_user_data['id'].astype(str) == test_user_id]

if not target_user.empty:
    print(f"\nQuerying ChromaDB for: {test_user_id}")
    query_start = time.time()
    results = collection.query(
        query_embeddings=[target_user.iloc[0]['vector']],
        n_results=number,
        include=["metadatas", "distances"]
    )
    query_end = time.time()

    
    print(f"\nTop {number} Similar Users:")
    print(f"Query Time: {query_end - query_start:.4f}s")
    for i in range(len(results['ids'][0])):
        print(f"{i+1}. User: {results['ids'][0][i]} | Distance: {results['distances'][0][i]:.4f}")