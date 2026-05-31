import chromadb
import pandas as pd
import time
import numpy as np

#testing
#batch_size cannot be more than 5461 since its done on the disk
number = 11
test_asin = "B088BDFDHX" 
batch_size = 5000

#data
master_data = pd.read_parquet("master_product_vectors_tfidf.parquet")

client = chromadb.PersistentClient(path="./chroma_db")

#delete old collection
try:
    client.delete_collection(name="product_sim")
    print("Deleted old 'product_sim' collection.")
except:
    pass

collection = client.get_or_create_collection(
    name="product_sim", 
    metadata={"hnsw:space": "cosine",
              "hnsw:construction_ef": 100,
              "hnsw:M": 16
              }
)

#batch the data
total_records = len(master_data)
start_time = time.time()
    
for i in range(0, total_records, batch_size):
    batch = master_data.iloc[i : i + batch_size]
    
    collection.add(
        ids=batch['id'].astype(str).tolist(),
        embeddings=batch['vector'].tolist(),
        metadatas = batch[['title']].to_dict('records')
    )

print(f"ingesting data complete in {time.time() - start_time:.2f}s")

#query
print(f"\nSearching for products similar to: {test_asin}")

#find the vector for test_asin
target_match = master_data[master_data['id'].astype(str) == test_asin]

if not target_match.empty:
    target_vector = target_match.iloc[0]['vector']
    
    print(f"\nQuerying ChromaDB for: {test_asin}")
    query_start = time.time()
    results = collection.query(
        query_embeddings=[target_vector],
        n_results=number,
        where={"id": {"$ne": test_asin}},
        include=["metadatas", "distances"]
    )
    query_end = time.time()
    
    print(f"\nTop {number} Similar Products:")
    print(f"Query Time: {query_end - query_start:.4f}s")
    
    for i in range(len(results['ids'][0])):
        product_id = results['ids'][0][i]
        title = results['metadatas'][0][i].get('title', 'N/A')
        distance = results['distances'][0][i]
        print(f"{i+1}. ID: {product_id} | Distance: {distance:.4f} | Title: {title}")
else:
    print(f"Error: ASIN {test_asin} not found in the master vector file.")