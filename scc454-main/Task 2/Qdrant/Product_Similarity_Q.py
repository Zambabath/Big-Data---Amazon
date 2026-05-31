import pandas as pd
import time
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct, 
    SearchParams
)

#for the api key and url
import os
from dotenv import load_dotenv

load_dotenv()
api = os.getenv("QDRANT_API_KEY")
url = os.getenv("DATABASE_URL")


#convert ASINs to numbers that Qdrant can use
def asin_to_int(asin_str):
    hash_object = hashlib.sha256(asin_str.encode('utf-8'))
    return int(hash_object.hexdigest(), 16) % 10**18


#testing
number = 11
test_asin = "B088BDFDHX"
collection_name = "product_sim"
    
#initialize client
client = QdrantClient(
        url=url, 
        api_key=api
    )

try:
    #data
    master_data = pd.read_parquet("master_product_vectors_tfidf.parquet")
    vector_dim = len(master_data.iloc[0]['vector'])

    #delete existing collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    #list extraction
    ids = master_data['id'].astype(str).tolist()
    vectors = master_data['vector'].tolist()
    titles = master_data['title'].astype(str).tolist()
    combined_meta = master_data['combined'].tolist()

    start_time = time.time()
        
    #zip is faster so use zip
    points = [
            PointStruct(
                id=asin_to_int(pid), 
                vector=vec, 
                payload={"combined_info": meta, "title": title}
            )
            for pid, vec, meta, title in zip(ids, vectors, combined_meta, titles)
        ]
        
    #upload points to collection in batches
    client.upload_points(
            collection_name=collection_name,
            points=points,
            batch_size=100, 
            #parallel=4
        )
    print(f"Ingesting data complete in {time.time() - start_time:.2f}s")

    #query
    print(f"\nSearching for products similar to: {test_asin}")
    target_match = master_data[master_data['id'].astype(str) == test_asin]

    if not target_match.empty:
        target_vector = target_match.iloc[0]['vector'].tolist()
        results = client.query_points(
            collection_name=collection_name,
            query=target_vector,
            limit=number,
            search_params=SearchParams(hnsw_ef=128, exact=False)
            )
            
        print(f"\nTop {number} Similar Products:")
        for i, point in enumerate(results.points):
            payload = point.payload
            original_asin = payload.get('original_asin', 'N/A')
            title = payload.get('title', 'N/A')
            score = point.score
            print(f"{i+1}. ASIN: {original_asin} | Score: {1-score:.4f} | Title: {title}")
    else:
        print(f"Error: ASIN {test_asin} not found.")

finally:
    client.close()