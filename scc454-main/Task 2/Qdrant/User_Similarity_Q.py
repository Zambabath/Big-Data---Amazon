import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    ScalarQuantization, ScalarType, ScalarQuantizationConfig,
    OptimizersConfigDiff
)
import pandas as pd

#for the api key and url
import os
from dotenv import load_dotenv

load_dotenv()
api = os.getenv("QDRANT_API_KEY")
url = os.getenv("DATABASE_URL")

#initialise parameters
number = 11
test_user_id = "AFSKPY37N3C43SOI5IEXEK5JSIYA" 
batch_size = 5000 
collection_name = "user_sim"

qdrant_client = QdrantClient(
    url=url, 
    api_key=api
)

#delete existing collection
try:
    if qdrant_client.collection_exists(collection_name=collection_name):
        print(f"Deleting existing collection: {collection_name}")
        qdrant_client.delete_collection(collection_name=collection_name)
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=256, 
            distance=Distance.COSINE, 
            on_disk=True
        ),
        #do scalar since working with large amount of data - loses some accuracy but acceptable
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            ),
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0
        )
    )

    #data
    master_user_data = pd.read_parquet("master_user_vectors_TFIDF.parquet")
    total_records = len(master_user_data)
    start_time = time.time()

    for i in range(0, total_records, batch_size):
        batch = master_user_data[i : i + batch_size]
        
        #convert columns to lists 
        user_ids = batch['id'].astype(str).tolist()
        vectors = batch['vector'].tolist()
        
        #generate UUIDs in a list
        uuids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, uid)) for uid in user_ids]

        #use zip to combine them efficiently
        points = [
            PointStruct(id=uuid_val, vector=vec, payload={"user_id": uid})
            for uuid_val, vec, uid in zip(uuids, vectors, user_ids)
        ]
        
        qdrant_client.upsert(collection_name=collection_name, points=points, wait=False)
        
        if i % 20000 == 0:
            print(f"Progress: {i}/{total_records} | Elapsed: {int(time.time() - start_time)}s")

    #reenable indexing
    qdrant_client.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20000)
    )

    while True:
        info = qdrant_client.get_collection(collection_name=collection_name)
        if info.status == "green":
            break#=
        #sleep the time cause it almost broke my laptop
        time.sleep(10)

    #query
    target_row = master_user_data[master_user_data["id"].astype(str) == test_user_id]

    if not target_row.empty:
        target_vector = target_row.iloc[0]["vector"]
        
        response = qdrant_client.query_points(
            collection_name=collection_name,
            query=target_vector,
            limit=number
        )

        print(f"\nTop {number} Users similar to {test_user_id}:")
        print("-" * 50)
        for i, hit in enumerate(response.points):
            distance = max(0, 1.0 - hit.score)
            print(f"{i+1:2}. User: {hit.payload['user_id']} | Distance: {distance:.4f}")
    else:
        print(f"User {test_user_id} not found in the dataset.")

finally:
    qdrant_client.close()