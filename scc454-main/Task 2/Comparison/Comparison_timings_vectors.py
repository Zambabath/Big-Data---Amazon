import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


timing_files = {
    "LanceDB User": "Lance.parquet",
    "LanceDB Product": "Lance_products.parquet",
    "Chroma Product": "Chroma_products.parquet",
    "Chroma User": "Chroma_users.parquet",
    "Qdrant Product": "Qdrant_product.parquet",
    "Qdrant User": "Qdrant_users.parquet",
}

results = []


for engine_name, file_path in timing_files.items():
    if os.path.exists(file_path):
        try:
            
            df = pd.read_parquet(file_path)
            
            
            time_cols = [c for c in df.columns if 'time' in c.lower() and 'seconds' in c.lower()]
            
            if time_cols:
                query_time = df[time_cols[0]].iloc[0]
                results.append({
                    "Engine": engine_name,
                    "Time (Seconds)": query_time
                })
            else:
                print(f"Warning: No valid timing column found in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"Warning: {file_path} not found.")


if results:
    comparison_df = pd.DataFrame(results).sort_values(by="Time (Seconds)")
    

    comparison_df.to_csv("vector_db_comparison.csv", index=False)
    print("\n--- Comparison DataFrame ---")
    print(comparison_df)
    print("\nSuccess: 'vector_db_comparison.csv' has been generated.")


    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(comparison_df)))

    bars = plt.bar(comparison_df["Engine"], comparison_df["Time (Seconds)"], color=colors, edgecolor='black')


    plt.ylabel("Query Time (Seconds)", fontsize=12)
    plt.xlabel("Vector Database / Engine", fontsize=12)
    plt.title("Search Performance Comparison: Query Latency", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=15)


    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}s', 
                 va='bottom', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    

    plt.savefig("vector_db_comparison.png", dpi=300)
    print("Success: 'vector_db_comparison.png' has been saved.")
    
    plt.show()

else:
    print("\nERROR: No data was extracted. Ensure the .parquet files are in the directory.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

timing_files = {
    "LanceDB User": "Lance.parquet",
    "LanceDB Product": "Lance_products.parquet",
    "Chroma Product": "Chroma_products.parquet",
    "Chroma User": "Chroma_users.parquet",
    "Qdrant Product": "Qdrant_product.parquet",
    "Qdrant User": "Qdrant_users.parquet",
}

results = []

for engine_name, file_path in timing_files.items():
    if os.path.exists(file_path):
        try:
            df = pd.read_parquet(file_path)
            time_cols = [c for c in df.columns if 'time' in c.lower() and 'seconds' in c.lower()]
            
            if time_cols:
                query_time = df[time_cols[0]].iloc[0]
                results.append({
                    "Sub-Engine": engine_name,
                    "Base Engine": engine_name.split(' ')[0],
                    "Time (Seconds)": query_time
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

if results:
    detail_df = pd.DataFrame(results)
    
    summed_df = detail_df.groupby("Base Engine")["Time (Seconds)"].sum().reset_index()
    summed_df = summed_df.sort_values(by="Time (Seconds)")

    summed_df.to_csv("vector_db_total_sums.csv", index=False)
    print("\n--- Total Execution Time per Database ---")
    print(summed_df)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.7, len(summed_df)))
    
    bars = plt.bar(summed_df["Base Engine"], summed_df["Time (Seconds)"], color=colors, edgecolor='black')

    plt.ylabel("Total Latency (Seconds)", fontsize=12)
    plt.xlabel("Database Engine", fontsize=12)
    plt.title("Total Performance Cost (User + Product Queries)", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}s', 
                 va='bottom', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("vector_db_total_comparison.png", dpi=300)
    print("\nSuccess: 'vector_db_total_comparison.png' and 'vector_db_total_sums.csv' saved.")
    plt.show()

else:
    print("\nERROR: No data was extracted from the parquet files.")