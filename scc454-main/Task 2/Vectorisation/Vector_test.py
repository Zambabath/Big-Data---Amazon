import os
import pandas as pd
import numpy as np
import pickle
import scipy.sparse
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import Normalizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

#bucnh of classes for the different methods
#some use pkl files cause they gave 0.0000 otherwise
class TFIDFSearch:
    def __init__(self, ids_df, matrix_path, model_path="tfidf_product.pkl"):
        self.id_col = '_id'
        self.df = ids_df.copy()
        
        if os.path.exists(matrix_path):
            self.matrix = scipy.sparse.load_npz(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix not found at {matrix_path}")
            
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    def search(self, query, top_k=10):
        q_vec = self.vectorizer.transform([str(query)])
        scores = self.matrix.dot(q_vec.T).toarray().flatten()
        return self.df.iloc[np.argsort(-scores)[:top_k]][self.id_col].astype(str).tolist()

class HashingPCASearch:
    def __init__(self, df, pca_path="pca_model.pkl"):
        self.id_col = '_id' 
        self.df = df.copy()
        self.matrix = np.stack(self.df['vector'].values)
        
        if os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                self.pca = pickle.load(f)
        else:
            raise FileNotFoundError(f"PCA model not found at {pca_path}")
        
        self.hasher = HashingVectorizer(n_features=1000, alternate_sign=False)
        self.normalizer = Normalizer()
        self.matrix = self.normalizer.transform(self.matrix)

    def search(self, query, top_k=10):
        q_h = self.hasher.transform([str(query)])
        q_pca = self.pca.transform(q_h.toarray())
        q_final = self.normalizer.transform(q_pca)
        
        scores = np.dot(self.matrix, q_final.T).flatten()
        return self.df.iloc[np.argsort(-scores)[:top_k]][self.id_col].astype(str).tolist()

class Word2VecSearch:
    def __init__(self, df, model_path="word2vec_product.model", weights_path="word2vec_product_weights.pkl"):
        self.id_col = '_id'
        self.df = df.copy()
        self.normalizer = Normalizer()
        
        self.matrix = np.stack(self.df['vector'].values)
        self.matrix = self.normalizer.transform(self.matrix)
        
        if os.path.exists(model_path):
            self.model = Word2Vec.load(model_path)
        else:
            raise FileNotFoundError(f"Word2Vec model not found at {model_path}")
            
        if os.path.exists(weights_path):
            with open(weights_path, "rb") as f:
                self.idf_dict = pickle.load(f)
        else:
            raise FileNotFoundError(f"Weights dictionary not found at {weights_path}")

    def search(self, query, top_k=10):
        q_tokens = simple_preprocess(str(query))
        
        valid_words = [word for word in q_tokens if word in self.model.wv and word in self.idf_dict]
        if not valid_words: 
            return []
            
        vecs = [self.model.wv[word] * self.idf_dict[word] for word in valid_words]
        q_vec = self.normalizer.transform(np.mean(vecs, axis=0).reshape(1, -1))
        
        scores = np.dot(self.matrix, q_vec.T).flatten()
        return self.df.iloc[np.argsort(-scores)[:top_k]][self.id_col].astype(str).tolist()
    
class SemanticSearch:
    def __init__(self, df, model_name='all-MiniLM-L6-v2'):
        self.id_col = '_id'
        self.df = df.copy()
        self.model = SentenceTransformer(model_name)
        self.matrix = np.stack(df['vector'].values)

    def search(self, query, top_k=10):
        query_vec = self.model.encode([str(query)], convert_to_tensor=True).cpu()
        scores = util.cos_sim(query_vec, self.matrix).numpy().flatten()
        return self.df.iloc[np.argsort(-scores)[:top_k]][self.id_col].astype(str).tolist()


if __name__ == "__main__":
    #get all the data
    paths = {
        "TFIDF_IDS": r"D:\454\master_product_ids_TFIDF.parquet",
        "SKLEARN": r"D:\454\product_vectors_incremental_pca.parquet",
        "SENTENCE": r"D:\454\master_product_vectors_sentence.parquet",
        "W2V": r"D:\454\master_product_vectors_word2vec.parquet"
    }
    
    tfidf_matrix_path = r"D:\454\master_product_vectors_TFIDF.npz"

    loaded_dfs = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: File not found.")
            continue
            
        df = pd.read_parquet(path)
        
        if 'id' in df.columns and '_id' not in df.columns:
            df = df.rename(columns={'id': '_id'})
        elif '_id' not in df.columns:
            df = df.reset_index().rename(columns={df.index.name if df.index.name else 'index': '_id'})
            
        df['_id'] = df['_id'].astype(str).str.strip()
        loaded_dfs[name] = df

    #initialise
    engines = {}
    #tfidf broke many times and this worked
    if "TFIDF_IDS" in loaded_dfs and os.path.exists(tfidf_matrix_path): 
        engines["TF-IDF"] = TFIDFSearch(loaded_dfs["TFIDF_IDS"], tfidf_matrix_path)
        
    if "SENTENCE" in loaded_dfs: engines["Semantic"] = SemanticSearch(loaded_dfs["SENTENCE"])
    if "W2V" in loaded_dfs: engines["Word2Vec"] = Word2VecSearch(loaded_dfs["W2V"])
    if "SKLEARN" in loaded_dfs: engines["Hashing+iPCA"] = HashingPCASearch(loaded_dfs["SKLEARN"])

    test_queries = [
        ("wide tooth sandalwood comb", "B088BDFDHX"),
        ("Detoxifying Charcoal Cleanser", "B076WQZGPM"),
        ("Plunger Bars", "B07NGFDN6G"),
        ("Headrang", "B06XJZ7955"),
        ("Loose Face Powder", "B07FVZVQKV"),
        ("Nomade", "B07ZJW55Z5"),
        ("Eyelash Growth Serum", "B07WFSQXL5"),
        ("Brazilian Body Wave", "B07MH4Z7J7"),
        ("Bow Headbands", "B08M5DXZDJ"),
        ("Chiffon Scrunchies", "B07R5S9WGC"),
        ("Nail Art Rhinestones", "B07PQX1S3C"),
        ("purifier", "B076WQZGPM"),
    ]

    results_rr = {name: [] for name in engines.keys()}

    #just to make sure it is working
    print("Calculating MRR")
    for query, target in test_queries:
        target = str(target).strip()
        for name, engine in engines.items():
            try:
                res = engine.search(query)
                rank = res.index(target) + 1 if target in res else 0
                results_rr[name].append(1 / rank if rank > 0 else 0)
            except Exception as e:
                print(f"Error in {name} search for query '{query}': {e}")
                results_rr[name].append(0)

    mrr_values = {name: np.mean(rr) for name, rr in results_rr.items()}
    sorted_mrr = dict(sorted(mrr_values.items(), key=lambda item: item[1], reverse=True))
    
    print("Final MRR")
    for name, score in sorted_mrr.items():
        print(f"{name:15}: {score:.4f}")

    pd.Series(sorted_mrr).to_csv("MRR_scores.csv")

    if sorted_mrr:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_mrr.keys(), sorted_mrr.values(), color='skyblue', edgecolor='navy')
        plt.title("Search Performance Comparison (MRR)")
        plt.ylabel("MRR Score")
        plt.ylim(0, 1.1)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')
        
        plt.savefig("vector_test_graph.png")
        plt.show()