import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import PROCESSED_DIR, MODEL_DIR

class EmbeddingManager:
    def __init__(self):
        # Load Model (Small, fast, and accurate)
        print("🔄 Loading AI Model (all-mpnet-base-v2)...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load Data
        self.pr_path = os.path.join(PROCESSED_DIR, "pr_clean.csv")
        self.issue_path = os.path.join(PROCESSED_DIR, "issue_clean.csv")
        
        self.df_prs = pd.read_csv(self.pr_path)
        self.df_issues = pd.read_csv(self.issue_path)
        
        # Handle missing values
        self.df_prs['text'] = self.df_prs['text'].fillna("")
        self.df_issues['text'] = self.df_issues['text'].fillna("")

    def generate_embeddings(self):
        print("\n🧠 Generating Embeddings...")
        
        # 1. Encode PRs
        print(f"   Processing {len(self.df_prs)} PRs...")
        self.pr_embeddings = self.model.encode(
            self.df_prs['text'].tolist(), 
            batch_size=32, 
            show_progress_bar=True, 
            normalize_embeddings=True # Important for Cosine Similarity
        )
        
        # 2. Encode Issues
        print(f"   Processing {len(self.df_issues)} Issues...")
        self.issue_embeddings = self.model.encode(
            self.df_issues['text'].tolist(), 
            batch_size=32, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        print(f"✅ Embeddings Generated!")
        print(f"   PR Shape: {self.pr_embeddings.shape}")
        print(f"   Issue Shape: {self.issue_embeddings.shape}")

    def build_faiss_index(self):
        print("\n🔎 Building FAISS Index...")
        
        # Dimension of embeddings (768 for mpnet-base)
        d = self.issue_embeddings.shape[1]
        
        # IndexFlatIP = Inner Product (Cosine Similarity since vectors are normalized)
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.issue_embeddings)
        
        print(f"✅ Index Built with {self.index.ntotal} vectors")

    def save_artifacts(self):
        print("\n💾 Saving Models...")
        
        # Save Embeddings (numpy format)
        np.save(os.path.join(MODEL_DIR, "pr_embeddings.npy"), self.pr_embeddings)
        np.save(os.path.join(MODEL_DIR, "issue_embeddings.npy"), self.issue_embeddings)
        
        # Save FAISS Index
        faiss.write_index(self.index, os.path.join(MODEL_DIR, "faiss_index.bin"))
        
        print("✅ Saved: pr_embeddings.npy, issue_embeddings.npy, faiss_index.bin")

    def test_retrieval(self):
        print("\n🧪 Testing Retrieval (Top 3 Matches for first 5 PRs)...")
        
        # Search: query, k=number of results
        D, I = self.index.search(self.pr_embeddings[:5], k=3)
        
        for i in range(5):
            print(f"\nPR #{self.df_prs.iloc[i]['pr_id']}: {self.df_prs.iloc[i]['title'][:50]}...")
            print("   Matches:")
            for rank, idx in enumerate(I[i]):
                score = D[i][rank]
                issue_title = self.df_issues.iloc[idx]['title']
                print(f"     {rank+1}. [{score:.2f}] {issue_title[:60]}")

if __name__ == "__main__":
    manager = EmbeddingManager()
    manager.generate_embeddings()
    manager.build_faiss_index()
    manager.save_artifacts()
    manager.test_retrieval()
