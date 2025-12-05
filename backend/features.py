import pandas as pd
import numpy as np
import os
import faiss
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from config import PROCESSED_DIR, MODEL_DIR

class FeatureEngineer:
    def __init__(self):
        # Load Data
        self.df_prs = pd.read_csv(os.path.join(PROCESSED_DIR, "pr_clean.csv"))
        self.df_issues = pd.read_csv(os.path.join(PROCESSED_DIR, "issue_clean.csv"))
        self.df_labels = pd.read_csv(os.path.join(PROCESSED_DIR, "labels.csv"))
        
        # Load Embeddings
        self.pr_emb = np.load(os.path.join(MODEL_DIR, "pr_embeddings.npy"))
        self.issue_emb = np.load(os.path.join(MODEL_DIR, "issue_embeddings.npy"))
        
        # Load FAISS Index
        self.index = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))
        
        self.scaler = MinMaxScaler()
        
        print("✅ Data loaded for feature engineering")

    def compute_cosine_similarity(self):
        """Dot product of normalized vectors = Cosine Similarity"""
        print("\n🔢 Computing Cosine Similarity...")
        
        cosine_sims = []
        for _, row in tqdm(self.df_labels.iterrows(), total=len(self.df_labels)):
            pr_idx = int(row['pr_id']) - 1  # Adjust for 0-indexing
            issue_idx = int(row['issue_id']) - 1
            
            # Clamp indices to valid range
            if pr_idx < 0 or pr_idx >= len(self.pr_emb):
                cosine_sims.append(0.0)
                continue
            if issue_idx < 0 or issue_idx >= len(self.issue_emb):
                cosine_sims.append(0.0)
                continue
            
            # Dot product (both vectors normalized)
            sim = np.dot(self.pr_emb[pr_idx], self.issue_emb[issue_idx])
            cosine_sims.append(float(sim))
        
        return np.array(cosine_sims)

    def compute_bm25_similarity(self):
        """BM25 ranking (keyword-based)"""
        print("\n📚 Computing BM25 Similarity...")
        
        # Tokenize issue texts
        issue_tokens = [str(text).split() for text in self.df_issues['text']]
        bm25 = BM25Okapi(issue_tokens)
        
        bm25_sims = []
        for _, row in tqdm(self.df_labels.iterrows(), total=len(self.df_labels)):
            try:
                pr_idx = int(row['pr_id']) - 1
                issue_idx = int(row['issue_id']) - 1
                
                if pr_idx < 0 or pr_idx >= len(self.df_prs):
                    bm25_sims.append(0.0)
                    continue
                
                pr_text = str(self.df_prs.iloc[pr_idx]['text']).split()
                
                # Get BM25 scores for all issues
                scores = bm25.get_scores(pr_text)
                
                if issue_idx < 0 or issue_idx >= len(scores):
                    bm25_sims.append(0.0)
                    continue
                
                bm25_sims.append(float(scores[issue_idx]))
            except:
                bm25_sims.append(0.0)
        
        return np.array(bm25_sims)

    def compute_title_overlap(self):
        """Word overlap in titles"""
        print("\n📝 Computing Title Overlap...")
        
        overlaps = []
        for _, row in tqdm(self.df_labels.iterrows(), total=len(self.df_labels)):
            try:
                pr_idx = int(row['pr_id']) - 1
                issue_idx = int(row['issue_id']) - 1
                
                if pr_idx < 0 or pr_idx >= len(self.df_prs):
                    overlaps.append(0.0)
                    continue
                if issue_idx < 0 or issue_idx >= len(self.df_issues):
                    overlaps.append(0.0)
                    continue
                
                pr_title = set(str(self.df_prs.iloc[pr_idx]['title']).lower().split())
                issue_title = set(str(self.df_issues.iloc[issue_idx]['title']).lower().split())
                
                if len(pr_title) == 0 or len(issue_title) == 0:
                    overlaps.append(0.0)
                    continue
                
                overlap = len(pr_title & issue_title) / max(len(pr_title), len(issue_title))
                overlaps.append(float(overlap))
            except:
                overlaps.append(0.0)
        
        return np.array(overlaps)

    def compute_file_similarity(self):
        """File path overlap"""
        print("\n🗂️ Computing File Similarity...")
        
        file_sims = []
        for _, row in tqdm(self.df_labels.iterrows(), total=len(self.df_labels)):
            try:
                pr_idx = int(row['pr_id']) - 1
                
                if pr_idx < 0 or pr_idx >= len(self.df_prs):
                    file_sims.append(0.0)
                    continue
                
                pr_files = set(str(self.df_prs.iloc[pr_idx]['files']).split())
                
                # Issues don't have files, so this is always 0 or low
                # But we keep it for future extensions
                file_sims.append(0.1)  # Small constant
            except:
                file_sims.append(0.0)
        
        return np.array(file_sims)

    def create_features(self):
        print("\n" + "="*60)
        print("🎯 STEP 5: FEATURE ENGINEERING")
        print("="*60)
        
        # Compute all features
        features_dict = {
            'pr_id': self.df_labels['pr_id'].values,
            'issue_id': self.df_labels['issue_id'].values,
            'label': self.df_labels['label'].values,
            'cosine_sim': self.compute_cosine_similarity(),
            'bm25_sim': self.compute_bm25_similarity(),
            'title_overlap': self.compute_title_overlap(),
            'file_sim': self.compute_file_similarity()
        }
        
        df_features = pd.DataFrame(features_dict)
        
        # Normalize features to [0, 1]
        feature_cols = ['cosine_sim', 'bm25_sim', 'title_overlap', 'file_sim']
        df_features[feature_cols] = self.scaler.fit_transform(df_features[feature_cols])
        
        print(f"\n✅ Features Created: {len(df_features)} samples")
        print(f"   Positive (label=1): {(df_features['label'] == 1).sum()}")
        print(f"   Negative (label=0): {(df_features['label'] == 0).sum()}")
        print(f"\n   Features: {feature_cols}")
        
        # Save
        output_path = os.path.join(PROCESSED_DIR, "features.csv")
        df_features.to_csv(output_path, index=False)
        print(f"\n💾 Saved to {output_path}")
        
        return df_features

if __name__ == "__main__":
    engineer = FeatureEngineer()
    features_df = engineer.create_features()
    
    # Show sample
    print("\n📊 Sample Features (first 5 rows):")
    print(features_df.head())
