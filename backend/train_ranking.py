import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import ndcg_score, accuracy_score, f1_score
import os
import pickle
from config import PROCESSED_DIR, MODEL_DIR

class RankingModelTrainer:
    def __init__(self):
        self.feature_path = os.path.join(PROCESSED_DIR, "features.csv")
        self.df = pd.read_csv(self.feature_path)
        print(f"📊 Loaded Features: {len(self.df)} rows")

    def prepare_data(self):
        print("\n🧹 Preparing Training Data...")
        
        # Define Features and Target
        self.features = ['cosine_sim', 'bm25_sim', 'title_overlap', 'file_sim']
        self.target = 'label'
        
        X = self.df[self.features]
        y = self.df[self.target]
        groups = self.df['pr_id'] # We group by PR to keep rankings together
        
        # Split Data (GroupShuffleSplit ensures all candidates for one PR stay in same set)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        self.X_train = X.iloc[train_idx]
        self.X_test = X.iloc[test_idx]
        self.y_train = y.iloc[train_idx]
        self.y_test = y.iloc[test_idx]
        
        # Create Query Groups (required for LambdaMART ranking)
        # This tells XGBoost: "The first 10 rows belong to PR #1, the next 10 to PR #2..."
        self.train_groups = self.df.iloc[train_idx].groupby('pr_id').size().to_numpy()
        self.test_groups = self.df.iloc[test_idx].groupby('pr_id').size().to_numpy()
        
        print(f"✅ Train Size: {len(self.X_train)} (Groups: {len(self.train_groups)})")
        print(f"✅ Test Size:  {len(self.X_test)} (Groups: {len(self.test_groups)})")

    def train_model(self):
        print("\n🤖 Training Ranking Model (XGBoost - LambdaMART)...")
        
        # Create DMatrix (special XGBoost data structure)
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtrain.set_group(self.train_groups)
        
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.dtest.set_group(self.test_groups)
        
        # Parameters optimized for ranking
        params = {
            'objective': 'rank:ndcg',      # LambdaMART Ranking Objective
            'eval_metric': 'ndcg@5',       # Optimize for top-5 results
            'max_depth': 4,                # Tree depth (prevent overfitting)
            'eta': 0.1,                    # Learning rate
            'subsample': 0.8,              # Row sampling
            'colsample_bytree': 0.8,       # Column sampling
            'tree_method': 'hist'          # Histogram-based (faster)
        }
        
        # Train
        self.model = xgb.train(
            params, 
            self.dtrain, 
            num_boost_round=100, 
            evals=[(self.dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        print("✅ Training Complete!")

    def evaluate_model(self):
        print("\n📈 Evaluating Performance...")
        
        # Predict
        y_pred = self.model.predict(self.dtest)
        
        # Convert raw scores to binary predictions (just for simple metrics)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        acc = accuracy_score(self.y_test, y_pred_binary)
        f1 = f1_score(self.y_test, y_pred_binary)
        
        print(f"📊 Test Accuracy: {acc:.4f}")
        print(f"📊 Test F1-Score: {f1:.4f}")
        
        # Feature Importance
        importance = self.model.get_score(importance_type='gain')
        print("\n⭐ Feature Importance:")
        for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {feat}: {score:.2f}")

    def save_model(self):
        print("\n💾 Saving Ranking Model...")
        
        model_path = os.path.join(MODEL_DIR, "ranking_model.json")
        self.model.save_model(model_path)
        
        print(f"✅ Saved to {model_path}")

if __name__ == "__main__":
    trainer = RankingModelTrainer()
    trainer.prepare_data()
    trainer.train_model()
    trainer.evaluate_model()
    trainer.save_model()
