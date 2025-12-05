import json
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from config import DATA_DIR, MODEL_DIR

class RiskEngine:
    def __init__(self):
        self.raw_data_path = os.path.join(DATA_DIR, "prs.json")
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"📊 Loaded {len(self.data)} PRs for Risk Analysis")

    def extract_features(self):
        print("\n⚙️ Extracting Risk Features...")
        
        features = []
        labels = []
        
        for pr in self.data:
            # --- FEATURES ---
            
            # 1. Complexity: Number of files changed
            num_files = len(pr.get('files', []))
            
            # 2. Documentation: Length of description
            desc_len = len(pr.get('body', "") or "")
            
            # 3. Title Clarity: Length of title
            title_len = len(pr.get('title', "") or "")
            
            # 4. Testing: Does it touch test files?
            files_str = " ".join(pr.get('files', [])).lower()
            has_tests = 1 if 'test' in files_str else 0
            
            # 5. Scope: Interaction with critical files (e.g., config, core)
            is_critical = 1 if any(x in files_str for x in ['config', 'core', 'api', 'auth']) else 0
            
            features.append([num_files, desc_len, title_len, has_tests, is_critical])
            
            # --- LABEL (GROUND TRUTH) ---
            # If Merged = Safe (0)
            # If Closed (not merged) = Risky (1)
            is_merged = pr.get('merged', False)
            labels.append(0 if is_merged else 1)
            
        self.X = np.array(features)
        self.y = np.array(labels)
        self.feature_names = ['num_files', 'desc_len', 'title_len', 'has_tests', 'is_critical']
        
        print(f"✅ Extracted {len(features)} samples")
        print(f"   High Risk (Rejected): {sum(labels)}")
        print(f"   Low Risk (Merged): {len(labels) - sum(labels)}")

    def train_model(self):
        print("\n🔥 Training SRP-Net (XGBoost)...")
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Train XGBoost Classifier
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model Trained!")
        print(f"📊 Test Accuracy: {acc:.4f}")
        print("\n📋 Detailed Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Risky']))
        
        # Feature Importance
        print("\n⭐ Risk Drivers (Feature Importance):")
        importances = self.model.feature_importances_
        for name, imp in zip(self.feature_names, importances):
            print(f"   - {name}: {imp:.4f}")

    def save_model(self):
        print("\n💾 Saving Risk Model...")
        model_path = os.path.join(MODEL_DIR, "srpnet_model.json")
        self.model.save_model(model_path)
        print(f"✅ Saved to {model_path}")

if __name__ == "__main__":
    engine = RiskEngine()
    engine.extract_features()
    engine.train_model()
    engine.save_model()
