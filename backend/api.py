from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
import faiss
import os
from sentence_transformers import SentenceTransformer
from config import MODEL_DIR, PROCESSED_DIR
from rank_bm25 import BM25Okapi

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
model_emb = None
index = None
ranking_model = None
risk_model = None
df_prs = None
df_issues = None
pr_embeddings = None
issue_embeddings = None


def load_resources():
    global model_emb, index, ranking_model, risk_model
    global df_prs, df_issues, pr_embeddings, issue_embeddings

    print("⏳ Loading AI Models & Data... (Please wait)")

    # 1. Load Embedding Model
    model_emb = SentenceTransformer("all-mpnet-base-v2")

    # 2. Load DataFrames
    df_prs = pd.read_csv(os.path.join(PROCESSED_DIR, "pr_clean.csv")).fillna("")
    df_issues = pd.read_csv(os.path.join(PROCESSED_DIR, "issue_clean.csv")).fillna("")

    # 3. Load Embeddings
    pr_embeddings = np.load(os.path.join(MODEL_DIR, "pr_embeddings.npy"))
    issue_embeddings = np.load(os.path.join(MODEL_DIR, "issue_embeddings.npy"))

    # 4. Load FAISS Index
    index = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))

    # 5. Load Ranking Model
    ranking_model = xgb.Booster()
    ranking_model.load_model(os.path.join(MODEL_DIR, "ranking_model.json"))

    # 6. Load Risk Model
    risk_model = xgb.Booster()
    risk_model.load_model(os.path.join(MODEL_DIR, "srpnet_model.json"))

    print("✅ All Systems Ready!")


# --- API ENDPOINTS ---

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "online", "prs_loaded": len(df_prs)})


@app.route("/predict_story", methods=["POST"])
def predict_story():
    """
    Input: {"title": "Fix login bug", "body": "..."}
    Output: Ranked list of user stories
    """
    data = request.json
    query_text = f"{data.get('title', '')} {data.get('body', '')}"

    # 1. Generate Embedding
    query_vec = model_emb.encode([query_text], normalize_embeddings=True)

    # 2. Retrieval (FAISS) - Get top 20 candidates
    D, I = index.search(query_vec.astype(np.float32), k=20)
    candidate_indices = I[0]

    # 3. Reranking Feature Engineering
    candidates = []
    for idx in candidate_indices:
        if idx < 0 or idx >= len(df_issues):
            continue

        issue = df_issues.iloc[idx]

        # Cosine similarity between query embedding and issue embedding
        cosine_sim = float(np.dot(query_vec[0], issue_embeddings[idx]))

        # Build feature vector with correct names expected by the model
        feature_names = ["cosine_sim", "bm25_sim", "title_overlap", "file_sim"]
        feature_values = np.array([[cosine_sim, 0.5, 0.5, 0.1]], dtype=float)

        dmatrix = xgb.DMatrix(feature_values, feature_names=feature_names)

        score = float(ranking_model.predict(dmatrix)[0])

        candidates.append(
            {
                "issue_id": int(issue["issue_id"]),
                "title": issue["title"],
                "score": score,
                "cosine": cosine_sim,
            }
        )

    # Sort by XGBoost Score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({"matches": candidates[:5]})


@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    """
    Input: {"title": "...", "files": ["core.py", "test.py"]}
    Output: Risk Score (0-1)
    """
    data = request.json
    files = data.get("files", [])
    title = data.get("title", "")
    body = data.get("body", "")

    # Extract Features
    num_files = len(files)
    desc_len = len(body)
    title_len = len(title)
    files_str = " ".join(files).lower()
    has_tests = 1 if "test" in files_str else 0
    is_critical = 1 if any(x in files_str for x in ["config", "core", "api", "auth"]) else 0

    # Create Vector
    features = np.array(
        [[num_files, desc_len, title_len, has_tests, is_critical]], dtype=float
    )
    dmatrix = xgb.DMatrix(
        features,
        feature_names=[
            "num_files",
            "desc_len",
            "title_len",
            "has_tests",
            "is_critical",
        ],
    )

    # Predict
    risk_score = float(risk_model.predict(dmatrix)[0])

    # Simple threshold logic
    risk_label = "High Risk" if risk_score > 0.5 else "Low Risk"

    return jsonify(
        {
            "risk_score": risk_score,
            "risk_label": risk_label,
            "is_critical": bool(is_critical),
        }
    )


# --- STARTUP ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

