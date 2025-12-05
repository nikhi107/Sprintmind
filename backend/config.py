import os
from dotenv import load_dotenv

# Load token from .env file
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Repositories to analyze (You can change these!)
TARGET_REPOS = [
    "flask/flask",           # Medium size, good for testing
    "requests/requests"      # Stable, clean history
]

# Data Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"✅ Configuration Loaded")
print(f"   Target Repos: {TARGET_REPOS}")
