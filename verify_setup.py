import sys
import importlib

packages = [
    "github",
    "pandas",
    "numpy",
    "sentence_transformers",
    "faiss",
    "xgboost",
    "sklearn",
    "networkx",
    "rank_bm25",
    "flask"
]

print("=" * 60)
print("🔍 VERIFYING DEPENDENCIES")
print("=" * 60)
print()

failed = []
success_count = 0

for package in packages:
    try:
        importlib.import_module(package)
        print(f"✅ {package}")
        success_count += 1
    except ImportError as e:
        print(f"❌ {package} - NOT INSTALLED")
        failed.append(package)

print()
print("=" * 60)
if failed:
    print(f"⚠️  MISSING PACKAGES: {', '.join(failed)}")
    print(f"✅ INSTALLED: {success_count}/{len(packages)}")
    print()
    print("Try running:")
    print(f"  pip install {' '.join(failed)}")
    sys.exit(1)
else:
    print(f"✅ ALL {success_count} PACKAGES INSTALLED!")
    print("✅ ENVIRONMENT IS READY!")
    print("=" * 60)
