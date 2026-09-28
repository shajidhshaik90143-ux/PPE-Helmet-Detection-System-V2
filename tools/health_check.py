from pathlib import Path
import importlib


packages = ["ultralytics", "cv2", "numpy", "pandas", "yaml", "streamlit", "huggingface_hub"]

print("PPE V2 health check")
print("=" * 30)

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "installed")
        print(f"[OK] {package}: {version}")
    except Exception as exc:
        print(f"[FAIL] {package}: {exc}")

print("\nProject files:")
for p in [
    Path("app.py"),
    Path("dashboard.py"),
    Path("setup_model.py"),
    Path("config/settings.yaml"),
]:
    print("[OK]" if p.exists() else "[MISSING]", p)

print("\nRun: python setup_model.py")
