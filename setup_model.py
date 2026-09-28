import yaml
from src.model_manager import ensure_model


with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["model"]

path = ensure_model(
    cfg["repo_id"],
    cfg["filename"],
    cfg["local_path"],
)

print("\nPPE model setup complete.")
print("Model:", path)
print("You can now run: python app.py --source 0")
