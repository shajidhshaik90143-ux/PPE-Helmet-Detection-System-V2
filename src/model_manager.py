from pathlib import Path
from huggingface_hub import hf_hub_download


def ensure_model(repo_id: str, filename: str, local_path: str) -> str:
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 100_000:
        return str(destination)

    print(f"Downloading PPE model: {repo_id}/{filename}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(destination.parent),
        local_dir_use_symlinks=False,
    )

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != destination.resolve():
        destination.write_bytes(downloaded_path.read_bytes())

    print(f"Model ready: {destination}")
    return str(destination)
