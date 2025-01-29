from huggingface_hub import snapshot_download
import os

if not os.path.exists("weights"):
    os.mkdir("weights")
repo_id = "microsoft/kosmos-2-patch14-224"
model_name = repo_id.split("/")[1]

snapshot_download(repo_id=repo_id, local_dir=f"weights/{model_name}")
