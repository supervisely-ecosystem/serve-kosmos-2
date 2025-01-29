from huggingface_hub import snapshot_download
import os


current_dir = os.getcwd()
repo_id = "microsoft/kosmos-2-patch14-224"
model_name = repo_id.split("/")[1]
snapshot_download(repo_id=repo_id, local_dir=f"{current_dir}/{model_name}")
