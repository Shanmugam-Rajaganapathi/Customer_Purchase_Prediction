from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Authenticate with your HF token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Space repo details
space_repo_id = "ShanRaja/Customer-Purchase-Prediction"
space_repo_type = "space"
local_folder = "tourism_project/deployment"  # Folder containing app.py, Dockerfile, etc.

# Check if Space exists, create if not
try:
    api.repo_info(repo_id=space_repo_id, repo_type=space_repo_type)
    print(f"Space '{space_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{space_repo_id}' not found. Creating new Space...")
    create_repo(repo_id=space_repo_id, repo_type=space_repo_type, private=False)
    print(f"Space '{space_repo_id}' created successfully!")

# Upload folder contents to the Space
api.upload_folder(
    folder_path=local_folder,
    repo_id=space_repo_id,
    repo_type=space_repo_type
)

print("Streamlit app files uploaded to Hugging Face Space successfully!")
