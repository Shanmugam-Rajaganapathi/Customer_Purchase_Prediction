from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Authenticate
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the space exists
space_id = "ShanRaja/Customer-Purchase-Prediction"
space_repo_type = "space"

try:
    api.repo_info(repo_id=space_id, repo_type=space_repo_type)
    print(f"Space '{space_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{space_id}' not found. Creating new space...")
    create_repo(repo_id=space_id, repo_type=space_repo_type, private=False)
    print(f"Space '{space_id}' created.")

# Upload dataset
dataset_id = "ShanRaja/Customer-Purchase-Prediction"
data_repo_type = "dataset"

api.upload_folder(
    folder_path="tourism_project/data",
    path_in_repo="",             # Upload to root of dataset repo
    repo_id=dataset_id,
    repo_type=data_repo_type
)
print("Dataset uploaded successfully!")
