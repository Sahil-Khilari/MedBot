import os
import gdown


def download_model_checkpoint(gdrive_file_id: str, output_path: str):
    """
    Download model checkpoint from Google Drive if it doesn't already exist.

    Args:
        gdrive_file_id: Google Drive file ID string
        output_path: Local path where the checkpoint will be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        print(f"Downloading model checkpoint to {output_path}...")
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        gdown.download(url, output_path, quiet=False)
        print("Download complete!")
