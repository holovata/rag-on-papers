import os
import requests

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# Directory to save downloaded articles
DOWNLOAD_PATH = os.path.join(BASE_DIR, 'data/downloaded_papers')

def clear_download_directory(save_dir):
    """Clear the download folder before downloading new papers."""
    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            file_path = os.path.join(save_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error while deleting file {file_path}: {e}")
    else:
        os.makedirs(save_dir)

def download_arxiv_papers(articles, save_dir=DOWNLOAD_PATH):
    """Download arXiv papers by their ID."""
    clear_download_directory(save_dir)
    for article in articles:
        paper_id = article["id"]
        url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        save_path = os.path.join(save_dir, f"{paper_id}.pdf")

        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"Paper {paper_id} successfully downloaded and saved to {save_path}")
            else:
                print(f"Failed to download paper {paper_id}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error while downloading paper {paper_id}: {e}")
