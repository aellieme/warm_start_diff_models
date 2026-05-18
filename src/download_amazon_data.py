import os, gzip, shutil, urllib.request

CATEGORIES = ["Baby", "Beauty", "Sports_and_Outdoors", "Toys_and_Games"]
BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
TARGET_DIR = "data/amazon"
os.makedirs(TARGET_DIR, exist_ok=True)

def download_and_extract_raw(category):
    file_name = f"reviews_{category}_5.json.gz"
    local_gz = os.path.join(TARGET_DIR, file_name)
    local_json = local_gz.replace(".gz", "")

    if os.path.exists(local_json):
        print(f"{local_json} already exists, skipping.")
        return local_json

    url = f"{BASE_URL}/{file_name}"
    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, local_gz)
    except Exception as e:
        print(f"Failed: {e}")
        if os.path.exists(local_gz):
            os.remove(local_gz)
        return None

    print("Extracting...")
    with gzip.open(local_gz, 'rb') as f_in:
        with open(local_json, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(local_gz)
    print(f"Saved to {local_json}")
    return local_json

for cat in CATEGORIES:
    download_and_extract_raw(cat)
print("All done.")