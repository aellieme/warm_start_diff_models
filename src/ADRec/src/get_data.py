# prepare_gts_datasets.py
import os
import pickle
import pandas as pd
import requests
from tqdm import tqdm
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder
# from huggingface_hub import hf_hub_download

def download_file(url, local_filename):
    """Download a file with a progress bar."""
    # with requests.get(url, stream=True, timeout=(10, 60)) as r:
    with requests.get(url, stream=True, timeout=(100, 560)) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 8192
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size),
                              total=total_size // chunk_size,
                              unit='KB', unit_scale=True,
                              desc=f"Downloading {os.path.basename(local_filename)}"):
                f.write(chunk)
    return local_filename

def save_dataset_with_gts(dataset_name, df, output_dir='../datasets/data/'):
    """
    Process interactions DataFrame with time-split (70% train, 10% val, 20% test)
    and save as dataset.pkl.
    Expects df with columns: userid, movieid, timestamp
    """
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    
    df = df.copy()

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['userid'] = user_enc.fit_transform(df['userid'])
    df['movieid'] = item_enc.fit_transform(df['movieid']) + 1  # shift, 0 = padding

    df = df.sort_values('timestamp').reset_index(drop=True)

    T_valid = df['timestamp'].quantile(0.7)
    T_test = df['timestamp'].quantile(0.8)

    train_dict = {}
    val_seq_dict = {}
    val_tgt_dict = {}
    test_seq_dict = {}
    test_tgt_dict = {}

    for uid, group in df.groupby('userid'):
        group = group.sort_values('timestamp')
        items = group['movieid'].tolist()
        times = group['timestamp'].tolist()

        train_seq = [item for item, ts in zip(items, times) if ts <= T_valid]
        if len(train_seq) > 0:
            train_dict[uid] = train_seq

        val_window = [(item, ts) for item, ts in zip(items, times) if T_valid < ts <= T_test]
        if val_window:
            val_tgt = val_window[-1][0]
            val_hist = [item for item, _ in val_window[:-1]]
            val_seq_dict[uid] = train_seq + val_hist
            val_tgt_dict[uid] = val_tgt

        test_window = [(item, ts) for item, ts in zip(items, times) if ts > T_test]
        if test_window:
            test_tgt = test_window[-1][0]
            test_hist = [item for item, _ in test_window[:-1]]
            full_val_seq = [item for item, _ in val_window] if val_window else []
            test_seq_dict[uid] = train_seq + full_val_seq + test_hist
            test_tgt_dict[uid] = test_tgt

    val_seq_list = [val_seq_dict[uid] for uid in sorted(val_seq_dict.keys())]
    val_tgt_list = [val_tgt_dict[uid] for uid in sorted(val_seq_dict.keys())]
    test_seq_list = [test_seq_dict[uid] for uid in sorted(test_seq_dict.keys())]
    test_tgt_list = [test_tgt_dict[uid] for uid in sorted(test_seq_dict.keys())]

    data_pkl = {
        'train': list(train_dict.values()),
        'val_seq': val_seq_list,
        'val_tgt': val_tgt_list,
        'test_seq': test_seq_list,
        'test_tgt': test_tgt_list,
        'item_count': len(item_enc.classes_),
        'train_dict': train_dict,
        'val_seq_dict': val_seq_dict,
        'val_tgt_dict': val_tgt_dict,
        'test_seq_dict': test_seq_dict,
        'test_tgt_dict': test_tgt_dict,
    }

    output_path = os.path.join(output_dir, dataset_name, 'dataset.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data_pkl, f)

    print(f"Saved {len(data_pkl['train'])} training sequences to {output_path}")
    return output_path

def prepare_ml100k():
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "ml-100k.zip"
    download_file(url, zip_path)
    with ZipFile(zip_path, 'r') as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith('u.data'):
                with zip_ref.open(name) as f:
                    df = pd.read_csv(f, sep='\t', header=None,
                                     usecols=[0,1,3],
                                     names=['userid', 'movieid', 'timestamp'])
                break
        else:
            raise FileNotFoundError("u.data not found")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    save_dataset_with_gts('ml-100k', df)
    os.remove(zip_path)

def prepare_ml1m():
    url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "ml-1m.zip"
    download_file(url, zip_path)
    with ZipFile(zip_path, 'r') as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith('ratings.dat'):
                with zip_ref.open(name) as f:
                    df = pd.read_csv(f, sep='::', header=None,
                                     usecols=[0,1,3],
                                     names=['userid', 'movieid', 'timestamp'],
                                     engine='python')
                break
        else:
            raise FileNotFoundError("ratings.dat not found")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    save_dataset_with_gts('ml-1m', df)
    os.remove(zip_path)
    
def download_from_huggingface(dataset_name, repo_id, filename):
    """Helper to download a dataset file from Hugging Face Hub."""
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_csv(local_path)

def clean_amazon_reviews(df):
    """Keep only necessary columns for our processing."""
    if 'reviewerID' in df.columns and 'asin' in df.columns and 'unixReviewTime' in df.columns:
        return df[['reviewerID', 'asin', 'unixReviewTime']].rename(
            columns={'reviewerID': 'userid', 'asin': 'movieid', 'unixReviewTime': 'timestamp'}
        )
    return None

# ===== Main Execution =====
if __name__ == '__main__':
    # 1. MovieLens 100k
    prepare_ml100k()
    prepare_ml1m() 

    # 2. Amazon subsets (McAuley Lab, 2014) – official source
    amazon_datasets = {
        'baby': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz',
        'beauty': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz',
        'sports': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz',
        'toys': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz',
    }
    # amazon_datasets = {
    #     'beauty': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz',
    #     'sports': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz',
    #     'toys': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz',
    # }

    for name, url in amazon_datasets.items():
        print(f"\nProcessing {name}")
        gz_path = f"{name}.json.gz"
        download_file(url, gz_path)
        df = pd.read_json(gz_path, lines=True, compression='gzip')
        df_clean = clean_amazon_reviews(df)
        if df_clean is not None and len(df_clean) > 0:
            save_dataset_with_gts(name, df_clean)
        os.remove(gz_path)

    

