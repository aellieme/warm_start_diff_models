# prepare_gts_datasets.py
import os
import sys
import pickle
import pandas as pd
import argparse
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
# from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_tools.warm_start import load_movielens

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'datasets' / 'data'
AMAZON_FILES = {
    'baby': 'reviews_Baby_5.json',
    'beauty': 'reviews_Beauty_5.json',
    'sports': 'reviews_Sports_and_Outdoors_5.json',
    'toys': 'reviews_Toys_and_Games_5.json',
}

def save_dataset_with_gts(dataset_name, df, output_dir=OUTPUT_DIR):
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

    df = df.sort_values('timestamp', kind='mergesort').reset_index(drop=True)

    T_valid = df['timestamp'].quantile(0.7)
    T_test = df['timestamp'].quantile(0.8)

    train_dict = {}
    val_seq_dict = {}
    val_tgt_dict = {}
    test_seq_dict = {}
    test_tgt_dict = {}

    for uid, group in df.groupby('userid'):
        group = group.sort_values('timestamp', kind='mergesort')
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

def load_dataset(dataset, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    if dataset == 'ml-1m':
        return load_movielens(data_dir / 'info')[['userid', 'movieid', 'timestamp']]
    if dataset == 'ml-100k':
        path = data_dir / 'ml-100k/u.data'
        return pd.read_csv(
            path, sep='\t', engine='python',
            header=None, names=['userid', 'movieid', 'rating', 'timestamp'],
            usecols=['userid', 'movieid', 'timestamp'],
        )
    path = data_dir / 'amazon' / AMAZON_FILES[dataset]
    df = pd.read_json(path, lines=True)
    return df[['reviewerID', 'asin', 'unixReviewTime']].rename(
        columns={'reviewerID': 'userid', 'asin': 'movieid', 'unixReviewTime': 'timestamp'}
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare one ADRec dataset from Polara or shared local sources.')
    parser.add_argument('--dataset', required=True, choices=['ml-1m', 'ml-100k', *AMAZON_FILES])
    args = parser.parse_args(argv)
    df = load_dataset(args.dataset)
    save_dataset_with_gts(args.dataset, df)


if __name__ == '__main__':
    main()

    

