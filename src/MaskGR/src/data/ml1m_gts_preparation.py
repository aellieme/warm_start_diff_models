import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from polara.datasets.movielens import get_movielens_data

# Setup project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sid.sid_generator import RKMeansSIDGenerator
from src.data.data_utils import get_item_texts_ml1m


def load_movielens_1m():
    """
    Load MovieLens-1M ratings and movie metadata using polara.
    
    Returns:
        ratings_df (pd.DataFrame): columns ['user_id', 'item_id', 'rating', 'timestamp']
        movies_df (pd.DataFrame): columns ['movieid', 'title', 'genres']
    """
    print("Loading MovieLens-1M ratings (polara)")
    ratings_df = get_movielens_data(include_time=True)
    ratings_df = ratings_df.rename(columns={'userid': 'user_id', 'movieid': 'item_id'})
    
    print("Loading movie metadata (titles and genres)...")
    _, movies_df = get_movielens_data(get_genres=True, split_genres=False)
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies.")
    return ratings_df, movies_df


def transform_indices(df, user_col='user_id', item_col='item_id'):
    """Map original IDs to contiguous 0..N-1 integers."""
    df = df.copy()
    users = df[user_col].unique()
    items = df[item_col].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {i: j for j, i in enumerate(items)}
    df['user_idx'] = df[user_col].map(user2idx)
    df['item_idx'] = df[item_col].map(item2idx)
    return df, user2idx, item2idx


def prepare_ml1m_gts(
    output_dir='data/ml1m_gts',
    sid_codebook_size=256,
    sid_num_layers=4,
    max_seq_len=50,
    seed=43
):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create output directories
    for subdir in ['training', 'validation', 'adaptation', 'testing']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sids'), exist_ok=True)

    # load data
    df_ratings, movies_df = load_movielens_1m()

    # transform indices
    df_ratings, user2idx, item2idx = transform_indices(df_ratings)
    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"Users: {num_users}, Items: {num_items}, Interactions: {len(df_ratings)}")

    # Build reverse mappings
    idx2user = {v: k for k, v in user2idx.items()}
    idx2orig_item = {v: k for k, v in item2idx.items()}

    # Build item info dict (for future reference)
    item_info = {}
    for _, row in movies_df.iterrows():
        orig_id = row['movieid']
        if orig_id in item2idx:
            idx = item2idx[orig_id]
            item_info[idx] = {
                'original_id': int(orig_id),
                'title': row['title'],
                'genres': row['genres']
            }

    # global temporal sort and split
    df_sorted = df_ratings.sort_values('timestamp').reset_index(drop=True)
    timestamps = df_sorted['timestamp'].values

    q70 = np.quantile(timestamps, 0.70)
    q80 = np.quantile(timestamps, 0.80)
    q90 = np.quantile(timestamps, 0.90)
    print(f"Global quantiles: 70%={q70}, 80%={q80}, 90%={q90}")

    train_df = df_sorted[df_sorted['timestamp'] <= q70].copy()
    adapt_df = df_sorted[(df_sorted['timestamp'] > q80) & (df_sorted['timestamp'] <= q90)].copy()
    test_df  = df_sorted[df_sorted['timestamp'] > q90].copy()

    print(f"Train size: {len(train_df)}")
    print(f"Adapt size: {len(adapt_df)}")
    print(f"Test  size: {len(test_df)}")

    # generate SIDs for all items (as in original MaskGR, no temporal leakage in content)
    print("Generating Semantic IDs for all items...")
    # Prepare text descriptions
    item_texts = {}
    for idx in range(num_items):
        orig_id = idx2orig_item[idx]
        movie_row = movies_df[movies_df['movieid'] == orig_id].iloc[0]
        item_texts[idx] = f"{movie_row['title']} ({movie_row['genres']})"

    # Подготовка текстов для генератора SID (словарь {item_idx: text})
    item_text_map = {i: item_texts[i] for i in range(num_items)}

    sid_gen = RKMeansSIDGenerator(
        num_layers=sid_num_layers,
        codebook_size=sid_codebook_size,
        embedding_model='flan-t5-xxl',
        seed=seed
    )
    all_item_ids = list(range(num_items))
    item_sid_dict = sid_gen.generate_sids(all_item_ids, item_text_map)
    # item_sid_dict = sid_gen.generate_sids(all_item_ids, item_text_map)

    # all_item_ids = list(range(num_items))
    # item_sid_dict = sid_gen.generate_sids(all_item_ids, item_texts)
    print("SID generation complete.")

    # build examples for train, adapt, test
    def build_examples_full_sequence(df_split):
        """Return list of dicts with full sequence (no target extraction)."""
        examples = []
        for user in df_split['user_idx'].unique():
            user_df = df_split[df_split['user_idx'] == user].sort_values('timestamp')
            item_ids = user_df['item_idx'].tolist()
            timestamps = user_df['timestamp'].tolist()
            sids = [item_sid_dict[item] for item in item_ids]
            if len(item_ids) == 0:
                continue
            if len(item_ids) > max_seq_len:
                item_ids = item_ids[-max_seq_len:]
                sids = sids[-max_seq_len:]
                timestamps = timestamps[-max_seq_len:]
            item_text_list = [item_texts[i] for i in item_ids]
            examples.append({
                'user_id': int(user),
                'item_ids': item_ids,
                'sids': sids,
                'timestamps': timestamps,
                'item_text': item_text_list
            })
        return examples

    train_examples = build_examples_full_sequence(train_df)
    adapt_examples = build_examples_full_sequence(adapt_df)

    # build validation examples strictly per TZ
    future_data = df_sorted[df_sorted['timestamp'] > q70].copy()
    val_examples = []
    for user in future_data['user_idx'].unique():
        user_future = future_data[future_data['user_idx'] == user].sort_values('timestamp')
        mask = user_future['timestamp'] <= q80
        if not mask.any():
            continue
        target_idx = mask[mask].index[-1]
        history_df = user_future.loc[:target_idx-1] if target_idx > user_future.index[0] else user_future.iloc[:0]
        target_row = user_future.loc[target_idx]
        item_ids = history_df['item_idx'].tolist() + [target_row['item_idx']]
        timestamps = history_df['timestamp'].tolist() + [target_row['timestamp']]
        sids = [item_sid_dict[item] for item in item_ids]
        if len(item_ids) > max_seq_len:
            item_ids = item_ids[-max_seq_len:]
            sids = sids[-max_seq_len:]
            timestamps = timestamps[-max_seq_len:]
        item_text_list = [item_texts[i] for i in item_ids]
        val_examples.append({
            'user_id': int(user),
            'item_ids': item_ids,
            'sids': sids,
            'timestamps': timestamps,
            'item_text': item_text_list
        })
    print(f"Validation users (strict): {len(val_examples)}")

    # build test examples (all events after q90, target is last)
    test_examples = []
    for user in test_df['user_idx'].unique():
        user_df = test_df[test_df['user_idx'] == user].sort_values('timestamp')
        item_ids = user_df['item_idx'].tolist()
        timestamps = user_df['timestamp'].tolist()
        sids = [item_sid_dict[item] for item in item_ids]
        if len(item_ids) == 0:
            continue
        if len(item_ids) > max_seq_len:
            item_ids = item_ids[-max_seq_len:]
            sids = sids[-max_seq_len:]
            timestamps = timestamps[-max_seq_len:]
        item_text_list = [item_texts[i] for i in item_ids]
        test_examples.append({
            'user_id': int(user),
            'item_ids': item_ids,
            'sids': sids,
            'timestamps': timestamps,
            'item_text': item_text_list
        })
    print(f"Test users: {len(test_examples)}")

    # write TFRecord files
    def serialize_example(user_id, item_id_list, item_text_list):
        feature = {
            'user_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[user_id])),
            'item_id': tf.train.Feature(int64_list=tf.train.Int64List(value=item_id_list)),
            'item_text': tf.train.Feature(bytes_list=tf.train.BytesList(value=item_text_list)),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write_tfrecord(examples, split_name):
        """Запись списка примеров в сжатый TFRecord файл."""
        output_path = os.path.join(output_dir, split_name, 'part_0.tfrecord.gz')
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(output_path, options=options) as writer:
            for ex in examples:
                # Преобразуем тексты в байты
                item_texts_bytes = [text.encode('utf-8') for text in ex['item_text']]
                serialized = serialize_example(
                    ex['user_id'],
                    ex['item_ids'],
                    item_texts_bytes
                )
                writer.write(serialized)
        print(f"Written {len(examples)} examples to {output_path}")

    write_tfrecord(train_examples, 'training')
    write_tfrecord(val_examples,   'validation')
    write_tfrecord(adapt_examples, 'adaptation')
    write_tfrecord(test_examples,  'testing')

    # save metadata
    meta = {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2user': idx2user,
        'idx2orig_item': idx2orig_item,
        'thresholds': {'q70': float(q70), 'q80': float(q80), 'q90': float(q90)},
        'num_users': num_users,
        'num_items': num_items,
        'max_seq_len': max_seq_len,
        'sid_config': {'num_layers': sid_num_layers, 'codebook_size': sid_codebook_size}
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(output_dir, 'item_info.json'), 'w') as f:
        json.dump(item_info, f, indent=2)

    # Save SID mapping
    torch.save(item_sid_dict, os.path.join(output_dir, 'sids', 'flan-t5-xxl_rkmeans_4_256_seed43.pt'))

    print("GTS preparation completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/ml1m_gts')
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()
    prepare_ml1m_gts(
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        seed=args.seed
    )
