"""Shared helpers for the repository's warm-start last-item protocol."""


def build_last_item_examples(
    history_data,
    test_data,
    user_col,
    item_col,
    time_col,
    candidate_items=None,
):
    """Return eligible users, known histories and one raw last target per user."""
    if candidate_items is None:
        candidate_items = set(history_data[item_col].unique().tolist())
    else:
        candidate_items = set(candidate_items)

    histories_before_test = (
        history_data.sort_values([user_col, time_col], kind='mergesort')
        .groupby(user_col)[item_col]
        .apply(list)
        .to_dict()
    )
    users, histories, targets = [], [], []
    for user_id, group in test_data.groupby(user_col, sort=True):
        test_items = group.sort_values(time_col, kind='mergesort')[item_col].tolist()
        target = test_items[-1]
        history = [
            item for item in histories_before_test.get(user_id, []) + test_items[:-1]
            if item in candidate_items
        ]
        if not history or target not in candidate_items:
            continue
        users.append(user_id)
        histories.append(history)
        targets.append([target])
    return users, histories, targets


def load_movielens(data_dir=None):
    from pathlib import Path
    import pandas as pd

    path = (Path(data_dir) if data_dir is not None else
            Path(__file__).resolve().parents[1] / 'data' / 'info') / 'ratings.dat'
    if path.exists():
        return pd.read_csv(path, sep='::', engine='python',
                           names=['userid', 'movieid', 'rating', 'timestamp'])

    try:
        import certifi
        import ssl
        import urllib.request
        from polara import get_movielens_data
        original_urlopen = urllib.request.urlopen
        def secure_urlopen(url, *args, **kwargs):
            kwargs.setdefault('context', ssl.create_default_context(cafile=certifi.where()))
            return original_urlopen(url, *args, **kwargs)
        urllib.request.urlopen = secure_urlopen
        try:
            data = get_movielens_data(include_time=True)
        finally:
            urllib.request.urlopen = original_urlopen
    except Exception as exc:
        path = (Path(data_dir) if data_dir is not None else
                Path(__file__).resolve().parents[1] / 'data' / 'info') / 'ratings.dat'
        print(f"Polara unavailable ({type(exc).__name__}: {exc}); loading {path}")
        data = pd.read_csv(path, sep='::', engine='python',
                           names=['userid', 'movieid', 'rating', 'timestamp'])
    return data
