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
        history_data.sort_values([user_col, time_col])
        .groupby(user_col)[item_col]
        .apply(list)
        .to_dict()
    )
    users, histories, targets = [], [], []
    for user_id, group in test_data.groupby(user_col, sort=True):
        test_items = group.sort_values(time_col)[item_col].tolist()
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
