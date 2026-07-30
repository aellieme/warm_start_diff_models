from collections import Counter
from collections.abc import Hashable, Iterable, Mapping


BUCKET_NAMES = ("tail", "middle", "head")


def count_item_interactions(item_ids: Iterable[Hashable]) -> dict[Hashable, int]:
    return dict(Counter(item_ids))


def build_popularity_buckets(
    item_counts: Mapping[Hashable, int],
    candidate_items: Iterable[Hashable],
) -> dict[Hashable, str]:
    candidates = list(candidate_items)
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate_items must not contain duplicates")
    if not candidates:
        return {}

    invalid_counts = {
        item: count
        for item, count in item_counts.items()
        if not isinstance(count, int) or isinstance(count, bool) or count < 0
    }
    if invalid_counts:
        raise ValueError("item counts must be non-negative integers")

    items_by_frequency: dict[int, list[Hashable]] = {}
    for item in candidates:
        frequency = item_counts.get(item, 0)
        items_by_frequency.setdefault(frequency, []).append(item)

    frequency_groups = [
        items_by_frequency[frequency] for frequency in sorted(items_by_frequency)
    ]
    group_count = len(frequency_groups)

    if group_count == 1:
        return {item: "middle" for item in candidates}
    if group_count == 2:
        return {
            item: bucket
            for group, bucket in zip(frequency_groups, ("tail", "head"))
            for item in group
        }

    cumulative_sizes = []
    total = 0
    for group in frequency_groups:
        total += len(group)
        cumulative_sizes.append(total)

    first_cut = min(
        range(1, group_count - 1),
        key=lambda index: abs(cumulative_sizes[index - 1] - len(candidates) / 3),
    )
    second_cut = min(
        range(first_cut + 1, group_count),
        key=lambda index: abs(cumulative_sizes[index - 1] - 2 * len(candidates) / 3),
    )

    bucket_by_item = {}
    for group_index, group in enumerate(frequency_groups):
        if group_index < first_cut:
            bucket = "tail"
        elif group_index < second_cut:
            bucket = "middle"
        else:
            bucket = "head"
        for item in group:
            bucket_by_item[item] = bucket

    return bucket_by_item
