from collections.abc import Hashable, Iterable, Mapping, Sequence


BUCKET_NAMES = ("tail", "middle", "head")


def _single_target(target) -> Hashable:
    if isinstance(target, (str, bytes)) or not isinstance(target, Iterable):
        return target

    values = target.tolist() if hasattr(target, "tolist") else list(target)
    if not isinstance(values, list):
        return values
    if len(values) != 1:
        raise ValueError("each test case must contain exactly one target item")
    return values[0]


def evaluate_bucketed_hr(
    targets: Sequence,
    predictions: Sequence,
    bucket_by_item: Mapping[Hashable, str],
    ks: Iterable[int],
) -> dict[str, dict]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")

    metric_ks = list(ks)
    if not metric_ks:
        raise ValueError("ks must not be empty")
    if len(metric_ks) != len(set(metric_ks)):
        raise ValueError("ks must not contain duplicates")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in metric_ks):
        raise ValueError("every k must be a positive integer")

    unknown_buckets = set(bucket_by_item.values()) - set(BUCKET_NAMES)
    if unknown_buckets:
        raise ValueError(f"unknown bucket names: {sorted(unknown_buckets)}")

    hits = {
        bucket: {k: 0 for k in metric_ks}
        for bucket in BUCKET_NAMES
    }
    num_cases = {bucket: 0 for bucket in BUCKET_NAMES}

    for raw_target, raw_predictions in zip(targets, predictions):
        target = _single_target(raw_target)
        if target not in bucket_by_item:
            raise ValueError(f"target item {target!r} is missing from popularity buckets")

        bucket = bucket_by_item[target]
        recommendations = (
            raw_predictions.tolist()
            if hasattr(raw_predictions, "tolist")
            else list(raw_predictions)
        )
        num_cases[bucket] += 1
        for k in metric_ks:
            hits[bucket][k] += int(target in recommendations[:k])

    return {
        bucket: {
            "num_cases": num_cases[bucket],
            "hr": {
                k: hits[bucket][k] / num_cases[bucket]
                if num_cases[bucket]
                else 0.0
                for k in metric_ks
            },
        }
        for bucket in BUCKET_NAMES
    }


def evaluate_bucketed_coverage(
    predictions: Sequence,
    bucket_by_item: Mapping[Hashable, str],
    ks: Iterable[int],
    candidate_items: Iterable[Hashable],
) -> dict[str, dict]:
    metric_ks = list(ks)
    if not metric_ks:
        raise ValueError("ks must not be empty")
    if len(metric_ks) != len(set(metric_ks)):
        raise ValueError("ks must not contain duplicates")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in metric_ks):
        raise ValueError("every k must be a positive integer")

    unknown_buckets = set(bucket_by_item.values()) - set(BUCKET_NAMES)
    if unknown_buckets:
        raise ValueError(f"unknown bucket names: {sorted(unknown_buckets)}")

    candidates = set(candidate_items)
    if candidates - set(bucket_by_item):
        raise ValueError("every candidate item must have a popularity bucket")

    catalog_by_bucket = {
        bucket: {item for item in candidates if bucket_by_item[item] == bucket}
        for bucket in BUCKET_NAMES
    }
    recommended = {
        bucket: {k: set() for k in metric_ks}
        for bucket in BUCKET_NAMES
    }

    for raw_predictions in predictions:
        recommendations = (
            raw_predictions.tolist()
            if hasattr(raw_predictions, "tolist")
            else list(raw_predictions)
        )
        for k in metric_ks:
            for item in recommendations[:k]:
                if item in candidates:
                    recommended[bucket_by_item[item]][k].add(item)

    return {
        bucket: {
            "coverage": {
                k: len(recommended[bucket][k]) / len(catalog_by_bucket[bucket])
                if catalog_by_bucket[bucket]
                else 0.0
                for k in metric_ks
            },
        }
        for bucket in BUCKET_NAMES
    }


def print_bucketed_hr(results: Mapping[str, dict]) -> None:
    ks = list(results["tail"]["hr"])
    header = f"{'bucket':<8} {'num_cases':<12}" + "".join(
        f" {'HR@' + str(k):<12}" for k in ks
    )
    print("\nPopularity bucket results:")
    print(header)
    for bucket in BUCKET_NAMES:
        values = results[bucket]
        row = f"{bucket:<8} {values['num_cases']:<12}"
        row += "".join(f" {values['hr'][k]:<12.6f}" for k in ks)
        print(row)
