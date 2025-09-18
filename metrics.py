from typing import List, Dict, Any
from difflib import SequenceMatcher


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def extract_named_lists(json_obj: Dict) -> Dict[str, List]:
    """
    Recursively search all lists corresponding to keys in JSON structure,
    and return {list_name: list_content}.
    """
    named_lists = {}

    def recurse(obj: Any, parent_key: str = ""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                recurse(value, key)
        elif isinstance(obj, list) and parent_key:
            named_lists[parent_key] = obj  # Only record lists corresponding to key names

    recurse(json_obj)
    return named_lists


def calculate_line_item_f1(predictions: Dict, ground_truth: Dict) -> float:
    tp, fp_fn = 0, 0

    # Extract lists with the same key names from prediction and ground truth
    pred_lists = extract_named_lists(predictions)
    print("pred_lists:", pred_lists)
    gt_lists = extract_named_lists(ground_truth)
    print("gt_lists:", gt_lists)

    common_keys = set(pred_lists.keys()) & set(gt_lists.keys())  # Only compare lists with the same key names

    for key in common_keys:
        pred_list, gt_list = pred_lists[key], gt_lists[key]

        # Compare only the innermost list contents
        for pred_item in pred_list:
            if not isinstance(pred_item, dict):
                continue

            found_match = False

            for idx, gt_item in enumerate(gt_list):
                if not isinstance(gt_item, dict):
                    continue

                # Key matching
                if pred_item.keys() == gt_item.keys():
                    match = True
                    for field in pred_item.keys():
                        if pred_item[field] == gt_item[field]:
                            continue
                        elif isinstance(pred_item[field], str) and isinstance(gt_item[field], str):
                            # Compute string similarity
                            if similarity(pred_item[field], gt_item[field]) >= 1:
                                continue
                            else:
                                match = False
                                break
                        else:
                            match = False
                            break

                    if match:
                        tp += 1
                        found_match = True
                        gt_list.pop(idx)  # Remove matched item to reduce computation
                        print("Matched line:", pred_item, gt_item)
                        break

            if not found_match:
                fp_fn += 1

    if tp + fp_fn == 0:
        return 0
    return tp / (tp + fp_fn / 2)


def flatten(data: dict):
    """
    Convert Dictionary into Non-nested Dictionary
    Example:
        input(dict)
            {
                "menu": [
                    {"name" : ["cake"], "count" : ["2"]},
                    {"name" : ["juice"], "count" : ["1"]},
                ]
            }
        output(list)
            [
                ("menu.name", "cake"),
                ("menu.count", "2"),
                ("menu.name", "juice"),
                ("menu.count", "1"),
            ]
    """
    flatten_data = list()

    def _flatten(value, key=""):
        if type(value) is dict:
            for child_key, child_value in value.items():
                _flatten(child_value, f"{key}.{child_key}" if key else child_key)
        elif type(value) is list:
            for value_item in value:
                _flatten(value_item, key)
        else:
            flatten_data.append((key, value))

    _flatten(data)
    return flatten_data


def normalize_dict(data):
    """
    Sort by value, while iterating over elements if data is a list
    """
    if not data:
        return {}

    if isinstance(data, dict):
        new_data = dict()
        for key in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = normalize_dict(data[key])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new_data[key] = value

    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            new_data = []
            for item in data:
                item = normalize_dict(item)
                if item:
                    new_data.append(item)
        else:
            new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
    else:
        new_data = [str(data).strip()]

    return new_data


def cal_f1(preds: List[dict], answers: List[dict]):
    """
    Calculate global F1 accuracy score (field-level, micro-averaged)
    by counting all true positives, false negatives and false positives.
    """
    # print("preds:", preds)
    # print("answers:", answers)
    total_tp, total_fn_or_fp = 0, 0
    for pred, answer in zip(preds, answers):

        pred, answer = flatten(normalize_dict(pred)), flatten(normalize_dict(answer))
        for field in pred:
            if field in answer:
                total_tp += 1
                answer.remove(field)
            else:
                found_match = False
                for gt_field in answer:
                    if isinstance(field[1], str) and isinstance(gt_field[1], str):
                        if similarity(field[1], gt_field[1]) >= 1:
                            total_tp += 1
                            answer.remove(gt_field)
                            found_match = True
                            break
                if not found_match:
                    total_fn_or_fp += 1

        total_fn_or_fp += len(answer)
        if total_tp + total_fn_or_fp == 0:
            cal_f1 = 0
        else:
            cal_f1 = total_tp / (total_tp + total_fn_or_fp / 2)
    return cal_f1
