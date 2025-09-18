import json
import os
import pandas as pd
from metrics import cal_f1, calculate_line_item_f1

def find_labels_by_filenames(folder_path, jsonl_path, output_excel_path):

    # results list for storing outputs
    results = []

    for filename in os.listdir(folder_path):
        # ensure the file ends with .txt and contains .pdf in the name
        if filename.endswith(".txt") and ".pdf" in filename:
            # process filename, remove .pdf
            new_filename = filename.replace(".pdf", "")
            # get full path
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            # rename file
            os.rename(old_path, new_path)

    # get all txt file names in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # open the jsonl file and parse line by line
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            filename = data["filename"]
            annotations = data["annotations"]

            # change filename from .pdf to .txt
            if filename.endswith('.pdf'):
                filename = filename.replace('.pdf', '.txt')

            # if filename exists in folder, process labels
            if filename in txt_files:
                print(f"Filename: {filename}")

                txt_file_path = os.path.join(folder_path, filename)
                with open(txt_file_path, "r", encoding="utf-8") as txt_file:
                    try:
                        prediction = json.load(txt_file)  # read JSON data from txt file
                    except json.JSONDecodeError:
                        prediction = "Invalid JSON format in the txt file."

                label_json = extract_text(annotations)
                label_json = json.loads(label_json)

                # evaluation functions
                global_f1 = cal_f1([prediction], [label_json])
                line_item_f1 = calculate_line_item_f1(prediction, label_json)
                print("batch_global_f1:", global_f1)
                print("batch_line_item_f1:", line_item_f1)

                # add results to the list
                results.append({"filename": filename, "global_f1": global_f1, "line_item_f1": line_item_f1})

    # save results to Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")


def extract_text(data):
    result = {
        "advertiser": "",
        "agency": "",
        "contract_num": "",
        "flight_from": "",
        "flight_to": "",
        "gross_amount": "",
        "line_item": [],
        "product": "",
        "property": "",
        "tv_address": ""
    }

    for item in data:
        key = item[0]
        values = item[1]

        if key in ["advertiser", "agency", "contract_num", "flight_from", "flight_to", "gross_amount", "product", "property", "tv_address"]:
            if values:
                result[key] = values[0][0].strip()
        elif isinstance(key, list) and key[0] in ["channel", "program_desc", "program_end_date", "program_start_date", "sub_amount"]:
            for line_item in values:
                line_item_dict = {
                    "channel": "",
                    "program_desc": "",
                    "program_end_date": "",
                    "program_start_date": "",
                    "sub_amount": ""
                }
                for i, sub_item in enumerate(line_item):
                    if i < len(key):
                        line_item_dict[key[i]] = sub_item[0].strip()
                result["line_item"].append(line_item_dict)

    return json.dumps(result, ensure_ascii=False, indent=4)


folder_path = "../datasets/results"  # replace with your folder path
jsonl_path = "../datasets/dataset.jsonl"  # replace with your jsonl file path
output_excel_path = "../datasets/results.xlsx"  # output Excel file path
find_labels_by_filenames(folder_path, jsonl_path, output_excel_path)
