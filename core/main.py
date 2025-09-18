import os
import base64
from openai import OpenAI
import json
import json_repair
from io import BytesIO
import time
from cnocr import CnOcr
ocr = CnOcr(det_model_fp= "ch_PP-OCRv4_det_infer.onnx")

client = OpenAI(
    api_key='',#your api key,
    base_url=''
)

def encode_image_to_base64(img):
    """Encode a PIL.Image object to base64 string."""
    try:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

schema = {
    "advertiser": "",
    "agency": "",
    "contract_num": "",
    "flight_from": "",
    "flight_to": "",
    "gross_amount": "",
    "line_item": [
        {
            "channel": "",
            "program_desc": "",
            "program_end_date": "",
            "program_start_date": "",
            "sub_amount": ""
        }
    ],
    "product": "",
    "property": "",
    "tv_address": ""
}


wrong ={
    "table":{
        "key1":["v1-1","v1-2"],
        "key2":["v2-1","v2-2"],
        "key3":["v3-1","v3-2"],
    }
 }
correct = {
        "table":[
            {
                "key1":"v1-1",
                "key2":"v2-1",
                "key3":"v3-1",
            },
            {
                "key1": "v1-2",
                "key2": "v2-2",
                "key3":"v3-2",
            }
        ]
}


def is_empty(item):
    # If it is a dictionary, recursively check all key-value pairs
    if isinstance(item, dict):
        # If all values in the dictionary are empty, return True
        return all(is_empty(value) for value in item.values())

    # If it is a list, check each element
    elif isinstance(item, list):
        # If all elements in the list are empty, return True
        return all(is_empty(element) for element in item)

    # For other types (such as strings, None, etc.), directly check if empty
    return item in [None, '', {}, [], set()]

def merge_jsons(json_list):
    def merge_dict(d1, d2):
        """Recursively merge two dictionaries"""
        for key in d2:
            if key in d1:
                # If both are lists, merge list contents
                if isinstance(d1[key], list) and isinstance(d2[key], list):
                    d1[key] = [item for item in d1[key] if not is_empty(item)]
                    d2[key] = [item for item in d2[key] if not is_empty(item)]
                    d1[key].extend(d2[key])
                # If both are dicts, recursively merge
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    merge_dict(d1[key], d2[key])
                # If one is list and the other is dict, handle specially
                elif isinstance(d1[key], list) and isinstance(d2[key], dict):
                    # Check if the dict has non-empty values
                    if any(d2[key].values()):  # Check if any value is non-empty
                        d1[key].append(d2[key])  # Append non-empty dict to list
                    d1[key] = [item for item in d1[key] if not is_empty(item)]
                elif isinstance(d1[key], dict) and isinstance(d2[key], list):
                    # If d1 is dict and d2 is list
                    if any(d1[key].values()):
                        d1[key] = [d1[key]]  # Convert dict to list
                        d1[key].extend(d2[key])  # Merge list contents
                    d1[key] = [item for item in d1[key] if not is_empty(item)]
                # For other types, keep non-empty values, prioritize d1
                else:
                    if d1[key] not in [None, '', {}, [], set()]:  # If d1 value is non-empty
                        continue  # Keep d1 value, skip merge
                    elif d2[key] not in [None, '', {}, [], set()]:  # If d2 value is non-empty
                        d1[key] = d2[key]  # Keep d2 value
                    else:
                        continue  # If both empty, skip
            else:
                # If key does not exist in d1, directly add
                d1[key] = d2[key]
        return d1

    # Initialize merged result
    merged_result = {}
    for js in json_list:
        merged_result = merge_dict(merged_result, js)
    return json.dumps(merged_result, ensure_ascii=False, indent=4)


def gen_json(messages,folder_path,i):
    # print("messages:",messages)
    response = client.chat.completions.create(
        model="", # model name
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        # max_tokens=4096
    )
    # Parse the returned JSON data
    try:
        # print("Output:",response.choices[0].message.content)
        response_content = response.choices[0].message.content
        new_json = json_repair.loads(response_content)

    except json.JSONDecodeError:

        print(f"{folder_path} page {i + 1} error:")
        new_json = {}
    return new_json


def save_merged_json_as_txt(merged_json, folder_path):
    """Save the merged JSON content as a .txt file in the results directory."""
    parent_dir, folder_name = os.path.split(folder_path)  # Get the last-level folder name
    results_dir = os.path.join(os.path.dirname(parent_dir), "doubao_ocr_noposition_form_results")  # Construct results dir path

    os.makedirs(results_dir, exist_ok=True)  # Ensure results dir exists

    output_txt_path = os.path.join(results_dir, f"{folder_name}.txt")  # Generate save path

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(merged_json)

    print(f"Saved merged JSON to: {output_txt_path}")

def process_images_in_folder(folder_path):
    """Process all images in a folder and generate a text file with results."""
    images_base64 = []
    # Initialize empty JSON data list
    json_list = []
    image_paths = []
    print("Processing:", folder_path)
    start_time = time.time()
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)


    for i, image_data in enumerate(image_paths):
        # print("process:",image_paths)
        try:
            # cnocr
            result = ocr.ocr(image_data)
            # Extract text and position
            cnocr_result = [{'text': item['text'], 'position': item['position']} for item in result]
            cnocr_result = str(cnocr_result)
            print("OCR extraction completed")

            position_schema = {"text": "recognized text", "position": "[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]"}



            messages_select_schema = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in contract recognition and JSON formatting."
                        "Your task is to organize the OCR extracted content of the provided image according to the given schema, keeping the schema hierarchy unchanged. Output one overall JSON object following the schema format and ensure its validity."
                        # f"Each key should contain text and position information, format as {position_schema}." # If needed, output position info
                        "You can use the provided OCR content and position info for reasoning. For example, 'position': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] defines a rectangular region with four vertices. If the vertical centers ((y1+y2)/2) are equal, contents belong to the same row; if the horizontal centers ((x1+x2)/2) are equal, contents belong to the same column."
                    )
                },

                {
                    "role": "system",
                    "content": f"Full schema:\n{schema}\nOCR extracted content: {cnocr_result}"
                },

                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Task background: The current image is one page from a scanned contract.\n"
                                "Task requirements:\n"
                                "1. Based on the image content, remove unnecessary parts from the full schema.\n"
                                "2. Ensure the trimmed schema maintains the same hierarchy as the full schema.\n"
                                "3. Do not add extra key-value pairs to the schema.\n"
                                "4. JSON objects must be generated row by row. Each row should be a separate JSON block, containing key-value pairs for each column. Do not merge all column data into one array. For example, correct format: {correct}, incorrect format: {wrong}.\n"
                            )
                        }
                    ]
                }
            ]

            new_json = gen_json(messages_select_schema, folder_path, i)
            print(new_json)

            json_list.append(new_json)
        except Exception as e:
            print({str(e)})
            with open('ocr_error_log.txt', 'a') as log_file:
                log_file.write(f"folder {folder_path}\n")
                log_file.write(f"Error with image {image_data} (index {i}): {str(e)}\n")
            continue  # Skip current loop, continue with next file

    # Input all JSON data again into model for merging
    if json_list:
        # Merge JSONs
        merged_json = merge_jsons(json_list)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.6f} seconds")
        average_inference_time = inference_time / len(json_list)
        print(f"Average inference time: {average_inference_time:.6f} seconds")
        print(merged_json)
        save_merged_json_as_txt(merged_json, folder_path)


def process_directory(input_dir, label_root):
    """Process all folders in the input directory and generate corresponding text files."""
    processed_folders = []


    for root, dirs, files in os.walk(input_dir):
        # print(root, dirs, files)
        # Skip empty directories
        if not files:
            continue

        # Check if the current folder contains images
        images_exist = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)
        if images_exist:
            # Determine the output text path
            relative_path = os.path.relpath(root, input_dir)

            # Process images in the folder
            process_images_in_folder(root)

            processed_folders.append(root)


input_dir = "../datasets/img_data"  # Replace with your input directory path
output_dir = "../datasets/results"  # Replace with your output directory path

process_directory(input_dir, output_dir)
