import numpy as np
from PIL import Image
import json_repair
from openai import OpenAI
import base64
import json
import cv2
from PIL import Image
from typing import Dict, Any, List, Optional
import os
from io import BytesIO

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key='9c54b618-2806-4546-b6b1-0a33783a91e8',
    base_url='https://ark.cn-beijing.volces.com/api/v3'
)


def encode_image_to_base64(image: Image.Image) -> str:
    """将 PIL.Image 对象转换为 Base64 字符串。"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


position_schema = {"text": "识别出的文本", "position": "[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]",
                   "page": "代表页数的数字"}


def call_doubao_vision_model(image: Image.Image, info: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用豆包视觉大模型，并返回补充的 JSON 数据。
    `schema` 仅包含缺失 JSON 块的结构。
    """
    base64_str = encode_image_to_base64(image)
    img_base64 = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名擅账单识别和JSON格式整理的专家。"
                f"你的任务是根据提供的图片和其原始信息，对图片内容进行识别，并保证层级结构不变，每个键对应的值应包含文本和位置信息和页数信息，格式为{position_schema}。"
                "可以根据其上下文的位置信息判断，例如position: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] 表示一个矩形区域的四个顶点的坐标，分别为左上角、右上角、右下角和左下角，用于定义文本块或表格单元格在图像中的位置。如果两个矩形区域的中心纵坐标(即(y1+y2)/2)相同，则说明两个内容属于同一行；如果两个矩形区域的中心横坐标(即(x1+x2)/2)相同，则说明两个内容属于同一列。"
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"其原始信息为{info}\n"
                        "如果原始信息有位置和页数信息，则保持不变；如果原始信息有缺失，则进行补充，并且根据其周围字段的信息补充合理的位置信息和页数"
                    )
                },
                img_base64
            ]
        }
    ]
    response = client.chat.completions.create(
        model="ep-20241213105550-sfsq6",  # 豆包模型
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=4096
    )
    # 解析返回的 JSON 数据
    try:
        print("每页输出：", response.choices[0].message.content)
        response_content = response.choices[0].message.content
        new_json = json_repair.loads(response_content)
    except json.JSONDecodeError:
        print(f"模型输出错误")
        new_json = {}
    return new_json


def extract_missing_fields(data: Any, parent_key: str = "") -> List[str]:
    """
    递归遍历 JSON，查找缺失字段，并提取其完整路径，包括数组索引。
    """
    missing_fields = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            if value == "":  # 发现缺失字段
                missing_fields.append(current_key)  # 记录完整路径
            missing_fields.extend(extract_missing_fields(value, current_key))

    elif isinstance(data, list):
        for index, item in enumerate(data):
            current_key = f"{parent_key}.{index}" if parent_key else str(index)
            missing_fields.extend(extract_missing_fields(item, current_key))

    return missing_fields


def remove_nested_and_duplicate_fields(fields: List[str]) -> List[str]:
    """
    去除嵌套和重复的字段。
    """
    # 去除重复字段
    unique_fields = list(set(fields))

    # 去除嵌套字段
    final_fields = []
    for field in unique_fields:
        is_nested = False
        for other_field in unique_fields:
            if field != other_field and field.startswith(other_field + "."):
                is_nested = True
                break
        if not is_nested:
            final_fields.append(field)

    return final_fields


def get_parent_info(data: Dict[str, Any], parent_key: str) -> Dict[str, Any]:
    """
    根据父级键提取完整的 text、position、page 信息。
    """
    keys = parent_key.split(".")
    temp = data
    for key in keys:
        temp = temp.get(key, {})
    return temp if isinstance(temp, dict) else {}


from typing import Dict, Any, Optional, List


def find_position_in_node(node: Any) -> List[List[int]]:
    """
    递归获取节点的所有坐标信息，并计算包含的最大范围。
    """
    positions = []

    if isinstance(node, dict):
        for value in node.values():
            positions.extend(find_position_in_node(value))
    elif isinstance(node, list):
        for item in node:
            positions.extend(find_position_in_node(item))
    elif isinstance(node, str):  # 可能的坐标信息
        if node.startswith("[[") and node.endswith("]]"):
            try:
                pos = eval(node)  # 解析坐标字符串
                if isinstance(pos, list) and all(isinstance(p, list) and len(p) == 2 for p in pos):
                    positions.extend(pos)
            except:
                pass
    return positions


def get_max_bounding_box(positions: List[List[int]]) -> Optional[List[List[int]]]:
    """
    计算所有点的最大包含范围（bounding box）。
    """
    if not positions:
        return None

    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] for p in positions)
    y_max = max(p[1] for p in positions)

    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def get_sibling_positions(data: Dict[str, Any], parent_key: str) -> Optional[Dict[str, Any]]:
    """
    获取父级key的前后兄弟节点的位置信息，支持兄弟节点是字典或列表。
    """
    keys = parent_key.split(".")
    temp = data

    # 递归查找父级节点
    for key in keys[:-1]:
        if isinstance(temp, dict):
            temp = temp.get(key, {})
        elif isinstance(temp, list):
            try:
                temp = temp[int(key)]  # 处理列表索引
            except (ValueError, IndexError):
                return None
        else:
            return None

    # 获取当前节点的兄弟节点（同级节点）
    current_key = keys[-1]
    siblings = {}

    if isinstance(temp, dict):
        fields = list(temp.keys())
        current_index = fields.index(current_key) if current_key in fields else -1

        # 查找前一个兄弟节点
        if current_index > 0:
            prev_key = fields[current_index - 1]
            prev_info = temp.get(prev_key, {})
            prev_positions = find_position_in_node(prev_info)
            siblings['prev'] = {"key": prev_key, "position": get_max_bounding_box(prev_positions)}

        # 查找后一个兄弟节点
        if current_index < len(fields) - 1:
            next_key = fields[current_index + 1]
            next_info = temp.get(next_key, {})
            next_positions = find_position_in_node(next_info)
            siblings['next'] = {"key": next_key, "position": get_max_bounding_box(next_positions)}

    elif isinstance(temp, list):
        try:
            current_index = int(current_key)

            # 查找前一个兄弟节点
            if current_index > 0:
                prev_info = temp[current_index - 1]
                prev_positions = find_position_in_node(prev_info)
                siblings['prev'] = {"index": current_index - 1, "position": get_max_bounding_box(prev_positions)}
            # 查找后一个兄弟节点
            if current_index < len(temp) - 1:
                next_info = temp[current_index + 1]
                next_positions = find_position_in_node(next_info)
                siblings['next'] = {"index": current_index + 1, "position": get_max_bounding_box(next_positions)}
        except (ValueError, IndexError):
            return None

    return siblings


def crop_image_based_on_siblings(image_path: str, parent_info: Dict[str, Any], siblings: Dict[str, Any],
                                 save_dir: str) -> Optional[Image.Image]:
    """
    根据父级key的前后兄弟节点位置裁剪图像。
    """
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        print(f"无效的图片路径: {image_path}")
        return None

    # 解析父级坐标
    parent_pos = []
    for key, value in parent_info.items():
        if isinstance(value, dict) and "position" in value:
            try:
                pos = json.loads(str(value["position"]).replace("'", "\""))  # 处理字符串格式
                if isinstance(pos, list):  # 确保是列表
                    parent_pos.append(pos)
            except json.JSONDecodeError:
                print(f"无法解析 position: {value['position']}")  # 处理异常情况
    parent_pos = [point for sublist in parent_pos for point in sublist]
    # parent_pos = json.loads(parent_info["position"].replace("'", "\"")) if "position" in parent_info else []
    # 解析兄弟坐标
    prev_pos = json.loads(str(siblings['prev']["position"]).replace("'", "\"")) if 'prev' in siblings and siblings[
        'prev'] else []
    next_pos = json.loads(str(siblings['next']["position"]).replace("'", "\"")) if 'next' in siblings and siblings[
        'next'] else []

    # 计算裁剪区域
    all_pos = parent_pos + prev_pos + next_pos
    # print("parent_pos: ", parent_pos)
    # print("prev_pos: ", prev_pos)
    if all_pos:
        # print("all----------------")
        x_coords = [p[0] for p in all_pos]
        y_coords = [p[1] for p in all_pos]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # 如果没有上一个兄弟，裁剪到图片最顶部
        if not prev_pos:
            y_min = 0

        # 如果没有下一个兄弟，裁剪到图片最底部
        if not next_pos:
            h, w = cv2.imread(image_path).shape[:2]
            y_max = h

        # 添加5px边界缓冲
        padding = 5
        print("padding--------------")
        h, w = cv2.imread(image_path).shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        print("pppppp-----------------")

        # 执行裁剪
        image = cv2.imread(image_path)
        cropped = image[y_min:y_max, x_min:x_max]

        # 将裁剪后的图片保存到指定目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"cropped_{os.path.basename(image_path)}")
            cv2.imwrite(save_path, cropped)
            print(f"裁剪图片已保存: {save_path}")
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), save_path
    return None





def update_json(data: Dict[str, Any], parent_key: str, new_data: Dict[str, Any]):
    """
    更新 JSON 数据，将识别出的内容填充回原始 JSON。
    """
    keys = parent_key.split(".")
    temp = data
    for key in keys[:-1]:
        temp = temp[key]
    temp[keys[-1]].update(new_data)


def fill_missing_data(image_paths: List[str], merged_json: Dict[str, Any], save_dir: str) -> Dict[str, Any]:
    """
    填充缺失数据，基于父级key的前后兄弟节点位置裁剪图像，并将裁剪后的图片和父级key信息输入模型。
    """
    missing_fields = extract_missing_fields(merged_json)
    print("原始 missing fields:", missing_fields)

    # 去除嵌套和重复字段
    final_fields = remove_nested_and_duplicate_fields(missing_fields)
    print("去重后的 missing fields:", final_fields)

    for parent_key in final_fields:
        try:
            parent_info = get_parent_info(merged_json, parent_key)
            print("parent_info:", parent_info)
            siblings = get_sibling_positions(merged_json, parent_key)
            print("siblings:", siblings)
            if parent_info:
                page = int(parent_info.get("page", 1))  # 默认第一页
                image_path = image_paths[page - 1]  # 页码从 1 开始，索引从 0 开始
                if not isinstance(image_path, str):
                    print(f"无效的图片路径: {image_path}")
                    continue
                print("image_path:", image_path)
                cropped_image, save_path = crop_image_based_on_siblings(image_path, parent_info, siblings, save_dir)
                if cropped_image:
                    print("图片裁剪完成")
                    new_data = call_doubao_vision_model(cropped_image, parent_info)
                    update_json(merged_json, parent_key, new_data)
        except Exception as e:
            print(f"处理字段 {parent_key} 时发生错误: {str(e)}")
    return merged_json
