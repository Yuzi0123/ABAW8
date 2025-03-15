# --------------- openai_inference.py ---------------
import os
import base64
import json
from openai import OpenAI
from PIL import Image
from io import BytesIO

# 配置参数
OPENAI_API_KEY = "your_openai_api_key"
ANNOTATIONS_FILE = "data/RAF-DB/anotations/annotations.json"
PROMPT_FILE = "prompt/image_label_explain.txt"

def load_prompt():
    with open(PROMPT_FILE, "r") as f:
        return f.read()

def load_annotations(ann_file):
    try:
        with open(ann_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading annotations {ann_file}: {e}")
        return {}

def process_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Image processing error: {e}")
        raise

def get_output_path(ann_file, model_suffix):
    dir_name = os.path.dirname(ann_file)
    base_name = os.path.basename(ann_file)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}_{model_suffix}{ext}"
    return os.path.join(dir_name, new_name)

def openai_inference():
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt_template = load_prompt()

    # 加载标注数据
    original_ann = load_annotations(ANNOTATIONS_FILE)
    if not original_ann:
        return
    
    # 创建结果副本
    result_ann = {}
    
    for item in original_ann:
        img_path = item.get("image_path", "")
        if not img_path or not os.path.exists(img_path):
            print(f"Invalid image path: {img_path}")
            continue
        
        # 处理图像
        try:
            base64_image = process_image(img_path)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            continue

        # 准备请求
        original_label = item.get("label", "")
        prompt_text = prompt_template.replace("<label>", original_label)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_tokens=1000,
                timeout=30
            )
            
            # 保存完整响应作为新标签
            content = response.choices[0].message.content
            result_ann[img_path] = {
                "image_path": img_path,
                "label": content
            }
            
        except Exception as e:
            print(f"API Error for {img_path}: {e}")

    # 保存结果
    output_path = get_output_path(ANNOTATIONS_FILE, "gpt")
    try:
        with open(output_path, "w") as f:
            json.dump(list(result_ann.values()), f, indent=2)
        print(f"Saved results to {output_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")

if __name__ == "__main__":
    openai_inference()