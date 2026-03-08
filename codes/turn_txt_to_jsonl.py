import os
import json

# --- 配置参数 ---
input_folder_path = './data/clothing-dataset-small-master/train/dress'  # 你的 txt 文件夹路径
output_folder_path = './data/clothing-dataset-small-master/train/dress' # 输出文件夹路径
custom_filename = "_basic_data_dress.jsonl"                             # 自定义文件名

# 拼接完整的输出文件路径
output_jsonl_path = os.path.join(output_folder_path, custom_filename)

def convert_txt_to_jsonl(input_dir, output_file):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 遍历文件夹中所有文件
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        # 加载 JSON 内容
                        content = json.load(f_in)
                        
                        # 自动关联对应的 jpg 文件名
                        content['image_filename'] = filename.replace('.txt', '.jpg')

                        # 将 JSON 对象转换为单行字符串并写入
                        json_line = json.dumps(content, ensure_ascii=False)
                        f_out.write(json_line + '\n')
                        count += 1
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

    print(f"转换完成！共处理 {count} 个文件，已保存至: {output_file}")

# 执行转换
convert_txt_to_jsonl(input_folder_path, output_jsonl_path)