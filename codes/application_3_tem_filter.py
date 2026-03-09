import json
import os

def update_candidate_list(info_path, annotation_path, output_dir, output_filename):
    # 1. 读取当日信息获取温度区间
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
            day_info = info_data[0] if isinstance(info_data, list) else info_data
            temp_min = day_info['temp_range']['min']
            temp_max = day_info['temp_range']['max']
            print(f"当日日期: {day_info.get('date', '未知')}")
            print(f"目标温度区间: {temp_min}℃ - {temp_max}℃")
    except Exception as e:
        print(f"读取当日信息文件失败 ({info_path}): {e}")
        return

    # 2. 筛选符合温度条件的候选衣物
    candidate_filenames = []
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                rec_temp = item.get('recommended_temperature')
                
                if rec_temp is not None and temp_min <= rec_temp <= temp_max:
                    candidate_filenames.append(item['image_filename'])
    except Exception as e:
        print(f"读取衣物编码文件失败 ({annotation_path}): {e}")
        return

    # 3. 创建输出目录并更新清单文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建文件夹: {output_dir}")

    # 分开设置输出路径与文件名
    output_full_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_full_path, 'w', encoding='utf-8') as f:
            for filename in candidate_filenames:
                f.write(filename + '\n')
        print(f"成功更新候选清单！共有 {len(candidate_filenames)} 件衣物入选。")
        print(f"清单文件保存至: {output_full_path}")
    except Exception as e:
        print(f"写入清单文件失败: {e}")

if __name__ == "__main__":
    # --- 路径配置区 ---
    # 输入文件夹路径（如果是当前目录可以设为 '.'）
    INPUT_DIR = './user_simulate' 
    # 输出文件夹路径
    OUTPUT_DIR = './user_simulate'

    # --- 文件名配置区 ---
    INFO_NAME = '_info_simulate.json'
    DATA_NAME = '_annotation_data.jsonl'
    MANIFEST_NAME = '_candidate_jpg_manifest.txt'

    # --- 逻辑执行 ---
    # 拼接完整路径
    info_full_path = os.path.join(INPUT_DIR, INFO_NAME)
    data_full_path = os.path.join(INPUT_DIR, DATA_NAME)

    update_candidate_list(
        info_full_path, 
        data_full_path, 
        OUTPUT_DIR, 
        MANIFEST_NAME
    )