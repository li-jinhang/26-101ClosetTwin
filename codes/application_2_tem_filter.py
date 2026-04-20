import json
import os

def update_candidate_list(input_dir, 
                          output_dir, 
                          info_filename="_info_simulate.json", 
                          annotation_filename="_annotation_data.jsonl", 
                          output_filename="_candidate_jpg_manifest.txt"):
    """
    根据用户信息文件中的温度区间，筛选出符合条件的衣物
    :param input_dir: 输入数据所在文件夹 (包含 info 和 annotation 数据)
    :param output_dir: 清单输出的文件夹
    :param info_filename: 当日信息文件名
    :param annotation_filename: 图片标注数据文件名
    :param output_filename: 筛选后的候选清单输出文件名
    """
    info_path = os.path.join(input_dir, info_filename)
    annotation_path = os.path.join(input_dir, annotation_filename)
    
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
        print(f"❌ 读取当日信息文件失败 ({info_path}): {e}")
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
        print(f"❌ 读取衣物编码文件失败 ({annotation_path}): {e}")
        return

    # 3. 创建输出目录并更新清单文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建文件夹: {output_dir}")

    output_full_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_full_path, 'w', encoding='utf-8') as f:
            for filename in candidate_filenames:
                f.write(filename + '\n')
        print(f"✅ 成功更新候选清单！共有 {len(candidate_filenames)} 件衣物入选。保存至: {output_full_path}")
    except Exception as e:
        print(f"❌ 写入清单文件失败: {e}")

# 供自身独立运行测试的代码
if __name__ == "__main__":
    update_candidate_list(
        input_dir='./user_simulate', 
        output_dir='./user_simulate'
    )