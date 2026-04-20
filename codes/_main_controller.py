# main_controller.py
from application_1_user_annotation import SupplementaryAnnotator
from application_2_tem_filter import update_candidate_list
from application_3_scene_score_sort import calculate_and_sort_scenarios

def run_clothing_recommendation_pipeline(target_folder):
    print(f"\n{'='*40}")
    print(f"开始处理目标文件夹: {target_folder}")
    print(f"{'='*40}\n")
    
    # --- 步骤 1：执行特征提取与标注 ---
    # 此处你不仅可以修改文件夹，如果模型换了位置也可以通过传入参数修改
    annotator = SupplementaryAnnotator(
        image_dir=target_folder,
        vit_model_path="./mobilevit-small", 
        regressor_path="models/temperature_regressor.pth",
        scenario_regressor_path="models/scenario_regressor.pth"
    )
    annotator.process()
    
    # --- 步骤 2：温度筛选 ---
    update_candidate_list(
        input_dir=target_folder, 
        output_dir=target_folder
    )
    
    # --- 步骤 3：场景相似度排序 ---
    calculate_and_sort_scenarios(
        input_dir=target_folder,
        model_name="all-MiniLM-L6-v2"
    )

if __name__ == "__main__":
    # 假设你有一个新的用户文件夹叫 "new_user_clothes"
    USER_FOLDER = "./user_simulate" 
    run_clothing_recommendation_pipeline(USER_FOLDER)