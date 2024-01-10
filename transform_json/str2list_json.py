import json

# 定义一个函数，将JSON文件中的每个描述转换为只包含该描述的列表
def convert_json_descriptions_to_lists(file_path):
    # 读取原始JSON文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 将每个描述转换为列表
    converted_data = {key: [value] for key, value in data.items()}

    return converted_data

# 原始JSON文件的路径
#original_file_path = '/Users/chenyubin/Desktop/no_emo/github/ViT1/captions/generated_captions1.json'  # 替换为你的文件路径
original_file_path= '/data/df/test_captions.json'


# 转换数据
converted_data = convert_json_descriptions_to_lists(original_file_path)

# 将转换后的数据保存到新的JSON文件
new_file_path = '/metrics/data_list/RES.json'  # 替换为你希望保存的新文件路径
with open(new_file_path, 'w') as file:
    json.dump(converted_data, file, indent=4)
