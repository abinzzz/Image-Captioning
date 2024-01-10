import json
import re

"""将输出文件生成为给定测试集的json文件格式"""


# 加载 JSON 文件
file_path = '/captions/generated_captions.json'

with open(file_path, 'r') as file:
    original_data = json.load(file)

# 根据您的示例更新句子格式化函数
def format_sentence_v3(sentence):
    # 移除 '<start>' 和 '<end>' 标记
    words = [word for word in sentence.split() if word not in ['<start>', '<end>']]
    # 将单词列表连接成字符串，并修正标点符号位置
    sentence = ' '.join(words).replace(' ,', ',').replace(' .', '.')
    # 句号后的字母大写
    sentence = re.sub(r'\. ([a-z])', lambda x: '. ' + x.group(1).upper(), sentence)
    # 句子首字母大写
    sentence = sentence[0].upper() + sentence[1:]
    return sentence

# 对原始 JSON 文件中的每个句子应用新的格式化函数
for key in original_data.keys():
    if isinstance(original_data[key], str):
        original_data[key] = format_sentence_v3(original_data[key])
    elif isinstance(original_data[key], list):
        original_data[key] = [format_sentence_v3(sentence) for sentence in original_data[key]]
# 将重新格式化的数据保存到新的 JSON 文件中
formatted_file_path = '/captions/generated_captions1.json'
with open(formatted_file_path, 'w') as file:
    json.dump(original_data, file, indent=4)
