from pycocoevalcap.eval import eval
import json


"""
gts是程序得到的
res是人工标注的
"""

with open('/Users/chenyubin/Desktop/no_emo/github/ViT1/metrics/data_list/GTS.json', 'r') as f:
    gts = json.load(f)
with open('/Users/chenyubin/Desktop/no_emo/github/ViT1/metrics/data_list/RES.json', 'r') as f:
    res = json.load(f)

if __name__ == '__main__':
    mp = eval(gts,res)
    print(mp)