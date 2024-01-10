import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
#rom pycocotools.coco import COCO
import json

class dfDataset(data.Dataset):
    """COCO自定义数据集继承torch.utils.data.DataLoader。"""
    def __init__(self, root, json_file, vocab, transform=None):
        """设置图像、标题和词汇包装的路径

        Args:
            root:  存放图像文件的目录路径
            json: 包含 COCO 数据集注释的 JSON 文件的路径
            vocab:  一个词汇表对象，用于处理文本数据（如标题）
            transform: 可选参数，用于进行图像转换处理
        """
        self.root = root
        with open(json_file, 'r') as file:
            self.df_data = json.load(file)
        self.ids = list(self.df_data.keys())#描述的id
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """根据提供的索引 index 获取数据集中的一个数据对(图像和标题)"""
        """
        anns:
        {48: {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}, 
         67: {'image_id': 116100, 'id': 67, 'caption': 'A panoramic view of a kitchen and all of its appliances.'}, 
         126: {'image_id': 318556, 'id': 126, 'caption': 'A blue and white bathroom with butterfly themed wall tiles.'}, 
         148: {'image_id': 116100, 'id': 148, 'caption': 'A panoramic photo of a kitchen and dining room'}}
         """
        # coco = self.coco
        vocab = self.vocab
        # ann_id = self.ids[index]#单一描述的id
        # caption = coco.anns[ann_id]['caption']#描述id对应的描述
        # img_id = coco.anns[ann_id]['image_id']#描述id对应的图像id
        img_id = self.ids[index]
        caption = self.df_data[img_id]
        #path = coco.loadImgs(img_id)[0]['file_name']#描述id对应的图像名称
        image = Image.open(os.path.join(self.root, img_id)).convert('RGB')#打开root + path路径的图像
        if self.transform is not None:
            image = self.transform(image)

        """
        这里的image不再是图像，而是tensor：
        tensor([[[-1.8268, -1.8097, -1.8097,  ..., -0.5767, -0.5767, -0.4568],
         [-1.8097, -1.7925, -1.7925,  ..., -0.5082, -0.3712, -0.4226],
         [-1.8097, -1.7925, -1.7925,  ..., -0.4911, -0.4397, -0.4739],
         ...,
         [-2.0837, -2.1008, -2.0837,  ...,  0.1254,  0.0912,  0.0227],
         [-2.0837, -2.1179, -2.0837,  ...,  0.1254,  0.0741, -0.0116],
         [-2.0837, -2.1179, -2.0837,  ...,  0.0912,  0.0398, -0.0458]]])
        """

        # 将标题(字符串)转换为字id.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())#使用 nltk 库对标题文本进行分词处理，并将所有文本转换为小写。
        """tokens = ['some', 'bikers', 'are', 'sitting', 'on', 'their', 'motorcycles', 'outside']"""

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        """caption = [1, 252, 880, 53, 225, 40, 103, 1556, 490, 2]"""
        target = torch.Tensor(caption)#转化为张量

        return image, target #返回图像及其所对应的描述单词

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """从元组列表中创建小批量张量(图片，标题)

    处理不同长度的标题，将它们转换为一个统一的批处理格式

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # 按标题长度(降序)对数据列表排序
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    #images是有16个元素的元组，每个元组都是3，224，224的tensor

    # 合并图像(从3D张量元组到4D张量)。
    #16 x 3，224，224 -->  变为16，3，224，224
    images = torch.stack(images, 0)

    # 合并标题(从1D张量元组到2D张量元组)。

    lengths = [len(cap) for cap in captions]#每个元素是标题的长度
    targets = torch.zeros(len(captions), max(lengths)).long()#标题数量x标题最大长度的全0矩阵
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

        """
        targets: 1开头，2结尾，后面填充长度
        tensor([[   1,    4,  959,  255,   78,    4, 2054,    7,    4, 6390,  825, 1353,
          112,    4, 9704,   19,    2],
        [   1,    4,   81,   14, 8354, 2442,  112,   33, 3951,   14,    4, 5606,
          324,   19,    2,    0,    0],
        [   1,    4,  170,   22,    4,  360,  171,   40,    4,  756,  102,    4,
         5295,   19,    2,    0,    0],
        [   1,    4,   15,   28,   29,  113,   65,  131,  317,   22, 1011,    7,
          329,    2,    0,    0,    0]]}
        """

        #lengths=描述中有多少个词
    return images, targets, lengths

def get_loader(root, json_file, vocab, transform, batch_size, shuffle, num_workers):
    """创建并返回一个用于加载 COCO 数据集的 torch.utils.data.DataLoader 实例"""
    # COCO caption dataset
    df = dfDataset(root=root,
                       json_file=json_file,
                       vocab=vocab,
                       transform=transform)




    #print(df)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=df,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
