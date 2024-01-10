# 基于Vision Transformer的图像字幕生成


Transformer是一种强大的模型，利用注意力机制。 <br>
最初由[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 提出，并用于序列到序列任务。<br>

后来[AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf) 提出了视觉变换器ViT，它直接在图像上应用Transformer。<br>

在这个任务中，我们使用预训练的```google/vit-base-patch16-224``` 预训练的ViT作为编码器，以及一个变换器解码器。







<br>

## 一.下载预训练模型
你可以通过这个[链接](https://drive.google.com/file/d/1VNbrIE9oFu12QnS_vQAti4r6bIMUBZsB/view?usp=sharing) 下载预训练的ViT模型 <br>
解压下载的google.zip

```
unzip google.zip
```


<br>

## 二.目录
```
.
├── Gradio.py
├── README.md
├── ViT_example.py
├── __pycache__
│   ├── Gradio.cpython-311.pyc
│   ├── build_vocab.cpython-311.pyc
│   ├── captioning_DIY.cpython-311.pyc
│   └── dataloader.cpython-311.pyc
├── build_vocab.py
├── build_vocab_coco.py
├── captioning_DIY.py
├── captions
│   ├── generated_captions.json
│   └── generated_captions1.json
├── data
│   ├── coco
│   └── df
├── dataloader.py
├── download.sh
├── flagged
│   ├── image
│   └── log.csv
├── google
│   └── vit-base-patch16-224
├── img
│   ├── Transformer.png
│   └── ViT.png
├── metric
│   ├── __init__.py
│   ├── data
│   ├── data_list
│   ├── demo.py
│   ├── examples
│   └── pycocoevalcap
├── models
│   ├── ViT_captioning_epoch0.pt
│   ├── ViT_captioning_epoch1.pt
│   ├── ViT_captioning_epoch2.pt
│   ├── ViT_captioning_epoch3.pt
│   └── ViT_captioning_epoch4.pt
├── requirements.txt
├── resize.py
├── sampler.py
└── transform_json
    ├── lower2upper.py
    ├── str2list.py
    └── str2list_json.py
```
`build_vocab.py`: 构建词汇表，从数据集中提取单词，并创建映射。  
`captioning_DIY.py`: 包含生成图像字幕的主要模型代码。  
`dataloader.py`: 负责加载和预处理数据集。  
`download.sh`: Shell脚本，用于下载数据集。  
`Gradio.py`: 创建Gradio界面，用于模型交互。  
`resize.py`: 调整图像大小，用于数据预处理。  
`sampler.py`: 用于从模型中采样和生成预测。  
`ViT_example.py`: 包含ViT的示例代码。  

<br>

## 三.安装指南
1. 克隆仓库：`git clone git@github.com:abinzzz/Image-Captioning.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 下载预训练模型和数据集：运行 `download.sh`并在[这里]([DeepFashion数据集](https://github.com/yumingj/DeepFashion-MultiModal))下载

## 四.运行



1. 数据预处理
```
python resize.py
```

<br>

如果出现报错如下报错：
```cmd
(ab) root@3090_0002:/data10/cyb/Vision-Transformer# python resize.py
Traceback (most recent call last):
  File "/data10/cyb/Vision-Transformer/resize.py", line 42, in <module>
    main(args)
  File "/data10/cyb/Vision-Transformer/resize.py", line 30, in main
    resize_images(image_dir, output_dir, image_size)
  File "/data10/cyb/Vision-Transformer/resize.py", line 20, in resize_images
    img = resize_image(img, size)
  File "/data10/cyb/Vision-Transformer/resize.py", line 8, in resize_image
    return image.resize(size, Image.ANTIALIAS)
AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'
```

**出错原因**: 原来是在pillow的10.0.0版本中，ANTIALIAS方法被删除了，使用新的方法 **(Image.LANCZOS,Image.Resampling.LANCZOS)** 即可


修改后的代码:
```python
def resize_image(image, size):
    """将图像调整为给定的大小"""
    return image.resize(size, Image.LANCZOS)
```

<br>

2. 为标题文本构建词汇表
```
python build_vocab.py
```

3. 安装Transformer
```
pip install transformers
```

-----

coco数据集的json文件格式：
```json
{"info": {"description": "COCO 2014 Dataset","url": "http://cocodataset.org","version": "1.0","year": 2014,"contributor": "COCO Consortium","date_created": "2017/09/01"},

 "images": [
           {"license": 3,"file_name": "COCO_val2014_000000391895.jpg","coco_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg","height": 360,"width": 640,"date_captured": "2013-11-14 11:18:45","flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg","id": 391895},
            {"license": 4,"file_name": "COCO_val2014_000000522418.jpg","coco_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000522418.jpg","height": 480,"width": 640,"date_captured": "2013-11-14 11:38:44","flickr_url": "http://farm1.staticflickr.com/1/127244861_ab0c0381e7_z.jpg","id": 522418}
            ]

"licenses": [
             {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},    
             {"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},
             {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},
             {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},
             {"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},
             {"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},
             {"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},
             {"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}
            ]

"annotations": [{"image_id": 203564,"id": 37,"caption": "A bicycle replica with a clock as the front wheel."},
                {"image_id": 179765,"id": 38,"caption": "A black Honda motorcycle parked in front of a garage."},
                {"image_id": 322141,"id": 49,"caption": "A room with blue walls and a white sink and door."},
                {"image_id": 16977,"id": 89,"caption": "A car that seems to be parked illegally behind a legally parked car"},
                {"image_id": 106140,"id": 98,"caption": "A large passenger airplane flying through the air."},
                {"image_id": 106140,"id": 101,"caption": "There is a GOL plane taking off in a partly cloudy sky."},
                {"image_id": 322141,"id": 109,"caption": "Blue and white color scheme in a small bathroom."},
                {"image_id": 322141,"id": 121,"caption": "This is a blue and white bathroom with a wall sink and a lifesaver on the wall."}
               ]
}
```



我们的deepfashion-mutimodal数据集的json文件：
```json
{
 "WOMEN-Jackets_Coats-id_00005611-01_4_full.jpg": "The upper clothing has long sleeves, cotton fabric and solid color patterns. The neckline of it is v-shape. The lower clothing is of long length. The fabric is denim and it has solid color patterns. This lady also wears an outer clothing, with cotton fabric and complicated patterns. This female is wearing a ring on her finger. This female has neckwear.",
 "WOMEN-Tees_Tanks-id_00005033-03_4_full.jpg": "Her tank shirt has no sleeves, chiffon fabric and graphic patterns. It has a round neckline. The person wears a long pants. The pants are with denim fabric and solid color patterns. The lady wears a ring.",
 "WOMEN-Rompers_Jumpsuits-id_00000245-01_1_front.jpg": "Her tank top has no sleeves, cotton fabric and solid color patterns. It has a v-shape neckline. This woman wears a long trousers. The trousers are with cotton fabric and solid color patterns. There is a ring on her finger. The lady wears a belt. There is an accessory on her wrist."
}
 
 ```
----


4. 训练图像描述
```
python captioning_DIY.py
```

5. 选取一个图像进行测试
```
python sampler.py --image_path <任意图像路径>
python sampler.py --image_path ./data/train2014/COCO_train2014_000000581921.jpg
```
