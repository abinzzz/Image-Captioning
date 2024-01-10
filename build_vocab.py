import nltk
import pickle
import argparse
from collections import Counter
import json
from pycocotools.coco import COCO
import logging
#nltk.download('punkt')

class Vocabulary(object):
    """简单词汇包装器"""
    def __init__(self):
        self.word2idx = {}#单词到索引的映射
        self.idx2word = {}#索引到单词的映射
        self.idx = 0

    def add_word(self, word):
        """向词汇表中添加新单词"""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        """当调用一个 Vocabulary 实例并传入一个单词时，该方法会返回该单词对应的索引"""
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        """词汇表的大小"""
        return len(self.word2idx)

def build_vocab(json_file, threshold):
    """构建一个词汇表（Vocabulary）"""
    """
    json文件内容:
    {"info": {
               "description": "COCO 2014 Dataset",
               "url": "http://cocodataset.org",
               "version": "1.0",
               "year": 2014,
               "contributor": "COCO Consortium",
               "date_created": "2017/09/01"
               },
    
    "images": [
     
                 {"license": 3,
                  "file_name": "COCO_val2014_000000391895.jpg",
                  "coco_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg",
                  "height": 360,
                  "width": 640,
                  "date_captured": "2013-11-14 11:18:45",
                  "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
                  "id": 391895},
                  
                {"license": 4,
                 "file_name": "COCO_val2014_000000522418.jpg",
                 "coco_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000522418.jpg",
                 "height": 480,
                 "width": 640,
                 "date_captured": "2013-11-14 11:38:44",
                 "flickr_url": "http://farm1.staticflickr.com/1/127244861_ab0c0381e7_z.jpg",
                 "id": 522418}
                 
                ]
    
    "licenses": [
    
                 {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                  "id": 1,
                  "name": "Attribution-NonCommercial-ShareAlike License"}, 
                     
                 {"url": "http://creativecommons.org/licenses/by-nc/2.0/",
                  "id": 2,
                  "name": "Attribution-NonCommercial License"}

                ]
    
    "annotations": [
                  
                    {"image_id": 203564,
                     "id": 37,
                     "caption": "A bicycle replica with a clock as the front wheel."},
                     
                    {"image_id": 179765,
                     "id": 38,
                     "caption": "A black Honda motorcycle parked in front of a garage."},

                   ]
    }
        """


    #coco = COCO(json)#使用 COCO 工具读取 JSON 文件，这通常包含了图像的注释信息，如图像的字幕。

    print(json_file)

    with open(json_file, 'r') as j:
        df = json.load(j)

    """
    coco下面包含了anns,dataset,imagetoanns,images
    
    anns:
    {48: {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}, 
     67: {'image_id': 116100, 'id': 67, 'caption': 'A panoramic view of a kitchen and all of its appliances.'}, 
     126: {'image_id': 318556, 'id': 126, 'caption': 'A blue and white bathroom with butterfly themed wall tiles.'}, 
     148: {'image_id': 116100, 'id': 148, 'caption': 'A panoramic photo of a kitchen and dining room'}}
     
     dataset:就是json文件内容
     
     imgToAnns:
     defaultdict(<class 'list'>, 
                 {318556: [
                           {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}, 
                           {'image_id': 318556, 'id': 126, 'caption': 'A blue and white bathroom with butterfly themed wall tiles.'}, 
                           {'image_id': 318556, 'id': 219, 'caption': 'A bathroom with a border of butterflies and blue paint on the walls above it.'},
                           {'image_id': 318556, 'id': 255, 'caption': 'An angled view of a beautifully decorated bathroom.'}, 
                           {'image_id': 318556, 'id': 3555, 'caption': 'A clock that blends in with the wall hangs in a bathroom. '}
                           ]
                }
                           
    imgs:
    {57870: {
             'license': 5, 
             'file_name': 'COCO_train2014_000000057870.jpg', 
             'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg', 
             'height': 480, 
             'width': 640, 
             'date_captured': '2013-11-14 16:28:13', 
             'flickr_url': 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg', 
             'id': 57870
             }, 
             
    384029: {
              'license': 5, 
              'file_name': 'COCO_train2014_000000384029.jpg', 
              'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000384029.jpg', 
              'height': 429, 
              'width': 640, 
              'date_captured': '2013-11-14 16:29:45', 
              'flickr_url': 'http://farm3.staticflickr.com/2422/3577229611_3a3235458a_z.jpg', 
              'id': 384029
              }
    }
    """


    counter = Counter()#使用 Counter 对象来跟踪每个单词出现的频率
    #ids = coco.anns.keys()

    #ids=每个描述对应的id
    #dict_keys([48, 67, 126, 148])

    for _,desc in df.items():#for i, id in enumerate(ids):
        """
        遍历 JSON 文件中的每个注释（字幕）
        使用 nltk.tokenize.word_tokenize 对每个字幕进行分词，将字幕分解成单词列表
        使用 counter.update(tokens) 更新这些单词的频率计数
        每处理1000个字幕，打印一条进度信息
        """
        #caption = str(coco.anns[id]['caption'])
        """  caption='A very clean and well decorated empty bathroom'   """

        tokens = nltk.tokenize.word_tokenize(desc.lower())
        """  tokens=['a', 'very', 'clean', 'and', 'well', 'decorated', 'empty', 'bathroom']  """
        counter.update(tokens)

        #单词：频率
        """Counter({'a': 1, 'very': 1, 'clean': 1, 'and': 1, 'well': 1, 'decorated': 1, 'empty': 1, 'bathroom': 1})"""

        # if (i+1) % 1000 == 0:
        #     print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    #如果单词频率小于“threshold”，则该单词被丢弃,word是单词，cnt是频率
    words = [word for word, cnt in counter.items() if cnt >= threshold]#单词的列表

    #创建一个 Vocabulary 实例，并添加一些特殊标记（如 '<pad>', '<start>', '<end>', '<unk>'）
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    """
       vocab:单词对应索引的字典
       idx2word: {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>', 4: 'a', 5: 'very'}
       word2idx:{'<end>': 2, '<pad>': 0, '<start>': 1, '<unk>': 3, 'a': 4, 'very': 5}
    """

    #将筛选后的词汇添加到 Vocabulary 实例中
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab





def main(args):
    vocab = build_vocab(json_file=args.caption_path, threshold=args.threshold)#建立词汇表

    vocab_path = args.vocab_path#存储词汇表的路径
    with open(vocab_path, 'wb') as f:
        """
        使用 pickle 模块的 dump 方法将 vocab 对象序列化并写入之前打开的文件 f 中
        这样可以将词汇表对象持久化保存到文件系统中
        """
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))#打印词汇表大小
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))#打印词汇表路径


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/df/train_captions.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/df/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
