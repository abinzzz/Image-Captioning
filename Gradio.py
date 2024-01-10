from abc import ABC, abstractmethod
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import pickle
from torchvision import transforms
import torch
#device = 'cuda'
from captioning_DIY import Encoder, Decoder, Img2Seq
from build_vocab import Vocabulary
#device = torch.device('cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device='cpu'

class ModelGenerate(ABC):
    @abstractmethod
    def response(image):
        pass


class FinetunedBLIP(ModelGenerate):
    def __init__(self):
        pass
        # self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # self.model_blip = BlipForConditionalGeneration.from_pretrained("menu1100/blip-deepfashion").to(device)


    def response(self, image: Image.Image) -> str:
        # inputs = self.processor(images=image, return_tensors="pt").to(device)
        # pixel_values = inputs.pixel_values
        # generated_ids = self.model_blip.generate(pixel_values=pixel_values, max_length=100)
        # generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return generated_caption
        return 'not implemented'


class ViT(ModelGenerate):
    def __init__(self, vocab_path, model_path) -> None:
        # 加载词汇表
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # 创建模型的实例
        self.encoder = Encoder().to(device)
        self.decoder = Decoder(len(self.vocab), 768, 6, 8, 768, 0.1, device).to(device)
        self.model = Img2Seq(self.encoder, self.decoder, self.vocab.word2idx['<pad>'], device=device)

        # 加载预训练模型
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)

    def response(self, image: Image.Image) -> str:
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        #image = Image.open(image)

        image = transform(image).unsqueeze(0).to(device)

        enc_src = self.encoder(image)

        # 生成描述
        src_mask = self.model.make_src_mask(enc_src)
        trg_indexes = [self.vocab.word2idx['<start>']]

        for _ in range(120):  # 假设最大长度为100
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = self.model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output, _ = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            if pred_token == self.vocab.word2idx['<end>']:
                break

        # 将词汇索引转换为单词
        decoded_words = [self.vocab.idx2word[i] for i in trg_indexes if
                         i not in [self.vocab.word2idx['<start>'], self.vocab.word2idx['<end>']]]

        # 将单词列表转换为字符串，正确处理空格和标点符号
        sentence_str = ' '.join(decoded_words).replace(" ,", ",").replace(" .", ".")

        # 按句号分割字符串为多个句子，并大写每个句子的首字母
        sentences = sentence_str.split('. ')
        sentences_capitalized = [s.capitalize() for s in sentences]

        return '. '.join(sentences_capitalized).strip()


class Transformer(ModelGenerate):
    def __init__(self) -> None:
        pass

    def response(self, image: Image.Image) -> str:
        return 'not implemented'

# 实例化三种模型

model1=Transformer()
model2=ViT(vocab_path="./data/df/vocab.pkl",model_path="./model/ViT_captioning_epoch4.pt")
model3=FinetunedBLIP()

models = {
    'Transformer Encoder + Decoder':model1.response,
    'ViT + Transformer Decoder':model2.response,
    'Finetuned BLIP':model3.response
}

import gradio as gr
import numpy as np


def process_image(input_data):
    if isinstance(input_data, str):  # 如果是文件路径
        img = Image.open(input_data)
    elif isinstance(input_data, np.ndarray):  # 如果是NumPy数组
        img = Image.fromarray(input_data)
    else:
        raise ValueError("Unsupported input type")
    return img


# def generate(image, model_name):
#     try:
#         image = process_image(image)
#         response = models[model_name](image)
#         return response
#     except:
#         return 'Please submit a picture.'

def generate(image, model_name):
    try:
        image = process_image(image)
        response = models[model_name](image)
        return response
    except Exception as e:
        print(f"Error generating caption: {e}")
        return 'An error occurred. Please submit a picture.'

# 创建 Gradio 接口，添加下拉菜单供用户选择模型
iface = gr.Interface(
    fn=generate,
    inputs=[gr.Image(), gr.Dropdown(["Transformer Encoder + Decoder", "ViT + Transformer Decoder", "Finetuned BLIP"],
                                    label="选择模型")],  # 文本、图像和下拉菜单输入
    outputs="text"
)

# 启动 Gradio 接口
iface.launch()





