#!/usr/bin/env python
# coding: utf-8
import logging
import torch
import torch.nn as nn

import numpy as np
import math
from transformers import ViTModel
from dataloader import get_loader
from torchvision import transforms
import pickle
from build_vocab import Vocabulary
from tqdm import tqdm
import argparse
#from Encoder import Encoder
import json

class PositionalEncoding(nn.Module):
    """Transformer 网络添加位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)#模型的维度和位置编码的最大长度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)#从 0 到 max_len 的连续值，表示位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))#缩放项
        pe[:, 0::2] = torch.sin(position * div_term)#为位置矩阵的偶数部分赋值正弦函数值
        pe[:, 1::2] = torch.cos(position * div_term)#为位置矩阵的奇数部分赋值余弦函数值。
        pe = pe.unsqueeze(0).transpose(0, 1)#调整位置编码矩阵的形状以便后续操作
        self.register_buffer('pe', pe)

    def forward(self, x):
        #将输入 x（通常是序列的嵌入表示）与位置编码相加，以便每个位置的嵌入都有唯一的表示
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):
    """"图像数据编码为固定长度的特征向量的编码器"""

    def __init__(self):
        super().__init__()

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.hid_dim = 768
        self.proj = nn.Linear(768, 768)  #全连接层

    def forward(self, src):
        # return = [batch size, patch len, hid dim]
        #print(self.vit(pixel_values=src).last_hidden_state.size())
        return self.proj(self.vit(pixel_values=src).last_hidden_state)



class MultiHeadAttentionLayer(nn.Module):
    """多头注意力机制,允许模型在计算注意力时同时关注来自不同位置的不同表示子空间"""
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0#确保隐藏维度可以均匀地分配到每个头
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads#每个头的维度
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Step 1: Linear transformations for query, key, and value
        Q = self.fc_q(query)  # [batch size, query len, hid dim]
        K = self.fc_k(key)  # [batch size, key len, hid dim]
        V = self.fc_v(value)  # [batch size, value len, hid dim]

        # Step 2: Split the embeddings into `self.n_heads` heads
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Step 3: Compute the energy (attention weights)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # Step 4: Apply mask (if any)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        # Step 5: Normalize attention weights
        attention = torch.softmax(energy, dim=-1)

        # Step 6: Apply attention to the value vector
        x = torch.matmul(self.dropout(attention), V)

        # Step 7: Concatenate heads and apply final linear layer
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    """对每个位置的特征独立地应用相同的全连接层变换"""
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 120):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)#创建一个嵌入层，用于将目标序列的令牌转换为固定维度的向量
        self.pos_embedding = nn.Embedding(max_length, hid_dim)#创建一个位置嵌入层，用于给目标序列的每个位置编码一个固定维度的向量
        # self.pos_encoding = PositionalEncoding(hid_dim, max_length)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len] #目标序列
        #enc_src = [batch size, src len, hid dim] #编码去输出
        #trg_mask = [batch size, 1, trg len, trg len] #目标序列掩码
        #src_mask = [batch size, 1, 1, src len] #源序列掩码

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]


        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)#生成位置序列，并将其复制到每个样本
        a = (self.tok_embedding(trg) * self.scale).size()
        b = self.pos_embedding(pos).size()
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))#对目标序列的令牌应用嵌入，缩放，加上位置嵌入，然后应用 dropout


        #每个解码器层进行迭代
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)

        #return output, attention
        return output,attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))  # residual connection and layer norm

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))  # residual connection and layer norm

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))  # residual connection and layer norm

        return trg, attention

class Img2Seq(nn.Module):
    def __init__(self, encoder, decoder, trg_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx#目标填充索引
        self.device = device


        
    def make_src_mask(self, src):
        """
        创建源数据（src）的掩码
        创建一个与源数据同样形状的全1张量，然后添加两个维度，并将其传输到与源数据相同的设备上
        """

        # src=[16,197,768]
        src_mask = torch.ones(src.size(0), src.size(1)).unsqueeze(1).unsqueeze(2).to(src.device)
        """
        src_mask=[batch size,1,1,src len]=[16,1,1,197]
        """
        #print(src_mask.size())
        #return src_mask
    
    def make_trg_mask(self, trg):
        """
        创建trg(caption)的掩码
        首先创建一个用于标识目标序列中非填充元素的掩码，然后创建一个下三角矩阵，用于确保解码器只能看到之前的元素（这对于序列生成任务很重要）。
        最后，它返回这两个掩码的逻辑与结果。
        """

        """
        trg=tensor(16,86)#
        trg_pad_mask=(16,1,1,86)#标识了目标序列中哪些元素是真实数据，哪些是填充数据
        trg_len=86
        trg_sub_mask=(86,86)#下三角矩阵，只能看到当前和之前的位置的信息，而不能看到未来的位置
        trg_mask=(16,1,86,86)#既考虑了填充元素的屏蔽，又保证了每个时间步只能看到之前的信息
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()

        #print(trg_pad_mask.size())
        #print(trg_sub_mask.size())
        trg_mask = trg_pad_mask & trg_sub_mask
        #print(trg_mask.size())

        return trg_mask

    def forward(self, src, trg):


        """
        src=tensor(16,3,224,224)#图片
        tgt=tensor(16,77)#图片对应描述的张量
        l=list(16)#描述中有多少个词
        """

        """
        trg=tensor(16,77) -> trg_mask=tensor(16,1,77,77)
        src=tensor(16,3,224,224) -> enc_src=tensor(16,197,768) -> src_mask=tensor(16,1,1,197)
        
        """
                
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src)#图片送入encoder
        #print(src.size())
        src_mask = self.make_src_mask(enc_src)    
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output, attention

def train(model, iterator, optimizer, criterion, clip, log_step=10):
    
    model.train()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    epoch_loss = 0
    
    p_bar = tqdm(enumerate(iterator), total=len(iterator))#创建一个进度条
    
    for i, batch in p_bar:
        
        src, tgt, l = batch

        """
        src=tensor(16,3,224,224)#图片
        tgt=tensor(16,98)#图片描述的张量
        l=list(16)
        """
        src = src.to(device)
        trg = tgt.to(device)


        
        optimizer.zero_grad()#清除梯度
        
        output, _ = model(src, trg[:,:-1])#将src和tgt送入模型中

        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()#反向传播
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)#裁剪梯度 防止爆炸
        
        optimizer.step()#更新模型权重
        
        epoch_loss += loss.item()
        
        if (i+1) % log_step == 0:
            p_bar.set_description(f'STEP {i+1} | Loss: {(epoch_loss/(i+1)):.3f} | Train PPL: {math.exp(epoch_loss/(i+1)):7.3f}')
        
    return epoch_loss / len(iterator)

def inference(src, tgt, model, device, vocab, max_len = 100):
    
    '''
    给定的图像到序列模型上执行预测
    src: single image from dataloader
    tgt: single sequence of word_id from dataloader
    model: Img2Seq model
    max_len: max length of decoded sentence (int)
    '''

    model.eval()
    """tgt: tensor(16,69)"""

    #将目标序列（tgt）转换为列表（gold），并将其ID转换为相应的单词（gold_sent）
    gold = tgt.tolist()
    """gold:list(16)中每个元素都是list(69)"""

    gold_sent = [vocab.idx2word[i] for i in gold[0]]
    #gold_sent = [vocab.idx2word[i] for i in gold]
    trg_indexes = [vocab.word2idx['<start>']]


    #使用模型的编码器处理源图像（src），并创建相应的源掩码
    """src:tensor(16,3,224,224)"""
    enc_src = model.encoder(src[0])
    src_mask = model.make_src_mask(enc_src)

    #最多迭代 max_len 次来生成预测序列
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)#将目标索引转换为张量，并传输到指定的设备

        trg_mask = model.make_trg_mask(trg_tensor)#使用模型生成目标掩码

        #在无梯度计算的环境下（torch.no_grad()），使用解码器和之前生成的掩码进行前向传播，得到输出和注意力
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        #从解码器输出中选择概率最高的单词作为预测单词
        pred_token = output.argmax(2)[:,-1].item()

        #将预测的单词加入到目标索引列表中
        trg_indexes.append(pred_token)


        if pred_token == vocab.word2idx['<end>']:#如果预测的单词是特殊的结束标记 <end>，则结束循环
            break
    
    trg_tokens = [vocab.idx2word[i] for i in trg_indexes]#将目标索引列表转换为单词列表
    
    print('gold sent', gold_sent)
    print('pred sent', trg_tokens)

    return trg_tokens[1:], attention

# def test(args):
#     device = torch.device('mps'  if torch.backends.mps.is_available() else 'cpu')
#
#     #数据转换
#     transform = transforms.Compose([
#         transforms.RandomCrop(224), #图像随机裁剪为 224x224 像素的大小
#         transforms.RandomHorizontalFlip(), #一定的概率（默认为 0.5，即 50%）水平翻转图像
#         transforms.ToTensor(), #将 PIL 图像或者 NumPy 数组转换成 PyTorch 张量（Tensor）
#         transforms.Normalize((0.485, 0.456, 0.406), #归一化处理
#                             (0.229, 0.224, 0.225))])
#
#     #加载预先准备好的词汇表
#     with open('data/coco/vocab.pkl', 'rb') as f:
#         vocab = pickle.load(f)
#
#     """
#     vocab:索引到单词
#     {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>', 4: 'a', 5: 'very',
#      6: 'clean', 7: 'and', 8: 'well', 9: 'decorated', 10: 'empty',
#      11: 'bathroom', 12: 'panoramic', 13: 'view', 14: 'of', 15: 'kitchen',
#      16: 'all', 17: 'its', 18: 'appliances', 19: '.', 20: 'blue',
#      21: 'white', 22: 'with', 23: 'butterfly', 24: 'themed', 25: 'wall',
#      26: 'tiles', 27: 'photo', 28: 'dining', 29: 'room', 30: 'stop',
#      31: 'sign', 32: 'across', 33: 'the', 34: 'street', 35: 'from',
#      36: 'red', 37: 'car', 38: 'vandalized', 39: 'beetle'}
#     """
#
#     #创建一个数据加载器 train_loader，它从指定的图像目录和字幕路径加载数据，应用之前定义的转换，并按照给定的批次大小和其他参数
#     #train_loader = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)
#
#     #print('vocab size',len(vocab))
#     enc = Encoder()
#     dec = Decoder(len(vocab), args.hidden_size, args.dec_layers, args.num_heads, args.hidden_size, args.dropout, device)
#     model = Img2Seq(enc, dec, vocab.word2idx['<pad>'],device=device)#cuda
#
#     # def count_parameters(model):
#     #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#     #
#     # print(f'The model has {count_parameters(model):,} trainable parameters')
#
#     # optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
#     # criterion = nn.CrossEntropyLoss(ignore_index = 0)
#     model = model.to(device)
#
#     # for epoch in range(args.num_epochs):
#     #
#     #     train_loss = train(model, train_loader, optimizer, criterion, args.clip)
#     #     torch.save(model.state_dict(), f'{args.model_path}/ViT_captioning_epoch{epoch}.pt')
#     #     print(f'EPOCH {epoch}\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#
#     # 加载模型
#     #model = Img2Seq(enc, dec, vocab.word2idx['<pad>'], device=device)
#     model_path = f'{args.model_path}/ViT_captioning_epoch{4}.pt'  # 加载最后一个 epoch 的模型
#     model.load_state_dict(torch.load(model_path))
#     model = model.to(device)
#
#
#     inference_loader = get_loader(args.image_dir, args.caption_test_path, vocab, transform, 16, shuffle=True, num_workers=args.num_workers)
#     for img, tgt, _ in inference_loader:
#         # print("img",img)
#         # print("tgt",tgt)
#         # print("_",_)
#         inference(img, tgt, model, device, vocab, args.max_len)





def Train(args):
    device = torch.device('mps'  if torch.backends.mps.is_available() else 'cpu')

    #数据转换
    transform = transforms.Compose([
        transforms.RandomCrop(224), #图像随机裁剪为 224x224 像素的大小
        transforms.RandomHorizontalFlip(), #一定的概率（默认为 0.5，即 50%）水平翻转图像
        transforms.ToTensor(), #将 PIL 图像或者 NumPy 数组转换成 PyTorch 张量（Tensor）
        transforms.Normalize((0.485, 0.456, 0.406), #归一化处理
                            (0.229, 0.224, 0.225))])

    #加载预先准备好的词汇表
    with open('data/df/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    print(vocab)

    """
    vocab:索引到单词
    {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>', 4: 'a', 5: 'very', 
     6: 'clean', 7: 'and', 8: 'well', 9: 'decorated', 10: 'empty', 
     11: 'bathroom', 12: 'panoramic', 13: 'view', 14: 'of', 15: 'kitchen', 
     16: 'all', 17: 'its', 18: 'appliances', 19: '.', 20: 'blue', 
     21: 'white', 22: 'with', 23: 'butterfly', 24: 'themed', 25: 'wall', 
     26: 'tiles', 27: 'photo', 28: 'dining', 29: 'room', 30: 'stop', 
     31: 'sign', 32: 'across', 33: 'the', 34: 'street', 35: 'from', 
     36: 'red', 37: 'car', 38: 'vandalized', 39: 'beetle'}
    """

    #创建一个数据加载器 train_loader，它从指定的图像目录和字幕路径加载数据，应用之前定义的转换，并按照给定的批次大小和其他参数
    train_loader = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)

    print('vocab size',len(vocab))
    enc = Encoder()
    dec = Decoder(len(vocab), args.hidden_size, args.dec_layers, args.num_heads, args.hidden_size, args.dropout, device)
    model = Img2Seq(enc, dec, vocab.word2idx['<pad>'],device=device)#cuda

    """Img2Seq(
          (encoder): Encoder(
            (vit): ViTModel(
              (embeddings): ViTEmbeddings(
                (patch_embeddings): ViTPatchEmbeddings(
                  (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
                )
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (encoder): ViTEncoder(
                (layer): ModuleList(
                  (0-11): 12 x ViTLayer(
                    (attention): ViTAttention(
                      (attention): ViTSelfAttention(
                        (query): Linear(in_features=768, out_features=768, bias=True)
                        (key): Linear(in_features=768, out_features=768, bias=True)
                        (value): Linear(in_features=768, out_features=768, bias=True)
                        (dropout): Dropout(p=0.0, inplace=False)
                      )
                      (output): ViTSelfOutput(
                        (dense): Linear(in_features=768, out_features=768, bias=True)
                        (dropout): Dropout(p=0.0, inplace=False)
                      )
                    )
                    (intermediate): ViTIntermediate(
                      (dense): Linear(in_features=768, out_features=3072, bias=True)
                      (intermediate_act_fn): GELUActivation()
                    )
                    (output): ViTOutput(
                      (dense): Linear(in_features=3072, out_features=768, bias=True)
                      (dropout): Dropout(p=0.0, inplace=False)
                    )
                    (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  )
                )
              )
              (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (pooler): ViTPooler(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (activation): Tanh()
              )
            )
            (proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (decoder): Decoder(
            (tok_embedding): Embedding(109, 768)
            (pos_embedding): Embedding(100, 768)
            (layers): ModuleList(
              (0-5): 6 x DecoderLayer(
                (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (enc_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (ff_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (self_attention): MultiHeadAttentionLayer(
                  (fc_q): Linear(in_features=768, out_features=768, bias=True)
                  (fc_k): Linear(in_features=768, out_features=768, bias=True)
                  (fc_v): Linear(in_features=768, out_features=768, bias=True)
                  (fc_o): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (encoder_attention): MultiHeadAttentionLayer(
                  (fc_q): Linear(in_features=768, out_features=768, bias=True)
                  (fc_k): Linear(in_features=768, out_features=768, bias=True)
                  (fc_v): Linear(in_features=768, out_features=768, bias=True)
                  (fc_o): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (positionwise_feedforward): PositionwiseFeedforwardLayer(
                  (fc_1): Linear(in_features=768, out_features=768, bias=True)
                  (fc_2): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (fc_out): Linear(in_features=768, out_features=109, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )"""

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    model = model.to(device)

    for epoch in range(args.num_epochs):

        train_loss = train(model, train_loader, optimizer, criterion, args.clip)
        torch.save(model.state_dict(), f'{args.model_path}/ViT_captioning_epoch{epoch}.pt')
        print(f'EPOCH {epoch}\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    # # 加载模型
    # model = Img2Seq(enc, dec, vocab.word2idx['<pad>'], device=device)
    # model_path = f'{args.model_path}/ViT_captioning_epoch{args.num_epochs - 1}.pt'  # 加载最后一个 epoch 的模型
    # model.load_state_dict(torch.load(model_path))
    # model = model.to(device)
    #     # 加载模型
    #
    #
    #
    #
    #
    # inference_loader = get_loader(args.image_dir, args.caption_test_path, vocab, transform, 16, shuffle=True, num_workers=args.num_workers)
    # for img, tgt in inference_loader:
    #     inference(img, tgt, model, device, vocab, args.max_len)

def main():
    Train(args)
    #test(args)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/df/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/df/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/df/train_captions.json', help='path for train annotation json file')
    parser.add_argument('--caption_test_path', type=str, default='data/df/test_captions.json', help='path for test annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=120)

    # Model parameters
    parser.add_argument('--hidden_size', type=int , default=768, help='dimension of lstm hidden states')
    parser.add_argument('--dec_layers', type=int , default=6, help='number of decoder layers in transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='amount of attention heads')
    parser.add_argument('--clip', type=int, default=1, help='gradient clipping value')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    args = parser.parse_args()
    main()
