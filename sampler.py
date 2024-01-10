from PIL import Image
from captioning_DIY import Img2Seq, Encoder, Decoder
from torchvision import transforms
import numpy as np
import argparse
import pickle
import torch
from build_vocab import Vocabulary


def sample_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = Image.open(image_path)
    raw_image = image.resize([224, 224], Image.LANCZOS)

    image = transform(image).unsqueeze(0)
    return image, raw_image


def capitalize_sentences(sentence_list):
    # 移除句子中的 <start> 和 <end> 标记
    cleaned_list = [word for word in sentence_list if word not in ['<start>', '<end>']]

    # 将单词列表转换为字符串，正确处理空格和标点符号
    sentence_str = ' '.join(cleaned_list).replace(" ,", ",").replace(" .", ".")

    # 按句号分割字符串为多个句子，并大写每个句子的首字母
    sentences = sentence_str.split('. ')
    sentences_capitalized = [s.capitalize() for s in sentences]

    return '. '.join(sentences_capitalized).strip()


def main(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    enc = Encoder()
    dec = Decoder(len(vocab), args.hidden_size, args.dec_layers, args.num_heads, args.hidden_size, args.dropout, device)
    model = Img2Seq(enc, dec, vocab.word2idx['<pad>'], device=device)
    model.load_state_dict(torch.load(args.model_path))
    src, raw_image = sample_image(args.image_path)

    src = src.to(device)
    model = model.to(device)
    enc_src = model.encoder(src)
    src_mask = model.make_src_mask(enc_src)
    trg_indexes = [vocab.word2idx['<start>']]

    for i in range(args.max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        if pred_token == vocab.word2idx['<end>']:
            break

    decoded_sent = [vocab.idx2word[i] for i in trg_indexes]

    # 将解码的句子列表转换为一段文字，并使每个句子的首字母大写
    final_text = capitalize_sentences(decoded_sent)
    print('Predicted Caption:', final_text)
    #print('Predicted Caption:', decoded_sent)
    #plot_attention(raw_image, decoded_sent, attention)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='./data/df/images/WOMEN-Shorts-id_00006003-01_4_full.jpg')
    parser.add_argument('--model_path', type=str, default='models/ViT_captioning_epoch4.pt',
                        help='path for saving trained models')
    #parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/df/vocab.pkl', help='path for vocabulary wrapper')
    #parser.add_argument('--image_dir', type=str, default='/data/df/resized2014', help='directory for resized images')
    #parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                        #help='path for train annotation json file')
    #parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--max_len', type=int, default=100, help='max length of decoded sentence')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=768, help='dimension of lstm hidden states')
    parser.add_argument('--dec_layers', type=int, default=6, help='number of decoder layers in transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='amount of attention heads')
    #parser.add_argument('--clip', type=int, default=1, help='gradient clipping value')
    parser.add_argument('--dropout', type=float, default=0.1)

    # parser.add_argument('--num_epochs', type=int, default=5)
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--num_workers', type=int, default=2)
    #parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)