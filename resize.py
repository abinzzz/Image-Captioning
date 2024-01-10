import argparse
import os
from PIL import Image


def resize_image(image, size):
    """将图像调整为给定的大小"""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """调整'image_dir'中的图像大小并保存到'output_dir'中"""
    if not os.path.exists(output_dir):#检查是否存在输出目录，不存在就创建
        os.makedirs(output_dir)

    images = os.listdir(image_dir)#获取image_dir中所有文件的列表
    num_images = len(images)
    for i, image in enumerate(images):#遍历并处理图像
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:#每处理100打印一条信息
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

def main(args):
    image_dir = args.image_dir#图像文件夹
    output_dir = args.output_dir#输出文件夹
    image_size = [args.image_size, args.image_size]#调整到的图像尺寸
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/df/images/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/df/resized2014/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)