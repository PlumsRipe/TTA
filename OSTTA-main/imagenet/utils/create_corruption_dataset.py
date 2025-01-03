from imagenet_c import *
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch
import gorilla


DATA_ROOT = '/data/tjut_m/OWTTT/imagenet/dataset/imagenet2012/'
CORRUPTION_PATH = './corruption'



# corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
#                     glass_blur, motion_blur, zoom_blur, snow, frost, fog,
#                     brightness, contrast, elastic_transform, pixelate, jpeg_compression)

# corruption_tuple = (motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, glass_blur)
corruption_tuple = (snow,)

corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}

class corrupt(object):
    def __init__(self, corruption_name, severity=5):
        self.corruption_name = corruption_name
        self.severity = severity
        return

    def __call__(self, x):
        # x: PIL.Image
        x_corrupted = corruption_dict[self.corruption_name](x, self.severity)
        return np.uint8(x_corrupted)

    def __repr__(self):
        return "Corruption(name=" + self.corruption_name + ", severity=" + str(self.severity) + ")"


print(os.path.join(DATA_ROOT, CORRUPTION_PATH))
if os.path.exists(os.path.join(DATA_ROOT, CORRUPTION_PATH)) is False:
    os.mkdir(os.path.join(DATA_ROOT, CORRUPTION_PATH))



for corruption in corruption_dict.keys():
    # if os.path.exists(os.path.join(DATA_ROOT, CORRUPTION_PATH, corruption + '.pth')):
    #     continue
    print(corruption)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        corrupt(corruption, 5)
    ])

    target_dataset = ImageNet(DATA_ROOT, 'val', transform=val_transform)

    target_dataloader = DataLoader(target_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=16)

    datas = []
    for batch in gorilla.track(target_dataloader):
        datas.append(batch[0])
    datas = torch.cat(datas)
    torch.save(datas, os.path.join(DATA_ROOT, CORRUPTION_PATH, corruption + '.pth'))


"""可视化"""
# data_path = os.path.join(DATA_ROOT, CORRUPTION_PATH, "snow" + '.pth')
# datas = torch.load(data_path)
# # 假设 datas 是 (N, H, W, C) 形状的张量，其中 N 是批量大小，C 是通道数，H 和 W 是高度和宽度。
# images_numpy = datas.cpu().numpy()
# images_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in images_numpy]
# import matplotlib.pyplot as plt
# save_dir = os.path.join("/data/tjut_m/OWTTT/imagenet/dataset/imagenet2012/", 'recovered_images')  # 定义保存目录
# os.makedirs(save_dir, exist_ok=True)
# for i, image in enumerate(images_pil):
#     save_path = os.path.join(save_dir, f"{snow}_{i}.jpg")  # 示例：使用 corruption 名称和索引作为文件名
#     image.save(save_path, format='JPEG')
