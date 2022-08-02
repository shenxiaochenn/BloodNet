import os
import torch
from bloodnet50 import bloodnet50
from torchvision.transforms import Compose, Normalize, ToTensor,CenterCrop,Resize
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, center_crop, to_pil_image

model =bloodnet50()
model.load_state_dict(torch.load('./bloodnet50_delmaxool4.pth'))
model =model.eval()
preprocessing = Compose([
        CenterCrop(96),
        ToTensor(),
        Normalize(mean=[0.1046, 0.0929, 0.0939], std=[0.1141, 0.1141, 0.1162])])

blood_list=os.listdir('./train1/1d_train/')
blood_list=blood_list[239:]
for i in blood_list:
    features = []
    def hook(module, input, output):
        features.append(output.clone().detach())
    handle = model.layer4[2].register_forward_hook(hook)
    img_=Image.open(f'./train1/1d_train/{i}')
    img=preprocessing(img_)
    out = model(img.unsqueeze(0))
    attn=features[0]
    handle.remove()
    attn=attn.mean(axis=1,keepdim=True)
    attn=attn.squeeze()
    attn=(attn-torch.min(attn))/(torch.max(attn)-torch.min(attn))
    result = overlay_mask(center_crop(img_,96), to_pil_image(attn.squeeze(), mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'./attentionmap1/{i}',dpi=600)