import os
import torch
from bloodnet50 import bloodnet50
from torchvision.transforms import Compose, Normalize, ToTensor,CenterCrop
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
import matplotlib.pyplot as plt

model =bloodnet50()
model.load_state_dict(torch.load('./bloodnet50_new.pth'))
model =model.eval()
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

def center_crop(img, dim):
    """Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
    width, height = img.shape[1], img.shape[0]

    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

blood_list = os.listdir('./train1/1d_train/')

for i in blood_list:
    rgb_img = cv2.imread(f'./train1/1d_train/{i}', 1)[:, :, ::-1]
    rgb_img = center_crop(rgb_img, (96, 96))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.1046, 0.0929, 0.0939],
                                    std=[0.1141, 0.1141, 0.1162])
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=True,
                        aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam,use_rgb=True)
    plt.imshow(cam_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'./fig/{i}',dpi=600)




