import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from pytorch_grad_cam import (
  GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
  AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
  LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
  FinerCAM
)

from resnet import resnet18

def CIFAR10(path, batch_size=1):
    transform=transforms.Compose([
      transforms.ToTensor(),
    ])
    data = datasets.CIFAR10(path, train=False, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

    return loader



def main():

  from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
  from pytorch_grad_cam.utils.image import show_cam_on_image
  import matplotlib.pyplot as plt
  from pytorch_grad_cam.utils.image import (
      show_cam_on_image, deprocess_image, preprocess_image
  )

  device = "cpu"
  # Model and Target Layers
  model = resnet18(device=device).to(device)
  model.eval()
  target_layers = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4]

  # Data and Targets
  data = CIFAR10("/data/DATASETS/CIFAR10", batch_size = 16)

  idx = 2
  for i, (input_tensor, label) in enumerate(data): 
    if i == idx: break
  
  rgb_image = np.transpose(np.asarray(input_tensor), (0, 2, 3, 1))
  targets = [ClassifierOutputTarget(i) for i in label]

  # GradCAM
  # algo = GradCAM
  algo = ScoreCAM
  # algo = FullGrad
  # algo = HiResCAM
  # algo = GradCAMPlusPlus
  # algo = AblationCAM
  # algo = XGradCAM
  # algo = LayerCAM
  with algo(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    B = grayscale_cam.shape[0]
    plt.figure(figsize=(min(4,B)*5, max(B//4,1)*5))
    for i in range(B):
      vis = show_cam_on_image(rgb_image[i], grayscale_cam[i], use_rgb=True)
      plt.subplot(max(B//4,1),min(4,B),i+1)
      plt.axis('off')
      plt.tight_layout()
      plt.imshow(vis)
    plt.savefig("fig.png")

if __name__ == "__main__":
  main()