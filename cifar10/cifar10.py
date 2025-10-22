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

class Model(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)


        if checkpoint is not None: self.load_state_dict(torch.load(checkpoint, weights_only=True))
            

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

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

  # Model and Target Layers
  model = Model("model.pt")
  model.eval()
  target_layers = [model.conv1, model.conv2]

  # Data and Targets
  data = CIFAR10("/data/DATASETS/CIFAR10", batch_size = 8)

  idx = 2
  for i, (input_tensor, label) in enumerate(data): 
    if i == idx: break
  
  rgb_image = np.transpose(np.asarray(input_tensor), (0, 2, 3, 1))
  targets = [ClassifierOutputTarget(i) for i in label]

  # GradCAM
  # algo = GradCAM
  # algo = ScoreCAM
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