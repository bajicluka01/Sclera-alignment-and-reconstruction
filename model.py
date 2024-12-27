import torch
import torch.nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import os

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

#diffusion
class Model:
    def __init__(self, images):
        self.images = images
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            ngpu = 1
        else:
            self.device = torch.device("cpu")
            ngpu = 0

        rootfolder = "train_images/"
        channels = 1
        threads = 2
        batch_size = 128
        image_size = 512
        zlatent = 100
        featuremaps = 64
        epochs = 5
        learningrate = 0.0002
        beta1 = 0.5
        
        dataset = dset.ImageFolder(root=rootfolder,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=threads)
        
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
