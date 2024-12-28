import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import os
import generator
import discriminator
import matplotlib.animation as animation
from IPython.display import HTML

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

        rootfolder = "SBVPI"
        channels = 3
        threads = 2
        batch_size = 128
        image_size = 64
        zlatent = 100
        featuremapsgenerator = 64
        featuremapsdiscriminator = 64
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
        #plt.figure(figsize=(8,8))
        #plt.axis("off")
        #plt.title("Training Images")
        #plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        #plt.show()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        netG = generator.Generator(ngpu, featuremapsgenerator, channels, zlatent).to(self.device)

        if (self.device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        netG.apply(weights_init)
        print(netG)

        netD = discriminator.Discriminator(ngpu, featuremapsdiscriminator, channels, zlatent).to(self.device)

        if (self.device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        netD.apply(weights_init)
        print(netD)

        criterion = nn.BCELoss()
        fixed_noise = torch.randn(64, zlatent, 1, 1, device=self.device)
        real_label = 1.
        fake_label = 0.
        optimizerD = optim.Adam(netD.parameters(), lr=learningrate, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=learningrate, betas=(beta1, 0.999))

        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        for epoch in range(epochs):
            for i, data in enumerate(dataloader, 0):
                netD.zero_grad()
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()


                noise = torch.randn(b_size, zlatent, 1, 1, device=self.device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                netG.zero_grad()
                label.fill_(real_label) 
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))

        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.show()
