import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -----------------------------
# Config
# -----------------------------
image_size = 256
batch_size = 32
latent_dim = 100
epochs = 200
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All classes and functions can be defined at the top level
# -----------------------------
# Custom Dataset Class
# -----------------------------


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(
            root_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, 0

# -----------------------------
# Models (Corrected for 256x256)
# -----------------------------


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0,
                               bias=False), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(
                512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(
                256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(
                128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(
                32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False), nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False), nn.LeakyReLU(
                0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(
                64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(
                128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(
                256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(
                512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), nn.BatchNorm2d(
                1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)


# ## --- MAIN EXECUTION BLOCK --- ##
# This is the required fix.
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "final_shayad", "class1")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = CustomImageDataset(root_dir=data_path, transform=transform)
    # On Windows, set num_workers=0 if you still face issues, but 2 should be fine.
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    os.makedirs("generated_256", exist_ok=True)
    print(f"Starting Training on {len(dataset)} images on {device}...")

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            b_size = real_imgs.size(0)

            labels_real = torch.ones(b_size, device=device)
            labels_fake = torch.zeros(b_size, device=device)

            # Train Discriminator
            netD.zero_grad()
            output_real = netD(real_imgs)
            lossD_real = criterion(output_real, labels_real)

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            output_fake = netD(fake_imgs.detach())
            lossD_fake = criterion(output_fake, labels_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            output = netD(fake_imgs)
            lossG = criterion(output, labels_real)
            lossG.backward()
            optimizerG.step()

        # Save generated samples after each epoch
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        utils.save_image(
            fake, f"generated_256/epoch_{epoch+1}.png", normalize=True, nrow=8)

        # Print progress
        print(
            f"✅ Finished Epoch [{epoch+1}/{epochs}] | Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}"
        )

    print("🎉 Training complete! Images saved in 'generated_256/' folder.")
