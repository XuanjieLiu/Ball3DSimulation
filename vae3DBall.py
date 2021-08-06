import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision.datasets as dst
from torchvision.utils import save_image
import gzip
import os


EPOCH = 150
BATCH_SIZE = 64
LATENT_CODE_NUM = 3
log_interval = 10
IMG_CHANNEL = 3
LEARNING_RATE = 0.0001

LAST_H = 10
LAST_W = 15

transform = transforms.ToTensor()
root = './Ball3DImg'
RESULT_PATH = 'vae3DBall/recon/'
train_data = datasets.ImageFolder(root, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

MODEL_PATH = "vae3DBall.pt"

FIRST_CH_NUM = 64
LAST_CN_NUM = FIRST_CH_NUM * 4

def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))

def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(FIRST_CH_NUM),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),

            nn.Conv2d(FIRST_CH_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(FIRST_CH_NUM * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),

            nn.Conv2d(FIRST_CH_NUM * 2, LAST_CN_NUM, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(FIRST_CH_NUM * 4),
            #nn.LeakyReLU(0.2, inplace=True)
            nn.ReLU(),

        )

        self.fc11 = nn.Linear(LAST_CN_NUM * LAST_H * LAST_W, LATENT_CODE_NUM)
        self.fc12 = nn.Linear(LAST_CN_NUM * LAST_H * LAST_W, LATENT_CODE_NUM)
        self.fc2 = nn.Linear(LATENT_CODE_NUM, LAST_CN_NUM * LAST_H * LAST_W)

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(LAST_CN_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(FIRST_CH_NUM * 4),
            #nn.ReLU(inplace=True),
            nn.ReLU(),


            nn.ConvTranspose2d(FIRST_CH_NUM * 2, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(FIRST_CH_NUM),
            # nn.ReLU(inplace=True),
            nn.ReLU(),

            nn.ConvTranspose2d(FIRST_CH_NUM, IMG_CHANNEL, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        z = mu + eps * torch.exp(logvar) * 0.5
        return z

    def forward(self, x):
        out = self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out.view(out.size(0), -1))  # batch_s, latent
        logvar = self.fc12(out.view(out.size(0), -1))  # batch_s, latent
        z = self.reparameterize(mu, logvar)  # batch_s, latent
        out3 = self.fc2(z).view(z.size(0), LAST_CN_NUM, LAST_H, LAST_W)  # batch_s, 8, 7, 7
        return self.decoder(out3), mu, logvar



def codes_sampler(dim):
    codes = np.zeros((BATCH_SIZE, LATENT_CODE_NUM))
    delta = 2 / BATCH_SIZE
    num = -1
    for i in range(0, BATCH_SIZE):
        codes[i][dim] = num
        num += delta
    return codes

def loss_func(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    print(f'BCE loss: {BCE/BATCH_SIZE}')
    print(f'KLD loss: {KLD/BATCH_SIZE}')
    return BCE + KLD



def test():
    if os.path.exists(MODEL_PATH):
        vae.load_state_dict(load_tensor(MODEL_PATH))
        print(f"Model is loaded")
    vae.eval()
    for i in range(0, LATENT_CODE_NUM):
        codes = codes_sampler(i)
        codes = Variable(torch.tensor(codes, dtype=torch.float)).cuda()
        sample = vae.decoder(vae.fc2(codes).view(BATCH_SIZE, LAST_CN_NUM, LAST_H, LAST_W)).cpu()
        save_image(sample.data.view(BATCH_SIZE, IMG_CHANNEL, 80, 120), RESULT_PATH + 'dim_' + str(i) + '.png')

def train(EPOCH):
    if os.path.exists(MODEL_PATH):
        vae.load_state_dict(load_tensor(MODEL_PATH))
        print(f"Model is loaded")
    vae.train()
    total_loss = 0
    for i, (data, _) in enumerate(train_loader, 0):
        data = Variable(data).cuda()
        optimizer.zero_grad()
        recon_x, mu, logvar = vae.forward(data)
        loss = loss_func(recon_x, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        if i % log_interval == 0:
            # sample = Variable(torch.randn(BATCH_SIZE, LATENT_CODE_NUM)).cuda()
            # sample = vae.decoder(vae.fc2(sample).view(BATCH_SIZE, LAST_CN_NUM , LAST_H, LAST_W)).cpu()
            # save_image(sample.data.view(BATCH_SIZE, IMG_CHANNEL, 80, 120), RESULT_PATH + 'sample_' + str(EPOCH) + '.png')
            save_image(recon_x.data.view(BATCH_SIZE, IMG_CHANNEL, 80, 120), RESULT_PATH + 'sample_' + str(EPOCH) + '.png')
            save_image(data.data.view(BATCH_SIZE, IMG_CHANNEL, 80, 120), RESULT_PATH + 'sample_' + str(EPOCH) + '_train.png')
            print('Train Epoch:{} -- [{}/{} ({:.0f}%)] -- Loss:{:.6f}'.format(
                EPOCH, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item() / len(data))
            )
    print('====> Epoch: {} Average loss: {:.4f}'.format(EPOCH, total_loss / len(train_loader.dataset)))
    save_tensor(vae.state_dict(), MODEL_PATH)



if __name__ == "__main__":
    if not os.path.isdir(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    vae = VAE().cuda()
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCH):
        train(epoch)

    # test()