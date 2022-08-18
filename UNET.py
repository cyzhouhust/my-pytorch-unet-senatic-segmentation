import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import models
from torchvision.models.vgg import VGG
import cv2
import numpy as np
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '1' #解决cuda溢出问题
# 将标记图（每个像素值代该位置像素点的类别）转换为onehot编码
def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))           #
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

# 利用torchvision提供的transform，定义原始图片的预处理步骤（转换为tensor和标准化处理） 
transform = transforms.Compose([
    transforms.ToTensor(), 
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

# 利用torch提供的Dataset类，定义我们自己的数据集
class BagDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('./bag_data'))

    def __getitem__(self, idx):
        img_name = os.listdir('./bag_data')[idx]
        imgA = cv2.imread('./bag_data/'+img_name)
        imgA = cv2.resize(imgA, (160, 160))

        imgB = cv2.imread('./bag_data_mask/'+img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
       # imgB = imgB/255               #将图片转换为灰度0-1之间
        #imgB = imgB.astype('uint8')
        #imgB = onehot(imgB, 2)
        #imgB = imgB.transpose(2,0,1)
        imgB = torch.LongTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)
        return imgA, imgB
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



def get_boundary(pic,is_mask):
    if not is_mask:
        pic = torch.argmax(pic,1).cpu().numpy().astype('float64')
    else:
        pic = pic.cpu().numpy()
    batch, width, height = pic.shape
    new_pic = np.zeros([batch, width + 2, height + 2])
    mask_erode = np.zeros([batch, width, height])
    dil = int(round(0.02*np.sqrt(width ** 2 + height ** 2)))
    if dil < 1:
        dil = 1
    for i in range(batch):
        new_pic[i] = cv2.copyMakeBorder(pic[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for j in range(batch):
        pic_erode = cv2.erode(new_pic[j],kernel,iterations=dil)
        mask_erode[j] = pic_erode[1: width + 1, 1: height + 1]
    return torch.from_numpy(pic-mask_erode)

def get_biou(pre_pic ,real_pic):
    inter = 0
    union = 0
    pre_pic = get_boundary(pre_pic, is_mask=False)
    real_pic = get_boundary(real_pic, is_mask=True)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = real_pic[i]
        inter += ((predict * mask) > 0).sum()
        union += ((predict + mask) > 0).sum()
    if union < 1:
        return 0
    biou = (inter/union)
    return biou
def get_miou(pre_pic, real_pic):
    miou = 0
    pre_pic = torch.argmax(pre_pic,1)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = real_pic[i]
        union = torch.logical_or(predict,mask).sum()
        inter = ((predict + mask)==2).sum()
        if union < 1e-5:
            return 0
        miou += inter / union
    return miou/batch

# 实例化数据集
bag = BagDataset(transform)

train_size = int(0.85 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

# 利用DataLoader生成一个分batch获取数据的可迭代对象
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

epochs=30
lr=1e-5
model=UNet(3,2)
loss=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=lr)
test_Acc = []
test_mIou = []
test_biou=[]

for epoch in range(epochs):
    tm=[]
    tb=[]
    for X,y in train_loader:
        total_loss=[]
        opt.zero_grad()
        y_hat=model(X)
        l=loss(y_hat, y)
        total_loss.append(l)
        l.backward()
        opt.step()
    total_loss=sum(total_loss)
    print('epoch:',epoch+1,'loss:',total_loss)


    with torch.no_grad():
        ttotal_loss=[]
        for tx,ty in test_loader:
            ty_hat=model(tx)
            tl=loss(ty_hat,ty)
            ttotal_loss.append(tl)
        ttotal_loss=sum(ttotal_loss)
        print('test_loss:',ttotal_loss)
        tm.append(get_miou(ty_hat, ty))
        tb.append(get_biou(ty_hat, ty))
    miou=sum(tm) / len(tm)
    biou=sum(tb) / len(tb)
    print('miou:', miou.item())
    print('biou:', biou.item())
    torch.save(model.state_dict(), 'checkpoints/unet_model_{}.pth'.format(epoch))







'''
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
'''