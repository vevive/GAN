from torch import nn
from configuration import Configuration as config

class Generator(nn.Module):
    '''生成器定义'''

    def __init__(self, config):
        super(Generator, self).__init__()
        Generator_feature = config.Generator_feature

        self.main = nn.Sequential(
            # 输入是一个nz维度的噪声，可以认为它是一个1*1*nz的feature map
            nn.ConvTranspose2d(config.noise,Generator_feature*8,4,1,0,bias=False),
            nn.BatchNorm2d(Generator_feature*8),
            nn.ReLU(True),
            #输出形状为(Generator_feature*8)*4*4

            nn.ConvTranspose2d(Generator_feature*8,Generator_feature*4,4,2,1,bias=False),
            nn.BatchNorm2d(Generator_feature*4),
            nn.ReLU(True),
            #(gf*4)*8*8

            nn.ConvTranspose2d(Generator_feature*4,Generator_feature*2,4,2,1,bias=False),
            nn.BatchNorm2d(Generator_feature*2),
            nn.ReLU(True),
            #(gf*2)*16*16

            nn.ConvTranspose2d(Generator_feature*2,Generator_feature,4,2,1,bias=False),
            nn.BatchNorm2d(Generator_feature),
            nn.ReLU(True),
            #gf*32*32

            nn.ConvTranspose2d(Generator_feature,3,5,3,1,bias=False),
            nn.Tanh()# 输出范围 -1~1 故而采用Tanh
            #3*96*96
        )

    def forward(self,input):
        return self.main(input)

class Discriminator(nn.Module):
    '''判别器定义'''
    def __init__(self,config):
        super(Discriminator,self).__init__()
        Discriminator_feature=config.Discriminator_feature

        self.main=nn.Sequential(
            #input 3*96*96
            nn.Conv2d(3,Discriminator_feature,5,3,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            #output Discriminator_feature*32*32

            nn.Conv2d(Discriminator_feature,Discriminator_feature*2,4,2,1,bias=False),
            nn.BatchNorm2d(Discriminator_feature*2),
            nn.LeakyReLU(0.2,inplace=False),
            #output (df*2)*16*16

            nn.Conv2d(Discriminator_feature*2,Discriminator_feature*4,4,2,1,bias=False),
            nn.BatchNorm2d(Discriminator_feature*4),
            nn.LeakyReLU(0.2,inplace=False),
            #output (df*4)*8*8

            nn.Conv2d(Discriminator_feature*4,Discriminator_feature*8,4,2,1,bias=False),
            nn.BatchNorm2d(Discriminator_feature*8),
            nn.LeakyReLU(0.2,inplace=False),
            #output (df*8)*4*4

            nn.Conv2d(Discriminator_feature*8,1,4,1,0,bias=False),
            nn.Sigmoid() # 输出一个数(概率)
        )
    def forward(self,input):
        return self.main(input).view(-1)