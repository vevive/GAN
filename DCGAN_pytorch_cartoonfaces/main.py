import os
import ipdb
import torch
import torchvision
import tqdm
import torch.nn as nn
from model import Generator, Discriminator
from torchnet.meter import AverageValueMeter
from configuration import Configuration as config


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(config, k_, v_)

    device = torch.device("cuda:0" if config.gpu else "cpu")
    if config.vis:
        from visdom import Visdom
        vis = Visdom(config.env)

    # dataset
    dataset = torchvision.datasets.ImageFolder(root=config.data_path,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.Resize(
                                                       config.image_size),
                                                   torchvision.transforms.CenterCrop(
                                                       config.image_size),
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)

    # networks
    netG = Generator(config).to(device)
    netD = Discriminator(config).to(device)
    if config.netg_path:
        netG.load_state_dict(torch.load(config.netg_path))
    if config.netd_path:
        netD.load_state_dict(torch.load(config.netd_path))

    # 定义损失函数和优化器
    criterion = nn.BCELoss().to(device)
    optimizer_g = torch.optim.Adam(
        netG.parameters(), config.lr1, betas=(config.beta1, 0.999))
    optimizer_d = torch.optim.Adam(
        netD.parameters(), config.lr2, betas=(config.beta1, 0.999))

    true_label = 1
    fake_label = 0
    fixed_noise = torch.randn(
        config.batch_size, config.noise, 1, 1, device=device)
    noises = torch.randn(config.batch_size, config.noise, 1, 1).to(device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    for epoch in range(config.max_epoch):
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            if ii % config.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                # 尽可能的把真图片判别为正确
                output = netD(real_img)
                error_d_real = criterion(output, true_label)
                error_d_real.backward()

                # 尽可能把假图片判别为错误
                noises.data.copy_(
                    torch.randn(config.batch_size, config.noise, 1, 1))
                fake_img = netG(noises).detach()
                output = netD(fake_img)
                error_d_fake = criterion(output, fake_label)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake+error_d_real
                errord_meter.add(error_d.item())

            if ii % config.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(torch.randn(
                    config.batch_size, config.noise, 1, 1))
                fake_img = netG(noises)
                output = netD(fake_img)
                error_g = criterion(output, true_label)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            if config.vis and ii % config.plot_every == config.plot_every-1:
                # 可视化
                if os.path.exists(config.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netG(fixed_noise)
                vis.images(fix_fake_imgs.detach().cpu().numpy()
                           [:64]*0.5+0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()
                           [:64]*0.5+0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])

        if(epoch+1) % config.save_every == 0:
            # 保存图片模型
            torchvision.utils.save_image(fix_fake_imgs.data[:64], '%s%s.png' % (
                config.save_path, epoch), normalize=True, range=(-1, 1))
            torch.save(netD.state_dict(), '/checkpoints/netD_%s.pth' % epoch)
            torch.save(netG.state_dict(), '/checkpoints/netG_%s.pth' % epoch)
            errord_meter.reset()
            errorg_meter.reset()


@torch.no_grad()
def generate(**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(config, k_, v_)

    device = torch.cuda.device('cuda:0')

    netG, netD = Generator(config).eval(), Discriminator(config).eval()
    noises = torch.randn(config.gen_search_num, config.noise, 1, 1).normal_(
        config.gen_mean, config.gen_std)
    noises = noises.to(device)

    netD.load_state_dict(torch.load(config.netd_path)).to(device)
    netG.load_state_dict(torch.load(config.netg_path)).to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = netG(noises)
    scores = netD(fake_img).detach()

    # 挑选最好的某几张
    indexs = scores.topk(config.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    torchvision.utils.save_image(torch.stack(
        result), config.gen_img, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    import fire
    fire.Fire()
