import torch
from models.U_net import AttU_Net
from models.swinir import SwinIR
from models.deform import align_FG


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_A(opt):
    netA = align_FG(opt)
    netA.cuda()
    return netA


def define_G(gpu_ids=[], height=320, width=320, window_size=8):
    netG = None
    use_gpu = len(gpu_ids) > 0
    # norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    netG = SwinIR(upscale=1, img_size=(height, width), in_chans=6,
                  window_size=window_size, img_range=1., depths=[2, 2, 2],
                  embed_dim=64, num_heads=[2, 2, 2], mlp_ratio=2, upsampler='')
    netG.cuda()
    netG.apply(weights_init)
    return netG


def define_R(gpu_ids=[], skip=False, opt=None):
    netR = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    netR = AttU_Net(9, 3)
    netR.cuda()
    netR.apply(weights_init)
    return netR
