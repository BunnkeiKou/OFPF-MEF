import numpy as np
import torch
from torch import nn

from .base_model import BaseModel
from . import networks


def latent2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


class SingleModel(BaseModel):
    def name(self):
        return 'OFPF-MEF'

    # opt为参数类
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, 600, 400)
        self.input_A2 = self.Tensor(nb, opt.input_nc, 600, 400)
        self.input_B = self.Tensor(nb, opt.output_nc, 600, 400)
        self.netA = networks.define_A(self.opt)
        self.netG = networks.define_G(gpu_ids=[], window_size=4)
        self.refinement_net = networks.define_R()

        if not self.isTrain or opt.continue_train:
            print("---is not train----")
            which_epoch = 400
            print("---model is loaded---, continue")
            self.load_network(self.netA, 'A', which_epoch)
            self.load_network(self.netG, 'G', which_epoch)
            self.load_network(self.refinement_net, 'R', which_epoch)

        print('---------- Networks initialized -------------')

        self.netA.eval()
        # self.netA.eval()
        self.netG.eval()
        self.refinement_net.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'

        self.input_A.resize_(input['A'].size()).copy_(input['A'])

        self.input_B.resize_(input['B'].size()).copy_(input['B'])
        if self.opt.mode == 'dynamic':
            self.input_A2.resize_(input['A2'].size()).copy_(input['A2'])
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def predict(self):
        self.avg_pool = nn.AvgPool2d(2, 2)
        with torch.no_grad():
            if self.opt.mode == 'dynamic':
                self.align_B1, self.align_B2 = self.netA.forward(self.input_B, self.input_A2)
                self.input_A1 = self.avg_pool(self.input_A)
                self.input = torch.cat((self.input_A1, self.align_B1), dim=1)
                self.o_lf = self.netG.forward(self.input)
                self.detail = self.refinement_net.forward(
                    torch.cat((self.o_lf, self.input_A, self.align_B2), 1))

            else:
                self.input = torch.cat((self.input_A, self.input_B), dim=1)
                self.input = self.avg_pool(self.input)
                self.o_lf = self.netG.forward(self.input)
                self.detail = self.refinement_net.forward(
                    torch.cat((self.o_lf, self.input_A, self.input_B), 1))

            self.refinement = self.o_lf + self.detail
        output = latent2im(self.refinement.data)
        return output

    # get image paths
    def get_image_paths(self):
        return self.image_paths

