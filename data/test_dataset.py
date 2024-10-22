import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class PairDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'low')
        self.dir_B = os.path.join(opt.dataroot, 'high')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if opt.mode == 'dynamic':
            self.dir_A2 = os.path.join(opt.dataroot, 'fake')
            self.A2_paths = make_dataset(self.dir_A2)
            self.A2_paths = sorted(self.A2_paths)
            self.A2_size = len(self.A2_paths)

        transform_list = []

        transform_list += [transforms.ToTensor()]
        # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255
        # transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        if self.opt.mode == 'dynamic':
            A2_path = self.A2_paths[index % self.A2_size]
            A2_img = Image.open(A2_path).convert('RGB')
            A2_img = self.transform(A2_img)
            return {'A': A_img, 'A2': A2_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'PairDataset'
