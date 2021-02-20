import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class Mobticon_crop_dataloader(Dataset):
    def __init__(self, image_dir, thermal=False, resize=(512,380),
                 classNum=4):
        self.thermal = thermal
        self.resize = resize
        self.classNum = classNum

        self.class_list = []
        if(self.classNum==4):
            self.class_list = ['dry', 'wet', 'snow', 'ice']
        else:
            self.class_list = ['none', 'blackIce']

        #### listing
        self.gt = []
        self.image_list = []
        self.thermal_list = []

        for condition in self.class_list:
            dist_list = ["05", "10", "15"]
            for dist in dist_list:
                img_dir = os.path.join(image_dir, condition, dist, "Img")
                thermal_dir = os.path.join(image_dir, condition, dist, "Thermal")

                tmp_image_list = os.listdir(img_dir)
                tmp_image_list = [os.path.join(img_dir, item) for item in tmp_image_list]
                tmp_thermal_list = os.listdir(thermal_dir)
                tmp_thermal_list = [os.path.join(thermal_dir, item) for item in tmp_thermal_list]

                self.image_list.extend(tmp_image_list)
                self.thermal_list.extend(tmp_thermal_list)

                tmp_gt = [str(self.class_list.index(condition))] * len(tmp_image_list)
                self.gt.extend(tmp_gt)



        # classV = randomKey[:-2]
        # self.gt.extend(str(self.class_list.index(classV)))

        # self.image_list.append(os.path.join(image_dir, classV, randomKey[-2:], "Img", randomValueImg))
        # self.thermal_list.append(os.path.join(image_dir, classV, randomKey[-2:], "Thermal", randomValueThermal))


        #### config transform
        self.transform_dict = {
            "init": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize)
            ]),
            "vis_norm": transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "fir_norm": transforms.Compose([
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        }

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #### image config
        image = Image.open(self.image_list[idx])
        image = self.transform_dict["vis_norm"](self.transform_dict["init"](image))

        #### gt config
        gt = [0]*len(self.class_list)
        # gt[self.class_list.index(self.gt[idx])] = 1
        gt[int(self.gt[idx])] = 1

        #### get thermal image
        if self.thermal:
            thermal = Image.open(self.thermal_list[idx])
            thermal = self.transform_dict["fir_norm"](self.transform_dict["init"](thermal))

            return {'input' : torch.cat((thermal, image), dim=0),
                    'label' : torch.tensor(gt),
                    'img_path' : self.image_list[idx]}
        else:
            return {'input' : image,
                    'label': torch.tensor(gt),
                    'img_path' : self.image_list[idx]}


