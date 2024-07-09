import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import cv2
from pytorch_lightning import seed_everything
import collections


def  get_test_batch(bs=1):
    image_base ='/hdd2/liuxiaoyu/12_21_test/sample_depth'
    image_list =['4_lion.png','4_Tiger_0.3.png','4_wolf.png','14_square chair.jpg',
        '15_boots.png','22_horse.jpg','23_llama.jpg','24_duck.jpg',
        '33_wolf.jpg','34_lion.jpg','107_cat holding apple.jpg','107_panda holding apple.jpg',
        '11_motorcycle.png','33_wolf.jpg','117_tiger riding a bike.png','118_a bear holding a sign.jpg',
        '119_bear playing guitar.jpeg','117_A cat riding a car.png','120_tiger on surfboard.jpg','111_turtle.png']
    batch_list={}
    source_list =[]
    prompt_list=[]
    for image_name in image_list[(bs-1)*4:bs*4]:
        print(os.path.join(image_base,image_name))
        source_filename = os.path.join(image_base,image_name)
        prompt = 'a photo of ' +image_name.split("_")[1].split('.')[0]
        print(prompt)
       
        source = cv2.imread( source_filename)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)[:,:512,:]

        source = source.astype(np.float32) / 255.0
        source_list.append(torch.from_numpy(source).unsqueeze(0))
        prompt_list.append(prompt)
    batch_list['jpg'] =  torch.concat(source_list,dim=0).cuda()
    batch_list['hint'] =  torch.concat(source_list,dim=0).cuda()
    batch_list['c_control'] =  torch.concat(source_list,dim=0).cuda()
    batch_list['c_mask'] =  torch.concat(source_list,dim=0).cuda()
    batch_list['txt'] =prompt_list
    return batch_list

    
        
        

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            
            if pl_module.current_epoch%1==0:
                new_state_dict = collections.OrderedDict()
                for k, v in pl_module.state_dict().items():
                    if 'c_pre_list' in k:
                        new_state_dict[k] = v
                base_name ='lightning_logs/'
                torch.save(new_state_dict, os.path.join(base_name,str(pl_module.current_epoch)+'.ckpt'))
                # self.log_local(pl_module.logger.save_dir, split, images,
                #             pl_module.global_step, pl_module.current_epoch, batch_idx)

           

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
