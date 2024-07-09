from share import *

import torch.multiprocessing


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    resume_path = 'models/control_v11f1p_sd15_depth.pth'
    batch_size = 8
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('models/cldm_v15.yaml').cpu()
    # pretrained_weights = torch.load(input_path)
    target_dict = {}
    pretrained_weights = load_state_dict(resume_path, location='cpu')
    for k in pretrained_weights.keys():
        target_dict[k] = pretrained_weights[k].clone()

    pretrained_weights = load_state_dict('models/v1-5-pruned.ckpt', location='cpu')
    for k in pretrained_weights.keys():
        target_dict[k] = pretrained_weights[k].clone()

    model.load_state_dict(target_dict,strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
  


    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    # checkpoint_callback = ModelCheckpoint(save_weights_only=True,every_n_epochs=1)
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=32, callbacks=[logger])


    # Train!
    trainer.fit(model, dataloader)
