import json
import cv2
import numpy as np
import os
import random



from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('data/data.json', 'rt', encoding='utf-8') as f:
            count =0
            for line in f:
                count = count+1
                self.data.append(json.loads(line))
        self.ref_num =1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']


        source = cv2.imread( source_filename)
        target = cv2.imread( target_filename)
      

        # Do not forget that OpenCV read images in BGR order.
       
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)[:,:512,:]
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)[:,512:,:]
     

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target,hint=source, txt=prompt)

