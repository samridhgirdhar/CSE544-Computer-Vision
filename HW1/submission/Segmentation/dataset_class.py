import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import csv
from tqdm import tqdm

class CamVidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_dict_path, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Read all image filenames
        self.image_names = sorted(os.listdir(images_dir))
        
        # If mask filenames match, we can just do the same
        self.mask_names = sorted(os.listdir(masks_dir))
        
        # Make color->classID mapping
        self.color2class = self.load_class_dict(class_dict_path)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Paths
        img_path = os.path.join(self.images_dir, self.image_names[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])
        
        # Open images
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("RGB")  # color-coded labels
        
        # Resize both image & mask

        image = image.resize((480, 360), Image.BILINEAR) 
        mask  = mask.resize((480, 360), Image.NEAREST)

                
        
        
        # Convert mask from color-coded to a 2D class index array
        mask_array = np.array(mask)       
        mask_index = self.rgb2class(mask_array)  
        
        # Convert image to tensor
        image_tensor = F.to_tensor(image) # shape [3, 360, 480], float in [0,1]
        
        # Normalize the image
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        image_tensor = F.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # Convert mask_index to torch long tensor
        mask_tensor = torch.from_numpy(mask_index).long()  # shape [360,480]
        
        return image_tensor, mask_tensor
    
    def load_class_dict(self, class_dict_path):
        """
        Reads class_dict.csv and returns a dict { (r,g,b): class_id, ... }.
        """
        color2class = {}
        with open(class_dict_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t') if '\t' in f.read() else csv.DictReader(open(class_dict_path,'r'))
            f.seek(0)
            # The CSV might have a header: name, r, g, b
            # We'll build an index in the order we encounter them
            idx = 0
            for row in reader:
                # Some CSVs might be separated by commas, 
                # so adapt your code as needed:
                r = int(row['r'])
                g = int(row['g'])
                b = int(row['b'])
                color2class[(r,g,b)] = idx
                idx += 1
        return color2class
    
    def rgb2class(self, mask_arr):
        """
        mask_arr: [H,W,3] with color-coded labels
        Return:   [H,W] with class indices
        """
        h, w, _ = mask_arr.shape
        out = np.zeros((h,w), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                rgb = tuple(mask_arr[i,j])
                if rgb in self.color2class:
                    out[i,j] = self.color2class[rgb]
                else:
                    # If we get a color not in dict, treat as 'Void' or background
                    out[i,j] = self.color2class.get((0,0,0), 0)
        return out
