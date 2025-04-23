from torch.utils.data import Dataset, DataLoader
from PIL import Image

class RussianWildlifeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open the image
        image = Image.open(img_path).convert("RGB")
        
        # Apply any transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label