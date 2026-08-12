import os
import glob
import uuid
import PIL.Image
import cv2
import numpy as np

try:
    import torch
    import torch.utils.data
    BaseDataset = torch.utils.data.Dataset
except Exception:
    torch = None
    BaseDataset = object


class XYDataset(BaseDataset):
    def __init__(self, directory, categories, transform=None, random_hflip=False):
        super(XYDataset, self).__init__()
        self.directory = directory
        self.categories = categories
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        cv_img = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        if cv_img is None:
            raise FileNotFoundError(f"Could not read image: {ann['image_path']}")
        
        # Convert BGR to RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width = cv_img.shape[:2]
        
        x = 2.0 * (ann['x'] / width - 0.5)  # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom
        
        # Horizontal Flip data augmentation
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            cv_img = cv2.flip(cv_img, 1) # horizontal flip
            x = -x
            
        pil_img = PIL.Image.fromarray(cv_img)
        if self.transform is not None:
            pil_img = self.transform(pil_img)
            
        if torch is not None:
            import torchvision.transforms.functional as F
            # Convert PIL image to Normalized PyTorch Tensor (3, H, W)
            tensor_img = F.to_tensor(pil_img)
            tensor_img = F.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            return tensor_img, ann['category_index'], torch.Tensor([x, y])
        else:
            return pil_img, ann['category_index'], np.array([x, y], dtype=np.float32)

    
    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        x = items[0]
        y = items[1]
        return int(x), int(y)
        
    def refresh(self):
        self.annotations = []
        for category in self.categories:
            category_index = self.categories.index(category)
            for image_path in glob.glob(os.path.join(self.directory, category, '*.jpg')):
                try:
                    x, y = self._parse(image_path)
                    self.annotations += [{
                        'image_path': image_path,
                        'category_index': category_index,
                        'category': category,
                        'x': x,
                        'y': y
                    }]
                except Exception:
                    pass
        
    def save_entry(self, category, image, x, y):
        category_dir = os.path.join(self.directory, category)
        os.makedirs(category_dir, exist_ok=True)
            
        filename = '%d_%d_%s.jpg' % (x, y, str(uuid.uuid1()))
        
        image_path = os.path.join(category_dir, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
        
    def get_count(self, category):
        i = 0
        for a in self.annotations:
            if a['category'] == category:
                i += 1
        return i


class HeatmapGenerator():
    def __init__(self, shape, std):
        import torch
        self.shape = shape
        self.std = std
        self.idx0 = torch.linspace(-1.0, 1.0, self.shape[0]).reshape(self.shape[0], 1)
        self.idx1 = torch.linspace(-1.0, 1.0, self.shape[1]).reshape(1, self.shape[1])
        self.std = std
        
    def generate_heatmap(self, xy):
        import torch
        x = xy[0]
        y = xy[1]
        heatmap = torch.zeros(self.shape)
        heatmap -= (self.idx0 - y)**2 / (self.std**2)
        heatmap -= (self.idx1 - x)**2 / (self.std**2)
        heatmap = torch.exp(heatmap)
        return heatmap