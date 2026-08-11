import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])


def preprocess(image):
    if isinstance(image, bytes):
        image = PIL.Image.open(io.BytesIO(image))
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    elif not isinstance(image, PIL.Image.Image):
        raise TypeError(f"Unsupported image type for preprocess: {type(image)}")

    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]