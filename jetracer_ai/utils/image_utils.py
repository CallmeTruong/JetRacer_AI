"""
Image utility and preprocessing functions for JetRacer AI.
"""
import io
import cv2
import PIL.Image
import numpy as np


def bgr8_to_jpeg(value, quality=75):
    """Encode OpenCV BGR image into JPEG bytes."""
    if value is None:
        return b''
    return bytes(cv2.imencode('.jpg', value, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1])


def preprocess_onnx(image):
    """
    Pure NumPy/OpenCV image preprocessing for ONNX Runtime inference.
    Zero PyTorch / CUDA dependency.
    """
    if isinstance(image, np.ndarray):
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = np.array(image)

    img_float = image_rgb.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_normalized = (img_chw - mean) / std

    return np.expand_dims(img_normalized, axis=0)


def preprocess(image):
    """
    PyTorch tensor preprocessing (Lazy PyTorch import).
    """
    import torch
    import torchvision.transforms as transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    if isinstance(image, bytes):
        image = PIL.Image.open(io.BytesIO(image))
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    elif not isinstance(image, PIL.Image.Image):
        raise TypeError(f"Unsupported image type for preprocess: {type(image)}")

    image_tensor = transforms.functional.to_tensor(image).to(device)
    image_tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image_tensor[None, ...]
