import numpy as np
import torch

from torchvision import transforms
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile


def PIL_to_Tensor(image: Image):
    """
    Transform a PIL image into a Torch tensor.

    :param image: PIL image.
    :return: Tensor version of the PIL image.
    """
    loader = transforms.Compose([
        transforms.ToTensor()
    ])
    return loader(image)

def to_PIL(image: torch.Tensor):
    """
    Convert Tensor to PIL Image

    :param image: Tensor version of the PIL image.
    :return: PIL image.
    """
    return transforms.functional.to_pil_image(image.cpu())

def load_image(path: str):
    """
    Return a loaded image from a given path.

    :param path: (str) relative path of the iamge.
    :return: PIL image.
    """
    image = Image.open(path)
    return image

def save_image(image, filename: str):
    """
    Save image 'image' at path 'filename'

    :param image: Image to save.
    :param filename: (str) Path where to save the image.
    :return: None
    """
    # Convert Tensor into PIL Image if necessary
    if type(image) != Image.Image:
        tmp_image = to_PIL(image)
    else:
        tmp_image = image
    with open(filename, 'wb') as file:
        tmp_image.save(filename, 'jpeg')
    
def plot_image(image: torch.Tensor):
    """
    Plot an image

    :param image: Image to plot.
    :return: None
    """
    # Convert Tensor to PIL Image if necessary
    if type(image) == torch.Tensor:
        image = to_PIL(image)
    display(image)

def resize(inputs, size: tuple=None, factor: float=None, interpolation='bilinear', device='cuda:0'):
    """
    Resize  an image given a target size

    :param inputs: (Tensor, PIL.Image) Image
    :param size: (tuple<int, int>) Size of the new image.
    :param factor: (float) Scale factor by which multiply the dimension of the image.
    :param interpolation: (str) interpolation used
    :param device: (str) Device where to compute the model.
    :return: (Tensor, PIL.Image) resized image
    """
    # Get GPU if there is one
    if not torch.cuda.is_available() or 'cuda:' not in device:
        device = "cpu"

    is_image = False
    if type(inputs) != torch.Tensor:
        is_image = True
        inputs = PIL_to_Tensor(inputs).to(device).unsqueeze(0)

    with torch.no_grad():
        model = torch.nn.Upsample(
            size=size,
            scale_factor=factor,
            mode=interpolation,
            align_corners=True
        ).to(device)
        outputs = model(inputs)
    
    if is_image:
        outputs = to_PIL(outputs.squeeze())

    return outputs
    