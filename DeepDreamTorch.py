import math
import random
import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

from torchvision import models
from torchvision import transforms, utils

from PIL import Image
from scipy.ndimage.filters import gaussian_filter

# Helpers for images

def to_PIL(image: torch.Tensor):
    """
    Convert Tensor to PIL Image
    """
    return transforms.functional.to_pil_image(image.cpu())

def load_image(path: str):
    """
    Return a loaded image from a given path
    """
    image = Image.open(path)
    return image

def save_image(image: torch.Tensor, filename: str):
    """
    Save image 'image' at path 'filename'
    """
    # Convert Tensor into PIL Image
    image = to_PIL(image)
    with open(filename, 'wb') as file:
        image.save(filename, 'jpeg')
    
def plot_image(image: torch.Tensor):
    """
    Plot an image
    """
    # Convert Tensor to PIL Image
    image = to_PIL(image)
    image.show()

def resize_image(image: torch.Tensor, size: list=None, factor: float=None, interpolation=Image.BILINEAR):
    """
    Resize image according to size OR factor
    """
    if factor is not None:
        print(image.size())
        size = np.array(image.size()[1:3]) * factor
        size = size.astype(int)
    else:
        size = size[0:2]
    
    size = tuple(size)
    image = to_PIL(image)
    
    resize_transformation = transforms.Compose([
        transforms.Resize(size, interpolation=interpolation),
        transforms.ToTensor()
    ])
    image = resize_transformation(image)
    return torch.Tensor(image).to(device)

#######################
# DeepDream Algorithm #
#######################

# Helpers

def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np

def construct_model(network, at_layer):
    """
    Return a Sequential model ending by the 'at_layer'nth layer.
    """
    layers = list(network.features.children())
    model = nn.Sequential(*layers[: (at_layer + 1)])
    if torch.cuda.is_available:
        model = model.cuda()
    return model

def dream(image, model, iterations, lr):
    """
    Updates the image to maximize outputs for n iterations
    """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()

def deepdream(image, network, at_layer, iterations, lr, octave_scale, num_octaves):
    """
    Main methode
    """
    model = construct_model(network, at_layer)

    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    octaves = [image]
    # Generate zoomed verion of the original image, and reapply to get more.
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[:-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, int(np.array(octave_base.shape) / np.array(detail.shape)), order=1)
        # Add deep dream detail to new octave dimension
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)

if __name__ == "__main__":
    # Get GPU if there is one
    if torch.cuda.is_available():
        print("GPU found.")
        device = "cuda:0"
    else:
        device = "cpu"

    # Get original GoogLeNet
    network = models.vgg19(pretrained=True)
    model = construct_model(network, 27)
    print(model)
    
    cannelle = load_image('cannelle.jpg')
    print("Image Loaded.")

    dreamed_image = deepdream(
        cannelle,
        network,
        at_layer=27,
        iterations=20,
        lr=0.01,
        octave_scale=1.4,
        num_octaves=10
    )

    save_image(dreamed_image, "dreamed_cannelle.jpg")
