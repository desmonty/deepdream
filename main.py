import torch

from torchvision import models

from DeepDreamTorch import *
from DeepDreamTorchUtils import *

if __name__ == "__main__":
    # Get GPU if there is one
    if torch.cuda.is_available():
        print("GPU found.")
        device = "cuda:0"
    else:
        device = "cpu"

    # Get network
    network = models.vgg19(pretrained=True)
    
    image = load_image('images/dark_forest_2.jpg')
    print("Image Loaded.")

    image = resize(image, factor=1)

    dreamMachine = DeepDream(image, network, device)
    dreamed_image = dreamMachine.apply(
        at_layer=13,
        iterations=40,
        lr=0.005,
        octave_scale=1.4,
        num_octaves=7
    )

    save_image(dreamed_image, "dreamed_images/dark_forest_2_dreamed.jpg")