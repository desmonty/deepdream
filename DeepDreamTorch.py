import math
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.autograd import Variable


from PIL import Image

from DeepDreamTorchUtils import *

#######################
# DeepDream Algorithm #
#######################

class DeepDream(object):
    """docstring for DeepDream"""
    def __init__(self, image, network, device):
        super(DeepDream, self).__init__()
        self.image = image
        self.network = network
        self.device = device
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

    def apply(self, at_layer, iterations: int, lr: float, octave_scale: float, num_octaves: int=None, blend: float=0.5):
        """
        Dream the image.

        Note: If only one of 'octave_scale' and 'num_octaves' are provided
              the other one will be automatically computed.
              IT IS NOT GUARANTEED TO WORK. 

        :param at_layer: (int, str) Layer to look at during dreaming.
        :param iterations: (int) Number of iterations of the Dream algorithms.
        :param lr: (float) Learning_rate of the Dream algorithm
        :param octave_scale: (float) Scale factor for the reduction of the image between two ocaves.
        :param num_octaves: (int) Number of octaves (Times we'll apply the scale factor)
        :param blend: (float) Blend factor between dreamed_image and normal image at each scale.
        :return: Dreamt image, transformed so that its nice.
        """
        #
        model = self.construct_model(self.network, at_layer)
        input_image = PIL_to_Tensor(self.image).to(self.device)
        
        # Complete parameters if not given.
        if num_octaves is None:
            min_dimension = min(input_image.size()[1:])
            num_octaves = math.floor(math.log(min_dimension / 32) / math.log(octave_scale))
        
        if octave_scale is None:
            max_dimension = max(input_image.size()[1:])
            octave_scale = math.pow(max_dimension / 224, 1.0 / num_octaves)

        # Transform the image in Tensor and preprocess it.
        input_image = to_PIL(input_image)
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        input_image = preprocess(input_image).unsqueeze(0).to(self.device)
        
        # Generate zoomed verion of the original image, and reapply to get more.
        octaves = [input_image]
        for _ in range(num_octaves - 1):
            new_octave = resize(octaves[-1], factor=1/octave_scale)
            octaves.append(new_octave)

        detail = torch.zeros_like(octaves[-1]).to(self.device)
        
        print("Octave: ", end="")
        for octave, octave_base in enumerate(octaves[::-1]):
            if octave > 0:
                # Upsample detail to new octave dimension
                detail = resize(detail, size=list(octave_base.size()[2:4]))

            # Add deep dream detail to new octave dimension
            input_image = 2.0 * (blend * octave_base + (1.0 - blend) * detail)

            # Get new deep dream image
            dreamed_image = self.dream(input_image, model, iterations, lr)
            
            # Extract deep dream details
            detail = dreamed_image - octave_base

            print(octave+1, end=" ")

        dreamed_image = to_PIL(self.deprocess(dreamed_image))
        print("Done !")
        return dreamed_image

    def dream(self, input: torch.Tensor, model, iterations: int, lr: float):
        """
        Main Dream function.
        Updates the image to maximize outputs for n iterations

        :param input: (Tensor) Image to dream on.
        :param model: Complete model to use for the dreaming.
        :param iterations: (int) Number of times we want to step toward the dream.
        :param lr: (float) Learning rate.
        :return: Dreamed image.
        """
        # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        input = Variable(input.to(self.device), requires_grad=True)
        for i in range(iterations):
            model.zero_grad()
            out = model(input)
            loss = out.norm()
            loss.backward()
            avg_grad = torch.abs(input.grad).mean()
            norm_lr = lr / avg_grad
            input.data += norm_lr * input.grad.data
            input.data = self.clip(input.data)
            input.grad.data.zero_()
        return input

    # Helpers
    def clip(self, image_input: torch.Tensor):
        """
        Clip image according to mean and std in the network.

        :param image_input: (Tensor) Image.
        :return: Clipped image.
        """
        for c in range(3):
            m, s = self.mean[c], self.std[c]
            image_input[0, c] = torch.clamp(image_input[0, c], -m / s, (1 - m) / s)
        return image_input

    def deprocess(self, image_input: torch.Tensor):
        """
        Deprocess the image_input. Basically just unnormalize it.

        :param image_input: (Tensor) Image to unnormalize.
        :return: Deprocessed image.
        """
        image_input = image_input.squeeze() * self.std.reshape((3, 1, 1)) + self.mean.reshape((3, 1, 1))
        image_input = torch.clamp(image_input, min=0.0, max=1.0)
        return image_input

    def construct_model(self, network, at_layer):
        """
        Return a Sequential model ending by the 'at_layer'nth layer.

        :param network: (torchvision.models) Network used for the image recognition.
        :param at_layer: Layer where to stop the computation.
        :return: Model that return the output at layer 'at_layer' when computing on model 'network'
        """
        layers = list(network.features.children())
        model = nn.Sequential(*layers[: (at_layer + 1)])
        return model.to(self.device)
