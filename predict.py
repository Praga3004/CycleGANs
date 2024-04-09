import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from PIL import Image
import torchvision.transforms as transforms
import torch

# data (path)
dataset_name = 'raw'
root = '../dataset/'+dataset_name

# data (img)
img_height = 256
img_width = 256
channels = 3

# training
epoch = 0 # epoch to start training from
n_epochs = 5 # number of epochs of training
batch_size = 1 # size of the batches
lr = 0.0002 # adam : learning rate
b1 = 0.5 # adam : decay of first order momentum of gradient
b2 = 0.999 # adam : decay of first order momentum of gradient
decay_epoch = 3 # suggested default : 100 (suggested 'n_epochs' is 200)
n_cpu=0


#Generator

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()
        
        channels = input_shape[0]
        
        # Initial Convolution Block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]
            
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2), # --> width*2, heigh*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            
        # Output Layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()
                 ]
        
        # Unpacking
        self.model = nn.Sequential(*model) 
        
    def forward(self, x):
        return self.model(x)
input_shape = (channels, img_height, img_width) # (3,256,256)
n_residual_blocks = 9 # suggested default, number of residual blocks in gene

G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_AB = G_AB.cuda()
def load_checkpoint(file="cycleGans.pt"):
    """
    Load the saved checkpoint from the specified file.

    Args:
    - file (str): Path to the saved checkpoint file.

    Returns:
    - dict: Dictionary containing the loaded checkpoint.
    """
    return torch.load(file)

# Example usage:
checkpoint = load_checkpoint("cycleGans.pt")
epoch = checkpoint['epoch']
G_AB.load_state_dict(checkpoint['state_dict_G_AB'])
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image
def transform_image(image_path):
    """
    Transform a single image using the specified transformations.

    Args:
    - image_path (str): Path to the image file.
    - transform (callable): Transformation to apply to the image.

    Returns:
    - Tensor: Transformed image as a PyTorch tensor.
    """
    # Load the image
    image = Image.open(image_path)
    transforms_ = [
    transforms.Resize(int(img_height*1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform=transforms.Compose(transforms_)
    image=to_rgb(image)
    
    # Apply transformations
    transformed_image = transform(image)
    
    return transformed_image

def predict_single_image(image_path, generator):
    """
    Generate an image for a single input image using the specified generator.

    Args:
    - image_path (str): Path to the input image file.
    - generator (nn.Module): PyTorch generator model used for image generation.
    - transform (callable): Transformation to apply to the input image.

    Returns:
    - PIL.Image.Image: Generated image as a PIL image.
    """
    # Transform the input image
    input_image = transform_image(image_path)
    device = next(generator.parameters()).device
    input_image = input_image.to(device)
    
    # Make prediction
    with torch.no_grad():
        # Add batch dimension and generate fake image
        fake_image = generator(input_image.unsqueeze(0))
    
    # Convert the generated image tensor to a PIL image
    generated_image = transforms.ToPILImage()(fake_image.squeeze(0).cpu())
    
    return generated_image

# Example usage:
# Assuming 'G_AB' is your generator model and 'transform' is your transformation
image_path = "T:/pragadesStuff/myStuff/cycleGans/CycleGANs/dataset/raw/MRI/mri-4.png"
generated_image = predict_single_image(image_path, G_AB)
generated_image.convert("L").show() 
