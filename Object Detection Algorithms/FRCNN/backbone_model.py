from torchvision.models import vgg16
from torchvision.models.vgg import VGG16_Weights
import torch
import torch.nn as nn
import torchvision.models as models

def vgg16_model(sub_samples,dummy_image):
    # Assuming dummy_img is your input image tensor with shape (batch_size, 3, 800, 800)
    # Load pre-trained VGG16 model
    vgg16 = models.vgg16(pretrained=True)

    # Extract features
    features = list(vgg16.features)
    input_features=dummy_image.shape[1]
    # Initialize an empty list to store the features at different resolutions
    output_features = []
    backbone_features=[]
    # Iterate through layers and apply to the input tensor
    for layer in features:
        input_tensor = layer(input_tensor)
        output_features.append(input_tensor.clone())

        # Check if spatial dimensions are smaller than 800//16 (50)
        if input_tensor.size()[2] < input_features // sub_samples:
            break
        backbone_features.append(layer)
    ffcn_model=nn.Sequential(*backbone_features)
    return ffcn_model
