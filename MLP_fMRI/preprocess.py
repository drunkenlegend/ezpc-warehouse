from PIL import Image
import numpy as np
import sys
import os
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Grayscale(),  # Convert to greyscale
        transforms.ToTensor(),
    ]
)


def get_arr_from_image(img):
    arr = preprocess(img).unsqueeze(0).cpu().detach().numpy()

    # broadcast to (1, 1, 54, 45, 45)
    arr = np.broadcast_to(arr, (1, 1, 54, 45, 45))

    # reshape to (1, 1, 45, 54, 45)
    arr = arr.reshape(1, 1, 45, 54, 45)

    return arr
