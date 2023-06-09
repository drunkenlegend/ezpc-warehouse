from PIL import Image
import numpy as np
import sys
import os
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

def get_arr_from_image(path):
    img = Image.open(path).convert("RGB")
    arr = preprocess(img).numpy()
    return arr

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py image.png")
        sys.exit(1)

    image_path = sys.argv[1]
    image_array = get_arr_from_image(image_path)

    # Save the preprocessed image array as a NumPy file
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    npy_path = file_name + ".npy"
    np.save(npy_path, image_array)

    print(f"Preprocessed image saved as: {npy_path}")

