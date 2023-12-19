import os
import h5py
import numpy as np
import urllib.request
from PIL import Image

# URL to the SVHN dataset file
url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"

# Download the dataset file to the Downloads folder
downloads_folder = os.path.expanduser("~/Downloads")
dataset_filename = os.path.join(downloads_folder, "svhn_train_32x32.mat")
urllib.request.urlretrieve(url, dataset_filename)

# Load the SVHN dataset
svhn_data = h5py.File(dataset_filename, "r")

# Access the images and labels
images = np.array(svhn_data["X"])
labels = np.array(svhn_data["y"])

# Close the dataset file
svhn_data.close()

# Specify the directory where you want to save the images
save_directory = os.path.join(downloads_folder, "svhn_images")

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Save the images as PNG files
for i in range(len(images)):
    image = Image.fromarray(images[i])
    label = labels[i][0] % 10  # Extract the label (digit)
    filename = os.path.join(save_directory, f"svhn_{i}_{label}.png")
    image.save(filename)
