import os
import torch
from constants import DATASET
from utils.encoder import Encoder


def extract_features():
    # Load your dataset of images
    # script_dir = os.path.dirname(__file__)
    # parent_parent_dir = os.path.dirname(os.path.dirname(script_dir))
    image_dir = os.path.join("dataset", "ILSVRC2012_img_val")
    print("Extracting image features from", image_dir)

    image_paths = [
        os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
    ]

    encoder = Encoder()

    all_image_features, image_names = encoder.get_all_image_features(image_paths)
    print(all_image_features.shape)

    # Save the features and image names to a file
    torch.save(all_image_features, os.path.join("dataset", "image_features.pt"))
    with open(os.path.join("dataset", "image_names.txt"), "w") as f:
        for image_name in image_names:
            f.write(image_name + "\n")

    print("Image features saved to dataset/image_features.pt")


if __name__ == "__main__":
    extract_features()
