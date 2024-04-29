import os
import torch
from encoder import Encoder
from constants import DATASET_DIR

def extract_features():
  # Load your dataset of images
  image_dir = os.path.join(DATASET_DIR, "ILSVRC2012_img_val")
  image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]

  encoder = Encoder()

  all_image_features, image_names = encoder.get_all_image_features(image_paths)

  # Save the features and image names to a file
  torch.save(all_image_features, os.path.join(DATASET_DIR, "image_features.pt"))
  with open(os.path.join(DATASET_DIR, "image_names.txt"), "w") as f:
    for image_name in image_names:
      f.write(image_name + "\n")

if __name__ == "__main__":
  extract_features()