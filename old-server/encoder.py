import torch
import os
from PIL import Image
import clip
from tqdm import tqdm


class Encoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def preprocess_text(self, text):
        text = "a photo of " + text.lower()  # Convert text to lowercase
        text = clip.tokenize([text]).to(self.device)
        return text

    def preprocess_images(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def get_single_image_features(self, image_path):
        image = self.preprocess_images(image_path)
        with torch.no_grad():
            image_features = self.model.encode_image(image)

        return image_features

    def get_all_image_features(self, image_paths):
        # Initialize a tensor to store all image features
        all_image_features = torch.empty((len(image_paths), 512)).to(self.device)

        # Initialize a list to store image names
        image_names = []

        # Extract features for each image
        for i, image_path in enumerate(tqdm(image_paths)):
            image_features = self.get_single_image_features(image_path)
            all_image_features[i] = image_features

            # Get the image name from the file path
            image_name = os.path.basename(image_path)
            image_names.append(image_name)

        return all_image_features, image_names

    def get_single_text_features(self, text):
        text = self.preprocess_text(text)
        with torch.no_grad():
            text_features = self.model.encode_text(text)

        return text_features
