import torch
import os
from PIL import Image
import clip
from tqdm import tqdm
from constants import BATCH_SIZE

class Encoder:
    def __init__(self, batch_size=BATCH_SIZE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.batch_size = batch_size

    def preprocess_text(self, text):
        text = "a photo of " + text.lower()  # Convert text to lowercase
        text = clip.tokenize([text]).to(self.device)
        return text

    def preprocess_images(self, image_paths):
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        return torch.stack([self.preprocess(image).to(self.device) for image in images])

    def get_single_image_features(self, image_path):
        image = self.preprocess_images(image_path)
        with torch.no_grad():
            image_features = self.model.encode_image(image)

        return image_features

    def get_image_features_batch(self, image_paths):
        image_batches = [image_paths[i:i+self.batch_size] for i in range(0, len(image_paths), self.batch_size)]
        all_image_features = []

        for batch in tqdm(image_batches):
            images = self.preprocess_images(batch)
            with torch.no_grad():
                batch_features = self.model.encode_image(images)
            all_image_features.append(batch_features)

        return torch.cat(all_image_features)

    def get_all_image_features(self, image_paths):
        all_image_features = self.get_image_features_batch(image_paths)

        # Initialize a list to store image names
        image_names = [os.path.basename(image_path) for image_path in image_paths]

        return all_image_features, image_names

    def get_single_text_features(self, text):
        text = self.preprocess_text(text)
        with torch.no_grad():
            text_features = self.model.encode_text(text)

        return text_features
