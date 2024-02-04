from indexer import create_faiss_index
from encoder import Encoder

class ImageRetrievalSystem:
    def __init__(self, feature_vectors):
        self.index = create_faiss_index(feature_vectors)
        self.encoder = Encoder()

    def search_text(self, text, k=5):
        text_features = self.encoder.get_single_text_features(text)
        text_features = text_features.cpu().numpy()
        D, I = self.index.search(text_features, k)
        return I
    
    def search_image(self, image_path, k=5):
        image_features = self.encoder.get_single_image_features(image_path)
        image_features = image_features.cpu().numpy()
        D, I = self.index.search(image_features, k)
        return I

    def search_image_by_features(self, image_features, k=5):
        image_features = image_features.cpu().numpy()
        D, I = self.index.search(image_features, k)
        return I