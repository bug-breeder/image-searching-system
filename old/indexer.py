import faiss
from config import DIMENSION

# # Load the vectors from the file
# feature_vectors = torch.load(os.path.join(DATASET_DIR, "image_features.pt"))

def create_faiss_index(feature_vectors):
  index = faiss.IndexFlatIP(DIMENSION)  # Create a FAISS index
  index.add(feature_vectors.cpu().numpy())  # Add the vectors to the index
  return index

# D, I = index.search(, 10)