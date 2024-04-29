import numpy as np
import os
import torch
from config import DATASET_DIR, RANDOM_STATE
from retrieval_system import ImageRetrievalSystem
from sklearn.model_selection import train_test_split
import re
from encoder import Encoder
from tqdm import tqdm
from indexer import create_faiss_index
from evaluate import mean_average_precision

def format_label(string):
  return '_'.join(string.split('_')[:-1])

if __name__ == "__main__":

  image_dir = os.path.abspath(os.path.join("datasets", "paris"))
  image_paths = [os.path.join(image_dir, filename)
                 for filename in os.listdir(image_dir)]
  # image_paths = image_paths[:20]  # change here

  encoder = Encoder()

  feature_vectors, image_names = encoder.get_all_image_features(image_paths)
  image_names = [format_label(image_name) for image_name in image_names]
  # print(feature_vectors, image_names)

  retrieval_system = ImageRetrievalSystem(feature_vectors)

  thresholds = [1, 5, 10, 20, 50]

  X_train, X_test, y_train, y_test = train_test_split(
      feature_vectors, image_names, test_size=0.1, random_state=RANDOM_STATE)

  mAPs = {}

  for threshold in tqdm(thresholds):
    rs = []
    for i in range(len(X_test)):
      image_feature = torch.unsqueeze(X_test[i],0)

      relevant_indices = retrieval_system.search_image_by_features(
        image_feature, k=threshold)[0]

      rs.append([1 if image_names[j] == y_test[i]
                else 0 for j in (relevant_indices)])

    mAP = mean_average_precision(rs)
    mAPs["mAP@" + str(threshold)] = mAP

  print(mAPs)
