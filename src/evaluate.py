import numpy as np
import os
import torch
from config import DATASET_DIR, RANDOM_STATE
from retrieval_system import ImageRetrievalSystem
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm


def precision_at_k(r, k):
  assert k >= 1
  r = r[:k] != 0
  if r.size != k:
    raise ValueError('Relevance score length < k')
  return np.mean(r)


def recall_at_k(r, k, n_relevant):
  assert k >= 1
  r = r[:k] != 0
  if r.size != k:
    raise ValueError('Relevance score length < k')
  return np.sum(r) / n_relevant


def average_precision(r):
  r = np.asarray(r) != 0
  out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
  if not out:
    return 0.
  return np.mean(out)


def mean_average_precision(rs):
  return np.mean([average_precision(r) for r in rs])


def tranform_query(query):
  return query.lower().replace('_', ' ')


def get_label_map():
  # Replace 'your_file.txt' with the actual path to your file
  file_path = os.path.join(DATASET_DIR, "map_clsloc.txt")

  # Initialize an empty dictionary to store the mapping
  mapping = {}

  # Read the file and populate the dictionary
  with open(file_path, 'r') as file:
    for line in file:
      # Split each line into parts based on spaces
      parts = line.strip().split()

      # Extract the number and corresponding string
      number = int(parts[1])
      string_value = ' '.join(parts[2:])

      # Add the entry to the dictionary
      mapping[number] = string_value
  return mapping


if __name__ == "__main__":

  feature_vectors = torch.load(
      os.path.join(DATASET_DIR, "image_features.pt"))

  label_map = get_label_map()

  with open(os.path.join(DATASET_DIR, "ILSVRC2012_validation_ground_truth.txt"), "r") as f:
    ground_truth_map = [int(line.strip()) for line in f.readlines()]

  with open(os.path.join(DATASET_DIR, "image_names.txt"), "r") as f:
    image_ground_truths = []
    for line in f.readlines():
      image_name = line.strip()
      number_match = re.search(r'ILSVRC2012_val_(\d+)\.JPEG', image_name)
      if number_match:
        image_number = int(number_match.group(1))
        image_ground_truths.append(ground_truth_map[image_number - 1])

  retrieval_system = ImageRetrievalSystem(feature_vectors)

  thresholds = [1, 5, 10, 20, 50]


  X_train, X_test, y_train, y_test = train_test_split(
      feature_vectors, image_ground_truths, test_size=0.1, random_state=RANDOM_STATE)

  mAPs = {}

  for threshold in tqdm(thresholds):
    rs = []
    for i in range(len(X_test)):
      text_query = label_map[y_test[i]]
      relevant_indices = retrieval_system.search_text(
        text_query, k=threshold)[0]
      rs.append([1 if image_ground_truths[j] == y_test[i]
                else 0 for j in (relevant_indices)])

    mAP = mean_average_precision(rs)
    mAPs["mAP@" + str(threshold)] = mAP

  print(mAPs)
