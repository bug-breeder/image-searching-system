from constants import INDEX_NAME
import os
from dotenv import load_dotenv, dotenv_values
from db import PineconeDB
import torch


def push_features_to_db():
    load_dotenv()

    # get api key from .env
    api_key = os.getenv("API_KEY")

    pinecone_db = PineconeDB(api_key)
    pinecone_db.create_index(INDEX_NAME)

    # Load PyTorch features vector
    features = torch.load("dataset/image_features.pt")

    data_to_insert = []

    with open("dataset/image_names.txt", "r") as f:
        image_names = f.readlines()

        for idx, feature in enumerate(features):
            img_name = image_names[idx].split(".")[0]  # remove file extensions
            data_to_insert.append({"id": img_name, "values": feature.tolist()})

        pinecone_db.insert_batch(INDEX_NAME, data_to_insert)
        print("Features pushed to Pinecone successfully!")
