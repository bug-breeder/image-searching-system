import os
import torch
from config import DATASET_DIR
import streamlit as st
from PIL import Image
from retrieval_system import ImageRetrievalSystem

def get_image():
    image_file = st.file_uploader("Upload Images",type=["png", "jpg", "jpeg"])
    img = None
    save_dir = None

    if image_file is not None:
        img = Image.open(image_file)
        st.image(img, width=250)
        from datetime import datetime
        now = datetime.now()
        save_dir = os.path.abspath(os.path.join('storage', now.strftime("%d-%m-%Y-%H-%M-%S")+'.jpg'))
        print(save_dir)
        # Saving upload
        img = img.convert('RGB')
        img.save(save_dir)
        st.success("Image uploaded successfully!")
        return save_dir


def main():
  feature_vectors = torch.load(
      os.path.join(DATASET_DIR, "image_features.pt"))

  with open(os.path.join(DATASET_DIR, "image_names.txt"), "r") as f:
    image_names = [line.strip() for line in f.readlines()]

  retrieval_system = ImageRetrievalSystem(feature_vectors)

  st.title('Image Retrieval')

  selection = st.sidebar.radio("Please select the type of query", options= ["Image", "Text"])

  if selection == "Text":
    with st.form(key='text_search_form'):
      query = st.text_input('Enter your query:')
      num_retrieved = st.number_input('Enter the number of similar images to retrieve:', min_value=1, max_value=10, value=6, step=1)
      submit_button = st.form_submit_button('Search')

      if query or submit_button:
          # Extract features from the text query
          indices = retrieval_system.search_text(query, k=num_retrieved)[0]
          similar_images = [os.path.join(DATASET_DIR, 'ILSVRC2012_img_val', image_names[i]) for i in indices]

          # Display the similar images in three columns
          columns = st.columns(3)
          for i, image_path in enumerate(similar_images):
            image = Image.open(image_path)
            columns[i % 3].image(image)
      else:
        st.write('Please enter a query.')

  else:
    with st.form(key='image_search_form'):
      input_img = get_image()
      num_retrieved = st.number_input('Enter the number of similar images to retrieve:', min_value=1, max_value=10, value=6, step=1)
      submit_button = st.form_submit_button('Search')

      if submit_button:
        if input_img:
          # Extract features from the text query
          indices = retrieval_system.search_image(input_img, k=num_retrieved)[0]
          similar_images = [os.path.join(DATASET_DIR, 'ILSVRC2012_img_val', image_names[i]) for i in indices]

          # Display the similar images in three columns
          columns = st.columns(3)
          for i, image_path in enumerate(similar_images):
            image = Image.open(image_path)
            columns[i % 3].image(image)
        else:
          st.write('Please upload an image.')

if __name__ == '__main__':
  main()
