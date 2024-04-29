import pinecone

class PineconeDB:
  def __init__(self, api_key) -> None:
    pinecone.init(
      api_key = api_key,
      environment="us-east-1" 
    )

  def create_index(self, index_name, dimension=512, metric="cosine"):
    if index_name not in pinecone.list_indexes():
      pinecone.create_index(index_name, dimension=dimension, metric=metric)
      print(f"Index {index_name} is created successfully!")

    else:
      print("Index already exists!")

  def insert_records(self, index_name, vector_data):
    index = pinecone.Index(index_name)
    index.upsert(vectors=vector_data)
    print("Inserted successfully")

  def delete(self, index_name, vector_id):
    index = pinecone.Index(index_name)
    index.delete(ids=vector_id)
    print("Deleted successfully!")
