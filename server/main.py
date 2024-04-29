import argparse
import subprocess
from utils.feature_extractor import extract_features
from constants import BATCH_SIZE
import os
from dotenv import load_dotenv, dotenv_values 


def main():
  parser = argparse.ArgumentParser(description='Image Search System')
  parser.add_argument('--extract', action='store_true', help='Extract features')
  parser.add_argument('--serve', action='store_true', help='Start the web server')
  parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')

  load_dotenv() 
 
  # accessing and printing value
  print(os.getenv("API_KEY"))


  args = parser.parse_args()

  if args.extract:
    # Extract features
    features = extract_features()

  if args.serve:
    # Start the web server
    subprocess.run(["streamlit", "run", "server.py"])

  if args.evaluate:
    # Run evaluate.py
    subprocess.run(["python", "evaluate.py"])

if __name__ == "__main__":
  main()