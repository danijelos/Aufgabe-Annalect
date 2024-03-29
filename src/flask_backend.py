from flask import Flask, jsonify, request
import pandas as pd
import json
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Implement Semantic Text Similarity Class
# implement Classifier
# The SentenceBERT model is used which is a modification of BERT that was developed especially for text similarity tasks
# It is much faster than BERT and thus more scalable
class STS():
  # generate the SentenceBERT model
  def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    super().__init__()
    self.model = SentenceTransformer(model_name)

  # function to generate the embeddings through the .encode function which generates an embedding using the CLS token
  def generate_embeddings(self, source_df, target_df):
    source_embeddings = self.model.encode(source_df["label_name"])
    target_embeddings = self.model.encode(target_df["description"])
    return source_embeddings, target_embeddings
  
  # compare each of the target segment with each of the source segments and safe the one with the highest cosine similarity
  # In the context of comparing embeddings, cosine similarity measures the cosine of the angle between the two embedding vectors.
  # If the angle is small (i.e., the vectors point in roughly the same direction), the cosine similarity will be close to 1, indicating high similarity.
  # If the angle is large (i.e., the vectors point in different directions), the cosine similarity will be close to -1 or 0, indicating low similarity.
  def find_most_similar(self, source_df, target_df):
        source_embeddings, target_embeddings = self.generate_embeddings(source_df, target_df)
        similarities = cosine_similarity(target_embeddings, source_embeddings)
        max_similarities_idx = np.argmax(similarities, axis=1)
        max_similarities = np.max(similarities, axis=1)
        return max_similarities_idx, max_similarities


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def post():
    data = request.get_json()
    # Extract test_audiences data from the JSON
    test_audiences = data['test_audiences']
    # Convert test_audiences to DataFrame
    target_df = pd.DataFrame(test_audiences)

    # Read in the source_segment data and generate dataframe
    df = pd.read_csv("~/anaconda3/envs/daniel_dev/git/data/source_segments_angepasst.csv", encoding = "ISO-8859-1", sep=';')
    df.drop(columns=['Unnamed: 5'], inplace=True)
    source_df = df.copy()

    # Instantiation and similarities generation
    sts_instance = STS()
    max_similarities_idx, max_similarities = sts_instance.find_most_similar(source_df, target_df)

    # Store the sentences in a list of dictionaries
    results = []
    # Iterate through target_df and store similarity results in a list of dictionaries
    # use i to iterate through the target_df
    i = 0
    for idx in max_similarities_idx:
        result = {}
        if idx in source_df.index:
            # Search for label_id_long, segment_description, and label_name in the source_df to output which segment has the highest similarity
            label_id = source_df.loc[idx, 'label_id_long']
            label_description = source_df.loc[idx, 'segment_description']
            label_name = source_df.loc[idx, 'label_name']
            result["Target Segment"] = int(target_df.loc[i, 'segment_id'])
            result["Target Description"] = target_df.loc[i, 'description']
            result["Source Segment"] = int(label_id)
            result["Source Description"] = label_description
            result["Source Label Name"] = label_name
            result["Cosine Similarity Score"] = float(max_similarities[i])
            results.append(result)
        else:
            result["Target Segment"] = int(target_df.loc[i, 'segment_id'])
            result["Target Description"] = target_df.loc[i, 'description']
            result["Source Segment"] = "No data found"
            result["Source Description"] = "No data found"
            result["Source Label Name"] = "No data found"
            result["Cosine Similarity Score"] = "No data found"
            results.append(result)
        i = i + 1
    
    # Convert the list of dictionaries to JSON
    json_results = json.dumps(results)
    print(json_results)
    return jsonify({"results": json_results})

if __name__ == "__main__":
    ## Uncomment for flask only (no docker container)
    app.run(port=5000,debug=True)
    ## Comment out for flask only (no docker container)
     #app.run(host="0.0.0.0", port=8080)