# Aufgabe Annalect

## Description

This repository contains a Flask-based Python API utilizing Docker containers for deployment. The main functionality involves semantic text similarity comparison between target segment descriptions and source segment label names. The project utilizes SentenceBERT embeddings for text representation and cosine similarity for comparison.

## Setup
In the src-Folder you find the code and in the data-Folder the data.
To work with the Container the following files are needed: flask_backend.py, Dockerfile, source_segments_angepasst.csv, and requirements.txt.
To test the program use the: flask_call.ipynb to make a test request, which uses the test_audiences.json data

## To use this project, follow these steps:

1. Adjust the `source_segments.csv` file by removing the '|' and ',' characters to ensure smooth loading of the CSV file.
2. Clone the repository to your local machine
3. Navigate to the project directory: cd your_repository
4. Install the necessary dependencies: pip install -r requirements.txt
5. To test the functionality:
- open a terminal, navigate to the project directory and type
-> "pip freeze > requirements.txt"
-> "docker build --tag <your_name> ."
-> "docker run -p 8080:8080 <your_name>:latest
-> Adjust the "load JSON" in the flask_call.ipynb with your own repository and data to test the API
-> Test the program with the flask_call.ipynb file

## Project Incremental Steps
1. Flask App Setup:
- Created a Flask app (flask_backend.py) with a simple x^2 method to test Flask functionality.
- Developed a Jupyter notebook (flask_call.ipynb) to make a POST request and verify the correct result.
2. Docker Containerization:
- Dockerized the simple Flask app for easier deployment and scalability.
3. Semantic Text Similarity Model:
- Developed a semantic text similarity model using SentenceBERT embeddings in text_similarity_test.ipynb.
- Generated embeddings for target segment descriptions and source segment label_names.
- Compared target segment embedding with every source segment embedding using cosine similarity and output the segment with the highest similarity.
4. Integration with Flask Backend:
- Integrated the semantic text similarity logic from text_similarity_test.ipynb into flask_backend.py.
- Chose 0.3 similarity score as a threshold, because even with 0.328 score the findings were okay:
{"Target Segment": 7, "Target Description": "HDGQ Branche/Industrie: automobil/fahrzeugbau/maschinenbau extended", "Source Segment": 20403070000, "Source Description": "interest in business and finance topics", "Source Label Name": "Interest  Business and Finance  Industries  Civil Engineering Industry", "Cosine Similarity Score": 0.32770323753356934},
- Loaded source_segment.csv in the backend (flask_backend.py).
- Provided JSON test files with the request for comparison in the flask_call.ipynb.
5. Containerization of New Logic:
- Containerized the updated logic.
- Copied source_segmentation.csv into the container for accessibility.
