# challenge-api-deployment
ImmoEliza FastApi Deployment Project

## Description

This repository, **challenge-api-deployment**, documents the successful completion of a learning project with a focus on deploying a machine learning model as an API. The project had a duration of 5 days, with a deadline on 09/06/2023.

## Usage

The project involved several essential steps, all successfully completed, to deploy a machine learning model as an API. Here's a summary of the accomplished tasks:

### Step 1: Project Preparation

- Created a dedicated project folder structure, including folders for `app.py`, `preprocessing/`, `model/`, and `predict/`.

### Step 2: Preprocessing Pipeline

- Implemented the `cleaning_data.py` file within the `preprocessing/` folder to preprocess incoming data.
- Developed the `preprocess()` function to handle data preprocessing, including handling NaN values and text data.
- Ensured the function returns an error message if the required information is missing.

### Step 3: Prediction

- Utilized the previously created machine learning model to predict house prices.
- Developed the `prediction.py` file within the `predict/` folder to execute predictions.
- Created the `predict()` function to take preprocessed data as input and return a price as output.

### Step 4: Create Your API

- In the `app.py` file, built an API with the following routes:
  - A route at `/` that accepts GET requests and returns "alive" if the server is running.
  - A route at `/predict` that accepts both POST and GET requests, facilitating data submission and providing data format instructions.

### Step 5: Create a Dockerfile

- Constructed a Dockerfile to package the API for deployment.
- The Docker image includes Ubuntu, Python 3.10, FastAPI, and all necessary project dependencies.
- Launched the `app.py` file with Python within the Docker container.

### Step 6: Deploy Your Docker Image in Render.com

- Deployed the Docker image on Render.com to make the API accessible via the internet.

### Step 7: Document Your API

- Provided clear documentation in the README for web developers, specifying available routes and their methods, expected data formats (mandatory or optional), and potential response formats in both success and error cases.

## Personal Situation

As a solo project, I managed all aspects of the development and deployment process.

---

This readme provides an overview of the successfully completed **challenge-api-deployment** personal project, its objectives, installation instructions, and a summary of the accomplished tasks. For any further details or inquiries, please feel free to reach out.
