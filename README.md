# price_decision_fastapi: Fast API for Amazon Stock prediction endpoint (MLOps and Machine learning project)
(Obsoleted Project, moved to to utlize Docker Hub and Azure App Service instead TBD) 

This is planned to be integrated with [main repository](https://github.com/chawitzoon/dash-app-ml-devops/tree/main) 

## Overview of ReadME
1. Introduction
2. Project overview
3. Project components
4. Future Work

## Introduction

This project demonstrate the CI/CD using AWS services. The detail are as follows:
1. ### Source Code Management (SCM): 

The source code is managed in this GitHub repository.

2. ### Continuous Integration (CI):

When changes are made to the main branch, GitHub Actions workflow is triggered.
The workflow checks out code (lint, test, etc), builds a Docker image using your Dockerfile.

3. ### Container Registry:

The built Docker image is then automatically pushed to a container registry. In this example, Docker Hub is used.
The container registry stores and manages the Docker images, making them available for deployment.

4. ### Continuous Deployment (CD):

The workflow can then pull the Docker image from the Docker Hub and deploy it to the target environment on Azure App Service
By using a container registry, it is ensured that your Docker images are versioned, stored securely, and can be easily accessed for deployment.


## Project overview

<img width="1792" alt="project_outline" src="tbd">


Technologies Used:
1. Github and Github Actions
2. Docker and Docker Hub
3. Azure App Service
4. FastAPI
5. Torch and Scikit-learn

## Project components
### Makefile: for install, test, format, lint. Used in development and CI
### requirements.txt: all dependencies for the program
### CLI Tools: There are two CLI tools. 
- cli.py: the main cli.py is the endpoint that serves out price predictions. TBD
- utilscli.py: This cli tool is planned for performing model retraining, model validating, etc TBD
### app.py: The FastAPI ML Microservice.
### mlib library
- mlib.py: includes functions that are called by app.py or other cli tools
- mlib_util.py: includes helper functions, model (LSTM) class, Agent class (for convinience in training, validating, visualization and test prediction)
- __init__.py: treat the directory as a callabale library
- mlib_model folder: including
  - lstm_model_hyper_param.json: hyper parameter used in model class and Agent class
  - lstm_model.pth: trained model weight, saved by torch save function
  - X_scaler.pkl: Scaler for inputs obtained during training
  - y_scaler.pkl: Scaler for output obtained during training
### test_app.py: used pytest for unit testing in CI
### test_mlib.py: used for experiment and EDA/model development phase
### Dockerfile: build Docker image and set the application to run by gunicorn
### .github/workflows/main.yml
- build job: for install, lint, test, format
- build-and-push job: build Docker Image and push to ECR
  - Configure AWS credentials
  - Login to Amazon ECR
  - Build, Tag, and Push the Image to Amazon ECR
  - (the keys and info are stored in Github Secrets)

## Snipshot of service used

<!-- ### Github Action
<img width="1792" alt="aws_project_outline" src="image_readme\github_action.png">

### Amazon ECR
<img width="1792" alt="aws_project_outline" src="image_readme\aws_ecr_screenshot.png">

### Azure App Service
<img width="1792" alt="aws_project_outline" src="image_readme\aws_app_runner_screenshot.png">

### FastAPI
note: the prediction endpoint is tested as GET request for simplicity for now.
<img width="1792" alt="aws_project_outline" src="image_readme\aws_fastapi_example.png"> -->

## Future Work

## note cli command

python cli.py --prices "1.1,2.2,3.3,4.4,5.5,6.6,7.7"

python utilscli.py retrain --tsize 0.2

python utilscli.py predict --prices "1.1,2.2,3.3,4.4,5.5,6.6,7.7" --host "http://localhost:8080/predict_next_price"
