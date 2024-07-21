# price_decision_fastapi: Fast API for Amazon Stock prediction endpoint organized by Docker and Azure services (MLOps and Machine learning project)

This is planned to be integrated with [main repository](https://github.com/chawitzoon/dash-app-ml-devops/tree/main) 

## Overview of ReadME
1. Introduction
2. Project overview
3. Project components
4. Future Work
5. Additional note on resource management concept

## Introduction

This project demonstrate the CI/CD using Azure services. The detail are as follows:
1. ### Source Code Management (SCM): 

The source code is managed in this GitHub repository.

2. ### Continuous Integration (CI):

When changes are made to the main branch, GitHub Actions workflow is triggered.
The workflow checks out code (lint, test, etc), builds a Docker image using your Dockerfile.

3. ### Container Registry:

The built Docker image is then automatically pushed to a container registry. In this example, Docker Hub is used.
The container registry stores and manages the Docker images, making them available for deployment.

4. ### Continuous Deployment (CD):

The workflow can then pull the Docker image from the Docker Hub and deploy it to the target environment on Azure App Service (TBD. Currently use the Azure App Service interface for structural development)
By using a container registry, it is ensured that your Docker images are versioned, stored securely, and can be easily accessed for deployment.


## Project overview

<img width="1792" alt="price_decision_fastapi_project_outline" src="image_readme\price_decision_fastapi_project_outline.png">


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
- cli.py: the main cli.py is the endpoint that serves out price predictions.
- utilscli.py: This cli tool is planned for performing model retraining, model validating, etc (in progress)
- note on cli command examples
  - python cli.py --prices "120,121.4,126.9,128.0,127.8,129.1,130.1"
  - (tbd) python utilscli.py retrain --tsize 0.2
  - python utilscli.py predict --prices "120,121.4,126.9,128.0,127.8,129.1,130.1" --host "http://localhost:8080/predict_next_price"

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
- lint-and-test job: for install, lint, test files
- build-and-push-to-dockerhub job: build Docker Image and push to DockerHub
  - Log in to DockerHub
  - Build, Tag, and Push the Image to DockerHub
  - (the keys and info are stored in Github Secrets)
- deploy-to-azure job: pull docker image from DockerHub and deploy in Azure App Service (TBD)

## Snipshot of service used

<!-- ### Github Action
<img width="1792" alt="price_decision_fastapi_project_outline" src="image_readme\github_action.png"> -->

### Docker Hub
<img width="1792" alt="price_decision_fastapi_project_dockerhub" src="image_readme\price_decision_fastapi_dockerhub.png">

### Azure App Service
<img width="1792" alt="price_decision_fastapi_project_azure_app_service" src="image_readme\price_decision_fastapi_project_azure_app_service.png">

<!-- ### FastAPI
the prediction POST request endpoint
<img width="1792" alt="aws_project_outline" src="image_readme\aws_fastapi_example.png"> -->

## Future Work

- do more EDA on the price prediction with more metric and features, including using the knowledge of indicators and standard knowledge in the trading field.
- add the decidion making endpoint to decide if we should buy/sell at how much volumn (based on algorithmic trading and reinforcement learning)
- implement the recent data retrival, checking the data drift, and retaining the model accordingly automatically (or periodically).

## Additional note on resource management concept
Even though this is not implemented in this project. It is important to have a branch strategy when managing and developing programs with multiple stakeholders and environments. Here are the branch strategies I come up with as some examaples. Note that each strategy suits each project based on the neccessity of the branches, environment, and how the workflow is done in each project. By considering these, we can achieve the best practice of resource management, reducing blockers, maximize the development flow speed. 
Since this project is a personal project with more straightforward workflow. It is more efficient to keep things based on its minimal requirement as possible.

### example 1
<img width="1792" alt="branch_manage_ex1" src="image_readme\branch_manage_ex1.png">
This example is a complicated manangement with UAT branch in Production environment. Some of the use cases might be the systems with relationships with multiple systems (both upstream and downstream systems). By doing so, we can UAT test in the Production data and connections while being in our test resource.

### example 2
<img width="1792" alt="branch_manage_ex2" src="image_readme\branch_manage_ex2.png">
This example more straightforward. All test can be done in our Development environment. This should be more conventional and default setting for most projects to simplify the steps and make the workflow more efficient.
