# Flipkart Review Sentiment Analysis MLOps Project

This project focuses on sentiment analysis of Flipkart reviews using machine learning operations (MLOps) techniques. Sentiment analysis is a natural language processing task aimed at determining the sentiment expressed in a piece of text, in this case, Flipkart product reviews.

The main objective of this project is to develop a robust pipeline for sentiment analysis that incorporates various MLOps practices such as version control, continuous integration/continuous deployment (CI/CD), experiment tracking, and model deployment.

In this README, we will walk you through the folder structure, pipeline stages, the model used, how to run the code, analysis using MLflow and DagsHub, CI/CD deployment using AWS EC2 and GitHub Actions, evaluation metrics, dependencies, conclusion, future work, and acknowledgements.

## Folder Structure

- **src/reviewAnalysis**: This directory contains the main source code for the project.
  - **components**: This subdirectory holds reusable components used in the pipeline, such as data ingestion, validation, transformation, model trainer, and model evaluation.
  - **utils**: Here, utility functions and helpers are stored, facilitating various tasks throughout the project.
  - **logging**: Contains configurations and setup for logging functionalities.
  - **config**: Holds project configuration files, including the main configuration file `configuration.py`.
  - **pipeline**: This directory houses the pipeline definition and stages, including data ingestion, validation, transformation, model trainer, and model evaluation.
  - **entity**: Contains entity classes used in the project, such as `config_entity.py`.
  - **constants**: Holds constant values used across the project.

- **config**: This directory contains project-wide configuration files, including `config.yaml` for general configurations and `params.yaml` for parameter configurations.

- **app.py** and **main.py**: These files serve as entry points for running the application and executing the main functionality.

- **Dockerfile**: This file specifies the instructions to build a Docker image for the project, ensuring consistency and portability of the environment.

- **requirements.txt**: Lists all the Python dependencies required to run the project. Useful for setting up the environment.

- **setup.py**: This file defines metadata about the project and its dependencies. Useful for packaging and distribution.

- **research/trials.ipynb**: This directory contains Jupyter Notebook files used for research, experiments, and trials during the development phase.

## Pipeline Stages

### Data Ingestion
Data is acquired from Kaggle (provide link to the dataset) and stored on Google Drive for easy access. To retrieve the data, we use the `gdown` library to download it from Google Drive and store it in the local environment. Subsequently, the downloaded data is saved in the artifacts folder within the data ingestion directory. We then use the `zipfile` module to unzip the downloaded file, ensuring that the data is readily available for further processing.

### Data Validation
In this stage, we perform thorough validation checks on the acquired data to ensure its integrity and quality. All necessary files are examined for completeness, consistency, and adherence to predefined standards. The validation process involves verifying file formats, data types, and structural integrity. The results of the validation checks are recorded in a `status.txt` file within the artifacts folder of the data validation directory. This allows for easy tracking and monitoring of the data validation process.

### Data Transformation

Data transformation is a crucial stage where raw text data undergoes preprocessing to make it suitable for model training. This involves several NLP techniques applied sequentially:

#### Tokenization
Tokenization is the process of breaking down the text into individual tokens or words. In this stage, we use tokenization to split the raw text data into its constituent units, enabling further analysis at the word level.

#### Stopword Removal
Stopwords are common words that do not carry significant meaning and often occur frequently in text data (e.g., "the", "is", "and"). Removing stopwords helps reduce noise in the data and improves the efficiency of subsequent analysis. We employ a predefined list of stopwords to filter out such words from the tokenized text.

#### Stemming
Stemming is the process of reducing words to their root form by removing affixes (e.g., suffixes, prefixes). This normalization technique helps consolidate words with similar meanings and reduces the dimensionality of the feature space. We utilize stemming to convert words to their base or root form, enhancing the consistency of the text data.

#### Lemmatization
Lemmatization is similar to stemming but ensures that the resulting word is valid by mapping it to its dictionary form (i.e., lemma). Unlike stemming, which may result in non-words, lemmatization preserves the semantic meaning of words. We employ lemmatization to transform words into their canonical form, facilitating accurate interpretation and analysis.

#### Vectorization
Vectorization is the process of converting text data into numerical vectors, which can be understood by machine learning algorithms. One common approach is to represent each word as a high-dimensional vector in a vector space model. This transformation allows the model to process textual information as numerical features. We used TF-IDF (Term Frequency-Inverse Document Frequency) to convert the preprocessed text data into vector representations.

#### Embedding
In particular, we employ Word Embeddings, a popular technique for representing words as dense vectors in a continuous vector space. Word Embeddings capture semantic relationships between words and preserve contextual information, making them well-suited for tasks such as sentiment analysis. We embed each word in the text data into a fixed-size vector space, where words with similar meanings are closer together in the vector space. This process transforms the text data into a numerical format that can be fed into the neural network model for training.



### Model Trainer
The model training stage involves preparing the data for model training and defining the neural network architecture. We split the preprocessed data into input features (`X_train`) and target labels (`y_train`). Then, we define a Sequential model using Keras, a high-level neural networks API. The model architecture includes an Embedding layer to convert words into dense vectors, a SpatialDropout1D layer to prevent overfitting, an LSTM (Long Short-Term Memory) layer for sequence modeling, and a Dense layer with softmax activation for multi-class classification. After training the model, we save the tokenizer used for text tokenization and the trained model weights as `tokenizer.pickle` and `model.h5`, respectively, in the artifacts folder within the model trainer directory.

### Model Evaluation
Model evaluation is performed to assess the performance of the trained model on unseen data. This involves:
- Splitting the preprocessed data into training and validation sets to evaluate the model's generalization capability.
- Saving the model scores and evaluation metrics, such as accuracy, precision, recall, and F1-score, for further analysis.
- Utilizing MLflow, an open-source platform for managing the end-to-end machine learning lifecycle, for experiment tracking and monitoring. MLflow enables us to log parameters, metrics, and artifacts, facilitating reproducibility and collaboration.

The workflow of the pipeline, from data ingestion to model evaluation, is visualized in the provided image, showcasing the sequential flow of operations.

![Sentiment Analysis Dashboard](static/MlPipeline-flipkartReviewAnalysis.png)


## How to Run the Code

To run the code, follow these steps:

### Clone the repository:
```bash
git clone https://github.com/subhayudutta/Sentiment-Analysis-of-Flipkart-Reviews.git
```

### Setting up the Environment

Activate the Conda environment named `review` using the following command:
   ```bash
   conda activate review
   ```

### Running the Pipeline
To execute all pipeline stages, you have two options:

1. Run main.py using Python:
    ```bash
    python main.py
    ```
This command will execute all pipeline stages sequentially.

2. Alternatively, you can use DVC (Data Version Control) to run the pipeline:
    ```bash
    dvc init  # Initializes DVC in your project directory.
    dvc repro  # Reproduces the pipeline defined in the DVC file to ensure data lineage and reproducibility.
    dvc dag  # Visualizes the pipeline as a directed acyclic graph (DAG), showing dependencies between stages.
    ```
This command will reproduce the pipeline using DVC, ensuring data lineage and reproducibility.

### Running the Flask App
To run the Flask web application, execute the following command:
```bash
python app.py
```
This command will start the Flask server with the specified host (0.0.0.0) and port (8080). This allows the Flask app to be accessible from any network interface on the specified port.

### Predicting Sentiment
Once the Flask app is running, you can access the following endpoints:
```
/predict: Use this endpoint to predict sentiment for a given text input. You can either use cURL commands or visit the endpoint through your web browser.
```

### Training the Model
To train the sentiment analysis model, use the following endpoint:
```
/train: Use this endpoint to trigger model training. You can either use cURL commands or visit the endpoint through your web browser.
```

### Testing Endpoints with Postman
You can use Postman, a popular API testing tool, to interact with the Flask app endpoints:

1. Open Postman and create a new request.
2. Set the request type to POST.
3. Enter the URL of the Flask app endpoint you want to test (e.g., http://localhost:5000/predict or http://localhost:5000/train).
4. If necessary, add any required headers or body parameters.
5. Click the "Send" button to make the request and view the response.

You can use Postman to test both the /predict and /train endpoints for predicting sentiment and training the model, respectively.

This section provides comprehensive instructions for running the code, including setting up the environment, running the pipeline, starting the Flask app, and interacting with the sentiment analysis model. Let me know if you need further assistance!

## Analysis using MLflow and DagsHub

To integrate your code with DagsHub for experiment tracking using MLflow, follow these steps:

1. Connect the code of your GitHub repository to DagsHub. Ensure you have the following environment variables set up:
   ```bash
   MLFLOW_TRACKING_URI=https://dagshub.com/username/Sentiment-Analysis-of-Flipkart.mlflow
   MLFLOW_TRACKING_USERNAME=<your_username>
   MLFLOW_TRACKING_PASSWORD=<your_token>
    ```

2. You can set these variables in your command prompt (cmd) using the following commands:
    ```bash
    set MLFLOW_TRACKING_URI=https://dagshub.com/username/Sentiment-Analysis-of-Flipkart.mlflow
    set MLFLOW_TRACKING_USERNAME=<your_username>
    set MLFLOW_TRACKING_PASSWORD=<your_token>
    ```
For Git Bash, use the export command instead of set.

3. Update the MLflow tracking URI in your code to point to your DagsHub repository. You can do this in the src/reviewAnalysis/config/configuration.py file, specifically in the get_evaluation_config function where the mlflow_uri is defined.
![Sentiment Analysis Dashboard](./static/Screenshot_3.jpg)

4. After running the pipeline using DVC (Data Version Control) with dvc repro, you can check the MLflow experiments through the DagsHub URL provided. You'll be able to view and analyze the experiment runs, including metrics, parameters, and artifacts.

5. Play around with the MLflow experiments in DagsHub to gain insights into the performance of your models and track the progress of your project.

This section provides detailed instructions for integrating your code with DagsHub for experiment tracking using MLflow. Let me know if you need further assistance!


## CI/CD Deployment using AWS EC2 and GitHub Actions

To deploy your application using AWS EC2 and GitHub Actions, follow these steps:

### Setting up AWS

1. Log in to the AWS console.

2. Create an IAM user for deployment with specific access:
   - Grant EC2 access for launching virtual machines.
   - Grant access to ECR (Elastic Container Registry) to save your Docker images in AWS.

### Description of Deployment

1. Build the Docker image of your source code.

2. Push your Docker image to ECR.

3. Launch your EC2 instance.

4. Pull your Docker image from ECR into the EC2 instance.

5. Launch your Docker image in the EC2 instance.

### Required IAM Policies

Ensure the IAM user has the following policies attached:
   - AmazonEC2ContainerRegistryFullAccess
   - AmazonEC2FullAccess

### Setup Steps

1. Create an ECR repository to store/save the Docker image. 

2. Create an EC2 instance (Ubuntu).

3. Install Docker in the EC2 instance (optional):
   ```bash
   sudo apt-get update -y
   sudo apt-get upgrade
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
    ```

### Configuring EC2 as a Self-hosted Runner

1. Navigate to your GitHub repository.
2. Go to **Settings** > **Actions** > **Runners**.
3. Click on **New self-hosted runner**.
4. Choose the operating system for your EC2 instance.
5. Follow the instructions provided to download and configure the runner on your EC2 instance.

### Setting up GitHub Secrets

To securely store sensitive information required for deployment, such as AWS credentials and repository URIs, you can use GitHub Secrets. Follow these steps to set up the necessary secrets:

1. Navigate to your GitHub repository.
2. Go to **Settings** > **Secrets** > **Actions**.
3. Click on **New repository secret**.
4. Add the following secrets:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key ID.
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key.
   - `AWS_REGION`: The AWS region where your resources are located (e.g., `ap-south-1`).
   - `AWS_ECR_LOGIN_URI`: The URI for logging into your ECR (e.g., `11111.dkr.ecr.ap-south-1.amazonaws.com`).
   - `ECR_REPOSITORY_NAME`: The name of your ECR repository (e.g., `review-analysis`).

These steps enable automated deployment of your application to AWS EC2 using GitHub Actions.

## Dependencies

This project relies on the following external libraries and packages:

- numpy: Library for numerical computing in Python.
- pandas: Data manipulation and analysis library.
- tensorflow: Open-source machine learning framework.
- matplotlib: Plotting library for creating visualizations.
- seaborn: Statistical data visualization library based on matplotlib.
- nltk: Natural Language Toolkit for text processing.
- regex: Regular expression operations library.
- ipykernel: IPython kernel for Jupyter notebooks.
- mlflow: Open-source platform for managing the end-to-end machine learning lifecycle.
- Flask: Lightweight WSGI web application framework.
- Jinja2: Template engine for Python.
- PyYAML: YAML parser and emitter for Python.
- python-box: Simple Python library for creating dictionaries with attribute-style access.
- ensure: Library for validating conditions in Python.
- dvc: Data Version Control for managing ML models and data pipelines.
- gdown: Command-line tool for downloading large files from Google Drive.

### Installation

To install the dependencies, first create a Python environment (optional but recommended) and then use pip to install the required packages:

1. **Create Python Environment**: Navigate to your project directory and create a Python environment using virtualenv or conda (if not already created):
   
   ```bash
   # Using virtualenv
   python3 -m venv myenv
   source myenv/bin/activate

   # Using conda
   conda create --name myenv python=3.8
   conda activate myenv
    ```

2.  **Install Required Packages**:Use pip to install the dependencies listed in the requirements.txt file:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**: Once all dependencies are installed, you can run the Flask application:
    ```bash
    python app.py
    ```

This section provides users with clear instructions on how to install the dependencies required for the project to run successfully. Adjust the dependencies and installation instructions according to the specific requirements of your project. Let me know if you need further assistance!

## Screenshot of the Live App

Here is a screenshot of the live sentiment analysis application:

![Screenshot](./static/Screenshot_2.jpg)

## Conclusion

In conclusion, this project showcases the implementation of sentiment analysis on Flipkart reviews using modern machine learning techniques and DevOps practices. By leveraging libraries such as TensorFlow, NLTK, and Flask, I have developed a robust sentiment analysis model and deployed it using AWS EC2 and GitHub Actions.

Throughout the project, I have emphasized the importance of reproducibility and scalability. I have utilized DVC for versioning our data and models, while MLflow has enabled us to track experiments and monitor model performance effectively.

As I continue to evolve this project, I aim to further enhance the model's accuracy and scalability, explore additional deployment options, and integrate more advanced features for better analysis and insights.

I welcome contributions from the community to help us improve and expand the capabilities of this project. Whether it's through bug fixes, feature enhancements, or new ideas, together we can make this project even more impactful in the field of sentiment analysis and beyond.

## License and Acknowledgements

This project is licensed under the [GNU License](LICENSE.md). Feel free to use, modify, and distribute the code for your own purposes.

I would like to acknowledge the contributions of the open-source community, without which this project would not have been possible. Special thanks to the creators and maintainers of libraries, frameworks, and tools that have been instrumental in the development and deployment of this project.

## Contact Me

For any inquiries, feedback, or collaboration opportunities, please feel free to reach out to me via email at duttasuvo90@gmail.com.