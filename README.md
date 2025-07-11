# Forecasting Air Quality with Linear Autoregressive Models

This repository contains the implementation for a final project in Applied Machine Learning, focusing on forecasting air quality (specifically NO2 levels) using linear autoregressive models. The project explores time series regression, autocorrelation, and the impact of regularization.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To set up the development environment, first create and activate a Python virtual environment:

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source ./.venv/bin/activate
```
To install the required packages:

```setup
pip install -r requirements.txt
```

## Data

This project utilizes the **UCI Air Quality Dataset**.

**Download and Setup:**
1. Navigate to the [UCI ML Repository - Air Quality page](https://archive.ics.uci.edu/ml/datasets/Air+Quality).
2. Download the `AirQualityUCI.zip` file.
3. Extract the contents of the zip file.
4. Place the `AirQualityUCI.csv` file into the `data/raw/` directory within the root of this repository.

*Note: The `AirQualityUCI.csv` file is included in this repository for convenience. However, for larger datasets, a download instruction would be the primary method.*

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

>📋  Pick a licence and describe how to contribute to your code repository. 
