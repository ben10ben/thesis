# Multivariate Deep Transfer Learning for Robust Building Electric Load Forecasting
Type: Master's Thesis

Author: Benedikt Rein

1st Examiner: Prof. Dr. Lessmann

2nd Examiner: Prof. Dr. Fabian

![alt text](https://github.com/ben10ben/thesis/blob/master/outputs/results/final_outputs/target_GP2.png)


# Table of Content
Summary
Working with the repo
Dependencies
Setup
Reproducing results
    ARIMA baseline
    TimeGPT baseline
    iTransformer baseline
    iTransformer Transfer Learning
    NHits/Transformer/TSMixer using Darts library
    Process and visualise outputs
Results

# Summary
Keywords: building eletric load forecastng, global time series forecasting, multivariate, deep transfer learning, pre-trained models, foundation models

This repository implements the experiments as descriped in the research paper provided. 

With increased smart load measuring infrastructure and more indivisualised consumption patterns due to decentralised solar produciton and charging of electric vehicles, entity specific forecasting becomes even more crucial for efficient grid opperation. Modelling thouands of covariates and handling of indivisualised statistic models becomes impractical, incentivising the use of channel-depdendent multivariate forecasting to capture channel interactions instead of extensive covariate modelling. New infrastructure or new smart-meters lack entity-specific data, leading us to also include transfer learning for the evaluation.

We evaluate multiple deep learning models, SARIMA and the pre-trained model TimeGPT on the building electric load forecasting task. We benchmark all deep learning models without transfer learning against multiple transfer learning approaches. We evaluate the metrics "Jumpstart" and "Asymptotic Performance". We can show significant improvements on the 2 widely used datasets, especially with TSMixer and iTransformer. The lesser known and irregular dataset shows the intricacy and complexitx of transfer learning as explored in the paper.

The "Building Data Genome Project 2" dataset is used for a case study where we show the potential to increase predictive performance during the first year of measurement by up to 29% or 1.6 % of MAPE, which equals savings of around 35 USD per building.  


# Working with the repo
## Dependencies
The code was written using Python 3.10 on Linux.

## Setup
- Clone this repository

- Run `setup.sh` to create the nessesary folders, download the datasets and create a virtual enviroment with correct dependencies.
If the download does not work, make sure the datasets are available in the referenced data folders

```bash
git clone https://github.com/ben10ben/thesis
cd thesis
bash setup.sh
```

For each experiment, we have one Jupyter Notebook which mostly can be executed individually.

# Calculate ARIMA results
Use the `arima_baselines.ipynb` notebook to fit a model for each dataset seris and predict on the test split without re-fitting.

Use `TimeGPT_baseline.ipynb` to process all datasets to the requested format by TimeGPT, do forecasts using the API and save the results.

Use `iTransformer_baselines.ipynb` to create iTransformer baselines and save the best pre-trained models to be used for transfer learning. Training can be resumed but stability is not guaranteed. To be sure, saved checkpoints can be deleted to restart from scratch. 

Use `iTransformer_full_tl.ipynb`after the baselines are trained and checkpointed. This notebook does not work without the pre-trained models.
This notebook loads and reshapes the pre-trained models according to the target dataset, does fine-tuning and inference on the target dataset and saves the results.Training can be restared from the pre-trained models, they are not checkpointed to retain the state after source training.

Use `darts_with_checkpointing.ipynb` to forecast using the NHits/Transformer/TSMixer models with the darts library. Checkpointing is implemented to use the best model but checkpoints are deleted after inference because of model size and number. Sub-experiments cannot be resumed but once a dataset-model-tl_setup combination is finished the results are saved and skipped when resuming training.  

Use `Process_results.ipynb` to load all outputs and process outputs for final tables and visualisaiton and to calculate the metrics 'jumpstart' and 'asymptotic performance'

# Results
![alt text](https://github.com/ben10ben/thesis/blob/master/outputs/results/final_outputs/startup_strategies_mae.png)

# Project structure

├── README.md
├── requirements.txt                                -- required libraries
├── setup.sh                                        -- setup script to download data, create folders, create venv
├── data                                            -- stores the used datasets
├── outputs                                         -- stores all outputs
    ├──models                                       -- stores pre-trained models for later use or checkpointing
    ├──results                                      -- stores predictions and forecasitng metrics
        ├──final_outputs                            -- stores final output tables and visuals for paper
└── src                                             -- all code used for the research
    ├── arima_baselines.ipynb                       -- ARIMA baseline experiments
    ├── TimeGPT_baseline.ipynb                      -- TimeGPT baseline experiments 
    ├── iTransformer_baselines.ipynb                -- iTransformer baseline and model checkpoints
    ├── iTransformer_full_tl.ipynb                  -- iTransformer transfer learning 
    ├── darts_with_checkpointing.ipynb              -- NHits/Transformer/TSMixer baseline and transfer learning
    ├── process_results.ipynb                       -- merge results, create tables and plots
        └── helpers                                 -- functions and classes used in the project
└── additional                                      -- deprecated experiments not used for research but potentially interesting
    ├── darts_without_checkpointing.ipynb           -- NHits/Transformer/TSMixer baselines and experiments using the last model instead of best
    ├── exploratory                                 -- dataset exploration and visualisation of data pre-processing
    ├── reproduce_electricityresults.ipynb          -- train iTransformer on multiple forecasting horizons and compare normalisation strategies
    └── transfer_learning_split_dataset.ipynb       -- iTransformer transfer learning after splitting a dataset on ids