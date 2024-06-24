# Multivariate Deep Transfer Learning for Robust Building Electric Load Forecasting
Type: Master's Thesis

Author: Benedikt Rein

1st Examiner: Prof. Dr. Lessmann

2nd Examiner: Prof. Dr. Fabian

Example results on the BDGP2 dataset using different training strategies:
![Results on BDGP2 dataset](https://github.com/ben10ben/thesis/blob/master/outputs/results/final_outputs/target_GP2.png)


# Table of Content
- [Summary](#Summary)
- [Working with the repo](#working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
    - [Run experiments](#Run-experiments)
- [Results](#Results)
- [Project Structure](#Project-structure)


# Summary
**Keywords**: building eletric load forecastng, global time series forecasting, multivariate, deep transfer learning, pre-trained models, foundation models

This repository implements the experiments as descriped in the research paper provided. 

With increased smart load measuring infrastructure and more indivisualised consumption patterns due to decentralised solar produciton and charging of electric vehicles, entity specific forecasting becomes even more crucial for efficient grid opperation. Modelling thouands of covariates and handling of indivisualised statistic models becomes impractical, incentivising the use of channel-depdendent multivariate forecasting to capture channel interactions instead of extensive covariate modelling. New infrastructure or new smart-meters lack entity-specific data, leading us to also include transfer learning for the evaluation.

We evaluate multiple deep learning models, SARIMA and the pre-trained model TimeGPT on the building electric load forecasting task. We benchmark all deep learning models without transfer learning against multiple transfer learning approaches. We evaluate the metrics "Jumpstart" and "Asymptotic Performance". We can show significant improvements on the 2 widely used datasets, especially with TSMixer and iTransformer. The lesser known and irregular dataset shows the intricacy and complexitx of transfer learning as explored in the paper.

The "Building Data Genome Project 2" dataset is used for a case study where we show the potential to increase predictive performance during the first year of measurement by up to 29% or 1.6 % of MAPE, which equals savings of around 35 USD per building.  


# Working with the repo

## Dependencies
The code was written using Python 3.10 on Linux. Dependencies are defined in the requirements file. 

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

## Run experiments
Use the `arima_baselines.ipynb` notebook to fit a model for each dataset seris and predict on the test split without re-fitting.

Use `TimeGPT_baseline.ipynb` to process all datasets to the requested format by TimeGPT, do forecasts using the API and save the results.

Use `iTransformer_baselines.ipynb` to create iTransformer baselines and save the best pre-trained models to be used for transfer learning. Training can be resumed but stability is not guaranteed. To be sure, saved checkpoints can be deleted to restart from scratch. 

Use `iTransformer_full_tl.ipynb`after the baselines are trained and checkpointed. This notebook does not work without the pre-trained models.
This notebook loads and reshapes the pre-trained models according to the target dataset, does fine-tuning and inference on the target dataset and saves the results.Training can be restared from the pre-trained models, they are not checkpointed to retain the state after source training.

Use `darts_with_checkpointing.ipynb` to forecast using the NHits/Transformer/TSMixer models with the darts library. Checkpointing is implemented to use the best model but checkpoints are deleted after inference because of model size and number. Sub-experiments cannot be resumed but once a dataset-model-tl_setup combination is finished the results are saved and skipped when resuming training.  

Use `Process_results.ipynb` to load all outputs and process outputs for final tables and visualisaiton and to calculate the metrics 'jumpstart' and 'asymptotic performance'

# Results

Results for the case study using the BDGP2, comparing 2 pre-trained iTransformer models against a non-pre-trained baseline.
![Case study on BDGP2 dataset](https://github.com/ben10ben/thesis/blob/master/outputs/results/final_outputs/startup_strategies_mae.png)

Transfer Learning Metrics on ELD and BDGP2 datasets (Percentage change between MSE)
![Transfer learning between ELD and BDGP2 with iTransformer and TSMixer](https://github.com/ben10ben/thesis/blob/master/outputs/results/final_outputs/tl_table_gp2_eld.png)

# Project structure

├── README.md\
├── requirements.txt &nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- required libraries\
├── setup.sh &nbsp;&ensp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- setup script to download data, create venv\
├── data &ensp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- stores the used datasets\
└── outputs &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- stores all outputs\
&emsp;├── models &nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- stores pre-trained models for later use\
&emsp;└── results &nbsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- stores predictions and forecasitng metrics\
&emsp;└── final_outputs &nbsp;&nbsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- stores tables and visuals for paper\
└── src &nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- all code used for the research\
&emsp;├── arima_baselines.ipynb &nbsp;&nbsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- ARIMA baseline experiments\
&emsp;├── TimeGPT_baseline.ipynb &emsp;&emsp;&emsp;&emsp;&emsp;-- TimeGPT baseline experiments\
&emsp;├── iTransformer_baselines.ipynb &nbsp;&nbsp;&nbsp;&emsp;&emsp;-- iTransformer baseline and checkpointing\
&emsp;├── iTransformer_full_tl.ipynb &nbsp;&nbsp;&emsp;&emsp;&emsp;&emsp;-- iTransformer transfer learning\
&emsp;├── darts_with_checkpointing.ipynb &nbsp;&nbsp;&nbsp;&emsp;-- NHits/Transformer/TSMixer baseline and TL\
&emsp;└── process_results.ipynb &nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- merge results, create tables and plots\
&emsp;└── helpers &nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- functions and classes used in the project\
└── additional &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- deprecated experiments not used for research\
&emsp;├── darts_no_checkpoint.ipynb &emsp;&emsp;&emsp;&emsp;-- NHits/Transformer/TSMixer experiments using last model\
&emsp;├── exploratory &nbsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- dataset exploration and visualisation of data processing\
&emsp;├── reproduce_eld.ipynb &nbsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- iTransformer on multiple horizons and compare normalisation\
&emsp;└── tl_split_dataset.ipynb &nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;-- iTransformer transfer learning after splitting dataset on ids\