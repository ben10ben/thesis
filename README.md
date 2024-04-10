TODO

automate model saving with sensible names

check if bavaria is loaded correctly and somewhat similar to electricity

visualize eu elec















# Title
Type: Master's Thesis

Author: Benedikt Rein

1st Examiner: Professor Dr. Lessmann

2nd Examiner: xxx

[Insert here a figure explaining your approach or main results]


# Table of Content
Summary
Working with the repo
Dependencies
Setup
Reproducing results
Training code
Evaluation code
Pretrained models
Results
Project structure

# Summary
(Short summary of motivation, contributions and results)

Keywords: xxx (give at least 5 keywords / phrases).

Full text: [include a link that points to the full text of your thesis] Remark: a thesis is about research. We believe in the open science paradigm. Research results should be available to the public. Therefore, we expect dissertations to be shared publicly. Preferably, you publish your thesis via the edoc-server of the Humboldt-Universität zu Berlin. However, other sharing options, which ensure permanent availability, are also possible.
Exceptions from the default to share the full text of a thesis require the approval of the thesis supervisor.

# Working with the repo
## Dependencies
Which Python version is required?

Does a repository have information on dependencies or instructions on how to set up the environment?

## Setup
[This is an example]

Clone this repository

Create an virtual environment and activate it

python -m venv thesis-env
source thesis-env/bin/activate
Install requirements
pip install --upgrade pip
pip install -r requirements.txt
Reproducing results
Describe steps how to reproduce your results.

Here are some examples:

Paperswithcode
ML Reproducibility Checklist
Simple & clear Example from Paperswithcode (!)
Example TensorFlow
Training code
Does a repository contain a way to train/fit the model(s) described in the paper?

## Evaluation code
Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

## Pretrained models
Does a repository provide free access to pretrained model weights?

# Results
Does a repository contain a table/plot of main results and a script to reproduce those results?

# Project structure
(Here is an example from SMART_HOME_N_ENERGY, Appliance Level Load Prediction dissertation)

├── README.md
├── requirements.txt                                -- required libraries
├── data                                            -- stores csv file 
├── plots                                           -- stores image files
└── src
    ├── prepare_source_data.ipynb                   -- preprocesses data
    ├── data_preparation.ipynb                      -- preparing datasets
    ├── model_tuning.ipynb                          -- tuning functions
    └── run_experiment.ipynb                        -- run experiments 
    └── plots                                       -- plotting functions  
