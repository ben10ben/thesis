!/bin/bash
mkdir data
mkdir data/electricity
#mkdir data/eu_electricity
mkdir data/south_germany_electricity
mkdir data/genome_project

mkdir outputs
mkdir outputs/logs
#mkdir outputs/tuning_logs
mkdir outputs/models
mkdir outputs/models/revin
mkdir outputs/models/stationary
mkdir outputs/models/base

mkdir results
mkdir results/arima
mkdir results/timegpt
mkdir results/base
mkdir results/revin
mkdir results/stationary
mkdir results/darts
mkdir results/iTransformer_baseline
mkdir results/iTransformer_split_dataset_tl



#download electricity dataset
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip" -O temp.zip
unzip temp.zip -d data/electricity
rm temp.zip

#eu grid electricity, not used for research
#wget https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv -O data/eu_electricity/eu_electricity.csv


# building genome project dataset
wget https://github.com/buds-lab/building-data-genome-project-2/raw/master/data/meters/cleaned/electricity_cleaned.csv?download= -O data/genome_project/genome_project_dataset.csv


# south germany datast
wget https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv -O data/south_germany_electricity/south_germany.csv


# use this for pip
python3.10 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip3 install -r requirements.txt

# use this if you prefer conda
#conda update -n base -c defaults conda
#conda create -n yourenvname python=3.10 anaconda
#source activate ma_env
#conda install pip
#pip3 install -r requirements.txt
