!/bin/bash
mkdir data
mkdir data/electricity
mkdir data/eu_electricity
mkdir data/south_germany_electricity
mkdir outputs
mkdir outputs/logs
mkdir outputs/tuning_logs
mkdir outputs/models
mkdir outputs/models/arima
mkdir outputs/models/revin
mkdir outputs/models/stationary
mkdir outputs/models/base




wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip" -O temp.zip
unzip temp.zip -d data/electricity
rm temp.zip

wget https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv -O data/eu_electricity/eu_electricity.csv


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
