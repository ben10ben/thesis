!/bin/bash
mkdir data/ELD
mkdir data/Bavaria
mkdir data/GP2

#download electricity dataset
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip" -O temp.zip
unzip temp.zip -d data/ELD
rm temp.zip

# building genome project dataset
wget https://github.com/buds-lab/building-data-genome-project-2/raw/master/data/meters/cleaned/electricity_cleaned.csv?download= -O data/GP2/genome_project_dataset.csv


# south germany datast
wget https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv -O data/Bavaria/south_germany.csv


# use this for pip
python3.10 -m venv myenv
source myenv/bin/activate
#pip install --upgrade pip
pip3 install -r requirements.txt

# use this if you prefer conda
#conda update -n base -c defaults conda
#conda create -n yourenvname python=3.10 anaconda
#source activate ma_env
#conda install pip
#pip3 install -r requirements.txt
