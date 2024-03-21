from config import *
import pandas as pd
from utils import helpers
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from datetime import datetime



class SlidingWindowTimeSeriesDataset(Dataset):
	"""
	handles all dataset creation from (timeseries x id) tensors
	can handle multiple targets as used in iTransformer
	"""
	def __init__(self, data, window_size, pred_lengths):
		self.data = data
		self.window_size = window_size
		self.pred_lengths = pred_lengths


		if type(pred_lengths) == int:
			self.max_pred_lengths = pred_lengths
		else:
			self.max_pred_lengths = max(pred_lengths)

	def __len__(self):
		return self.data.size(0) - self.window_size - self.max_pred_lengths

	def __getitem__(self, index):	
		# Check if the index is within bounds
		if index < 0 or (index + self.max_pred_lengths + self.window_size) > self.data.size(0):
			raise StopIteration("Index out of bounds")

 	   # Calculate which id and time step to use based on the index
		window = self.data[index : index + self.window_size]


		if type(self.pred_lengths) == int:
			targets = self.data[index + self.window_size : index + self.window_size + self.pred_lengths]
		else:
			targets = tuple(
				self.data[index + self.window_size : index + self.window_size + pred_length]
				for pred_length in self.pred_lengths
				)

		return window, targets



def df_to_tensor(df, standardize: bool):
	"""
	transforms pandas df to tensor
	-no need for specific order
	-must contain columns:
		-id
		-date
		-target
	-returns tensor in shape (timeseries x id)
	"""
	df.sort_values(by=['id', 'date'], inplace=True)
	# check what noral length is
	max_len = df.groupby("id").count()["target"].max()

	# Initialize a list to store tensors
	tensor_list = []

	# Unique IDs
	unique_ids = df['id'].unique()

	# For each 'id', create a tensor and add it to the list
	for unique_id in unique_ids:
		# Filter rows by 'id'
		df_id = df[df['id'] == unique_id]
		# Select relevant columns and convert to tensor
		tensor = torch.tensor(df_id['target'].values, dtype=torch.float32)
		# Add tensor to list if it is not too short
		if len(tensor) == max_len:
			tensor_list.append(tensor)

	# Concatenate all tensors along a new dimension, transpose
	result_tensor = torch.stack(tensor_list)
	result_tensor = result_tensor.transpose(0,1)
	

	# TODO standardize everything with train values!
	standardize_dict = None
	if standardize == True:
		result_tensor, standardize_dict = helpers.custom_standardizer(result_tensor)

	return result_tensor, standardize_dict


def format_electricity():
	"""
	-load from .txt file, resample from 15min to 1h intervalls
	-fill NANs inbetween and drop outside of range
	-remove time
	-save csv for faster loading
	-returns dict: 
		-train, val test
	"""
	
	try:
		dataset_dict = {
			"train" : pd.read_csv(CONFIG_DATA["electricity"] / "electricity_train.csv", index_col=False),
			"validation" : pd.read_csv(CONFIG_DATA["electricity"] / "electricity_val.csv", index_col=False),
			"test" : pd.read_csv(CONFIG_DATA["electricity"] / "electricity_test.csv", index_col=False)
			}
	
	except FileNotFoundError:
		df = pd.read_csv(CONFIG_DATA["electricity"] / "LD2011_2014.txt", index_col=0, sep=';', decimal=',')
		df.index = pd.to_datetime(df.index)
		df.sort_index(inplace=True)

		# Used to determine the start and end dates of a series
		output = df.resample('1h').mean().replace(0., np.nan)
		earliest_time = output.index.min()

		df_list = []
		for label in output:
			srs = output[label]
			start_date = min(srs.fillna(method='ffill').dropna().index)
			end_date = max(srs.fillna(method='bfill').dropna().index)
			active_range = (srs.index >= start_date) & (srs.index <= end_date)
			srs = srs[active_range].fillna(0.)
			tmp = pd.DataFrame({'target': srs})
			date = tmp.index
			tmp['date'] = date
			tmp['id'] = label
			df_list.append(tmp)
		output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)
		output = output[(output['date'] >= "2014-01-01")]
		
		# calc max length and only keep ids that have the same
		# TODO explain in paper why drop instead of backfill
		# rather less timeseries but real world data than synthetic data
		# refference paper with dropped
		id_counts = output.groupby('id')['date'].count()
		max_timestep_count = id_counts.max()
		ids_with_max_count = id_counts[id_counts == max_timestep_count].index
		
		# Filter the original DataFrame to keep only the desired ids
		output = output[output['id'].isin(ids_with_max_count)]

		# cutoffs taken from previous papers
		training_start_date 	= "2014-01-01"
		#validation_start_date 	= "2014-08-23" #TODO select bigger horizon for 720 preds: 720h + 96h = 34 days
		validation_start_date   = "2014-07-28" # new validation horizon
		test_start_date			= "2014-09-01"

		dataset_dict = {}
		dataset_dict["train"] = output[(output['date'] >= training_start_date) & (output["date"] <= validation_start_date)].copy() 
		dataset_dict["validation"] = output[(output['date'] > validation_start_date) & (output["date"] < test_start_date)].copy() 
		dataset_dict["test"] = output[(output['date'] >= test_start_date)].copy() 

		# Cast the string to a datetime variable
		training_start_date = datetime.strptime("2014-01-01", "%Y-%m-%d")
		validation_start_date = datetime.strptime( "2014-07-29", "%Y-%m-%d")
		test_start_date = datetime.strptime("2014-09-01", "%Y-%m-%d")

		train_len = validation_start_date - training_start_date
		val_len = test_start_date - validation_start_date

		print(f"Length train set: {train_len}")
		print(f"Length validation set: {val_len}")

        # reset index and save for faster loading
		print("Saving train, validation and test df for faster loading")
		for key, value in dataset_dict.items():
			dataset_dict[key] = value.reset_index(drop=True)
			value.to_csv(CONFIG_DATA["electricity"] / f"{key}.csv", sep=',', index=False)

	return dataset_dict

def load_electricity():
	try:
        # Specify the file path where you want to save the dictionary
		file_path = CONFIG_DATA["electricity"] / "electricity_dict.pkl"

        # Load the dictionary from the file using pickle.load
		with open(file_path, 'rb') as file:
			data_dict = pickle.load(file)


	except FileNotFoundError:
		data_dict = format_electricity()
		standardize_values = dict()

		for key, value in data_dict.items():
			data_dict[key], standardize_values[key] = df_to_tensor(value, standardize=True) #TODO standardize on train metrics
		# Save the dictionary to the file using pickle.dump
		with open(file_path, 'wb') as file:
			pickle.dump(data_dict, file)
			pickle.dump(standardize_values, file)

	return data_dict


def convert_data(data_dict, window_size, pred_length):

	train_window = SlidingWindowTimeSeriesDataset(data_dict["train"], window_size, pred_length)
	val_window = SlidingWindowTimeSeriesDataset(data_dict["validation"], window_size, pred_length)
	test_window = SlidingWindowTimeSeriesDataset(data_dict["test"], window_size, pred_length)


	dataloader_train = DataLoader(train_window, batch_size=32, shuffle=True)
	dataloader_validation = DataLoader(val_window, batch_size=32, shuffle=False)
	dataloader_test = DataLoader(test_window, batch_size=32, shuffle=False)

	train_features, train_labels = next(iter(dataloader_train))
	print(f"Feature batch shape: {train_features.size()}")

	return dataloader_train, dataloader_validation, dataloader_test

def load_eu_electricity():
	"""
	processes europe electricity dataset
	-load
	-handle nan
	-return dict: train, val test
	"""
	df = pd.read_csv(CONFIG_DATA["eu_electricity"] / "eu_electricity.csv")
		
	df = df.drop('cet_cest_timestamp', axis=1)

	# rename to fit DataSetClass
	df.rename(columns={'utc_timestamp': 'date'}, inplace=True)
	
	# drop forecasts because its synthetic data
	columns_to_delete = ['forecast']
	df = df.drop([col for col in df.columns if any(partial_name in col for partial_name in columns_to_delete)], axis=1)

	# drop columns with more than 40% missing values
	threshold = 0.3 * len(df)
	columns_to_drop = df.columns[df.isna().sum() > threshold]
	df = df.drop(columns=columns_to_drop)

	# drop rows with more than 10% NANs
	threshold = 0.05 * len(df.columns)
	df = df[df.isna().sum(axis=1) <= threshold]

	# forward fill if previous values are present
	df = df.fillna(method='ffill')

	# values at the start are set to zero, no more NANs
	df = df.fillna(0)

	return df
		
def eu_electricity_to_tensor():
    """
    -loads eu electricity data and turns into (time series x id) tensor
    """
    # load dataset
    df = load_eu_electricity()
    df = df.drop('date', axis=1)

    tensor_list = []

    # convert each time series to tensor
    for column in df.columns:
        tensor = torch.tensor(df[column].values, dtype=torch.float32)
        tensor_list.append(tensor)

    result_tensor = torch.stack(tensor_list)
    result_tensor = result_tensor.transpose(0,1)
    return result_tensor

def train_test_split_eu_elec(result_tensor, standardize=True):
    """
    -train test split for eu electricity dataset
    -returns dict with data
    -standardizes on train dataset values if True
    -returns dict with standardize metrics if True
    """
    train_cutoff = 0.7
    validation_cutoff = 0.8

    train_cutoff = int(round(train_cutoff * result_tensor.size(0)))
    validation_cutoff = int(round(validation_cutoff * result_tensor.size(0)))

    data_dict = {}
    data_dict["train"] = result_tensor[0:train_cutoff, :]
    data_dict["validation"] = result_tensor[train_cutoff : validation_cutoff, :]
    data_dict["test"] = result_tensor[validation_cutoff : , :]

    train_standardize_dict = None
    # normalize train and use matrics for val and test
    if standardize is True:
        data_dict["train"], train_standardize_dict = helpers.custom_standardizer(data_dict["train"])
        data_dict["validation"], _ = helpers.custom_standardizer(data_dict["validation"], train_standardize_dict)
        data_dict["test"], _ = helpers.custom_standardizer(data_dict["test"], train_standardize_dict)

    return data_dict, train_standardize_dict


def load_bavaria_electricity():
	"""
	returns 2D tensor of (timeseries x ID)
	"""

	df = pd.read_csv(CONFIG_DATA["south_germany"] / "south_germany.csv")

	# drop second timestamp column
	df = df.drop('cet_cest_timestamp', axis=1)
	df = df.drop('interpolated', axis=1)

	# rename to fit DataSetClass
	df.rename(columns={'utc_timestamp': 'date'}, inplace=True)
	df['date'] = pd.to_datetime(df['date'])



    # drop columns with more than 90% missing values
	threshold = 0.9 * len(df)
	columns_to_drop = df.columns[df.isna().sum() > threshold]
	df = df.drop(columns=columns_to_drop)

    # drop rows with more than 20% NANs
	threshold = 0.20 * len(df.columns)
	df = df[df.isna().sum(axis=1) <= threshold]

	# forward fill if previous values are present
	df = df.fillna(method='ffill')

    # values at the start are set to zero, no more NANs
	df = df.fillna(0)

    # Sort the DataFrame based on the 'date' column
	df = df.sort_values(by='date')

	df = df.drop(columns=['date'])

	data_array = df.values
	data_tensor = torch.tensor(data_array, dtype=torch.float32)

	return data_tensor