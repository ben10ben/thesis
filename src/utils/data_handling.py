from config import CONFIG_DATA
import pandas as pd
from torch.utils.data import Dataset
import torch

class SlidingWindowTimeSeriesDataset(Dataset):
	"""
	handles all dataset creation from (timeseries x id) tensors
	"""
	def __init__(self, data, window_size, pred_length):
		self.data = data
		self.window_size = window_size
		self.pred_length = pred_length

	def __len__(self):
		return self.data.size(0) - self.window_size - self.pred_length

	def __getitem__(self, index):	
		# Check if the index is within bounds
		if index < 0 or (index + self.pred_length + self.window_size) > self.data.size(0):
			raise StopIteration("Index out of bounds")

 	   # Calculate which id and time step to use based on the index
		window = self.data[index : index + self.window_size]
		target = self.data[index + self.window_size : index + self.window_size + self.pred_length]
		return window, target



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
	tensors = []

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
			tensors.append(tensor)

	# Concatenate all tensors along a new dimension, transpose
	result_tensor = torch.stack(tensors)
	result_tensor = result_tensor.transpose(0,1)

	if standardize == True:
		mean = torch.mean(result_tensor, dim=0)
		std = torch.std(result_tensor, dim=0)
		result_tensor = (result_tensor - mean) / std

	return result_tensor


def electricity_loader():
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
			"electricity_train" : pd.read_csv(CONFIG_DATA["electricity"] / "electricity_train.csv", index_col=False),
			"electricity_val" : pd.read_csv(CONFIG_DATA["electricity"] / "electricity_val.csv", index_col=False),
			"electricity_test" : pd.read_csv(CONFIG_DATA["electricity"] / "electricity_test.csv", index_col=False)
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
		
		dataset_dict = {}
		dataset_dict["electricity_train"] = output[(output['date'] >= "2014-01-01") & (output["date"] <= "2014-08-23")].copy() # cutoff taken from previous papers
		dataset_dict["electricity_val"] = output[(output['date'] > "2014-08-23") & (output["date"] < "2014-09-01")].copy() # cutoff taken from previous papers
		dataset_dict["electricity_test"] = output[(output['date'] >= "2014-09-01")].copy() # cutoff taken from previous papers

        # reset index and save for faster loading
		for key, value in dataset_dict.items():
			dataset_dict[key] = value.reset_index(drop=True)
			value.to_csv(CONFIG_DATA["electricity"] / f"{key}.csv", sep=',', index=False)


	return dataset_dict



def load_electricity_eu(self):
	"""
	processes europe electricity dataset
	-load
	-handle nan
	-return dict: train, val test
	"""
	df = pd.read_csv(CONFIG_DATA["euro_electricity"])
		
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
		