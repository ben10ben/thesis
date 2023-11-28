from torch import save
import numpy as np
from config import *
from torch.utils.data import Dataset
from typing import List

"""
DEPRECATED
class SlidingWindowTimeSeriesDataset(Dataset):
#    def __init__(self, data, window_size, pred_length: List[int]):
#    if not isinstance(pred_length, list):
#            raise TypeError("pred_length must be a list")
    def __init__(self, data, window_size, pred_length):
        self.data = data
        self.window_size = window_size
        self.pred_length = pred_length

        self.total_seq_size = self.window_size + self.pred_length
        
        if self.total_seq_size > data.shape[1]:
            raise ValueError("The total sequence size cannot be greater than the length of the time series data.")
        
    def __len__(self):
        # We count each individual window, for each id
        return self.data.shape[0] * (self.data.shape[1] - self.total_seq_size + 1)

    def __getitem__(self, index):
        # Calculate which id and time step to use based on the index
        num_windows_per_id = self.data.shape[1] - self.total_seq_size + 1
        id_index = index // num_windows_per_id
        time_index = index % num_windows_per_id
        
        # Extract the window for the current id and time step
        window = self.data[id_index, time_index:time_index+self.window_size, :]
        target = self.data[id_index, time_index+self.window_size:time_index+self.total_seq_size, :]
        
        return window, target
"""    


def create_checkpoint(model, optimizer, scheduler, epoch, loss, global_step, name):
	checkpoint = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict' : scheduler.state_dict(),
		'epoch': epoch,
		'loss': loss,
		'global_step_writer' : global_step,
	}
	save(checkpoint, f'{CONFIG_OUTPUT_PATH["series_standardized"]}/model_{name}_epoch_{epoch}_loss_{loss}.pt') # is this ok to do? dont want to load full torch 
	print(f"Checkpointing succesfull after epoch {epoch}")



def calc_percentiles(predictions, actuals, percentile):
	# take predictions, actuals in correct order and return percentiles
	# return p10, p50, p90
	
	errors = np.abs(predictions - actuals)

	# Sort the errors
	sorted_errors = np.sort(errors)

# Calculate p10, P50 and P90
	return np.percentile(sorted_errors, percentile)

