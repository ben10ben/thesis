from torch import save, mean, std
import numpy as np
from config import *


def create_checkpoint(model, optimizer, scheduler, epoch, loss, global_step, name):
	checkpoint = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict' : scheduler.state_dict(),
		'epoch': epoch,
		'loss': loss,
		'global_step_writer' : global_step,
	}
	# model, revin, affine, epoch, loss
	save(checkpoint, f'{CONFIG_OUTPUT_PATH["series_standardized"]}/{name}_epoch_{epoch}_loss_{loss}.pt') # is this ok to do? dont want to load full torch 
	print(f"Checkpointing succesfull after epoch {epoch}")



def calc_percentiles(predictions, actuals, percentile):
	# take predictions, actuals in correct order and return percentiles
	# return p10, p50, p90
	
	errors = np.abs(predictions - actuals)

	# Sort the errors
	sorted_errors = np.sort(errors)

    # Calculate p10, P50 and P90
	return np.percentile(sorted_errors, percentile)


def custom_standardizer(result_tensor):
    standardize_dict = {
        "mean" : mean(result_tensor, dim=0),
        "std" : std(result_tensor, dim=0)
        }
	
    result_tensor = (result_tensor - standardize_dict["mean"]) / standardize_dict["std"]

    return result_tensor, standardize_dict