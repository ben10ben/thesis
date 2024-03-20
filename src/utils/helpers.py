from torch import save, mean, std, max, tensor
import numpy as np
from config import *
import torch.nn as nn
import torch

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
	save(checkpoint, f'{CONFIG_MODEL_LOCATION["revin"]}/{name}_epoch_{epoch}_loss_{loss}.pt') # is this ok to do? dont want to load full torch 
	print(f"Checkpointing succesfull after epoch {epoch}")



def calc_percentiles(predictions, actuals, percentile):
	# take predictions, actuals in correct order and return percentiles
	# return p10, p50, p90
	
	errors = np.abs(predictions - actuals)

	# Sort the errors
	sorted_errors = np.sort(errors)

    # Calculate p10, P50 and P90
	return np.percentile(sorted_errors, percentile)


def custom_standardizer(result_tensor, standardize_dict=None):
	if standardize_dict is None:
		standardize_dict = {
			"mean" : mean(result_tensor, dim=0),
			"std" : std(result_tensor, dim=0)
			}
	
	# add small epsilon to prevent division by zero
	epsilon = tensor(1e-6)
	standardize_dict["std"] = standardize_dict["std"].where(standardize_dict["std"] > epsilon, epsilon)

	result_tensor = (result_tensor - standardize_dict["mean"]) / standardize_dict["std"]

	return result_tensor, standardize_dict


def mean_squared_error(predictions, targets):
    """
    Compute Mean Squared Error (MSE) between predictions and targets.
    """
    return torch.mean((predictions - targets)**2)

def mean_absolute_error(predictions, targets):
    """
    Compute Mean Absolute Error (MAE) between predictions and targets.
    """
    return torch.mean(torch.abs(predictions - targets))

def symmetric_mean_absolute_percentage_error(predictions, targets):
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE) between predictions and targets.
    """
    numerator = torch.abs(predictions - targets)
    denominator = (torch.abs(predictions) + torch.abs(targets)) / 2.0
    return torch.mean(200.0 * (numerator / (denominator + 1e-8)))

def percentile(predictions, targets, percentile_value):
    """
    Compute the specified percentile metric between predictions and targets.
    """
    errors = torch.abs(predictions - targets)
    return torch.quantile(errors, percentile_value)

    
def evaluate(preds_dict, predictions, targets):
	# for each prediciton length we calculate the metrics
	for key, value in predictions.items():
		preds_dict[key]["mse"] = preds_dict[key]["mse"] + mean_squared_error(value, targets[key])
		preds_dict[key]["mae"] = preds_dict[key]["mae"] + mean_absolute_error(value, targets[key])
		preds_dict[key]["mae"] = preds_dict[key]["smape"] + symmetric_mean_absolute_percentage_error(value, targets[key])
		preds_dict[key]["p10"] = preds_dict[key]["p10"] + percentile(value, targets[key], 0.1)
		preds_dict[key]["p50"] = preds_dict[key]["p50"] + percentile(value, targets[key], 0.5)
		preds_dict[key]["p90"] = preds_dict[key]["p90"] + percentile(value, targets[key], 0.9)

	return preds_dict
