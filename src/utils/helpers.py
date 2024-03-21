from torch import save, mean, std, max, tensor
import numpy as np
from config import *
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F

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

    
def full_eval(model, dataloader, device):
	model.eval()
	preds_dict = {
		96 : {
				"mse" : 0, "mae" : 0, "p10" : 0, "p50" : 0, "p90" : 0
		},
		192 : {
				"mse" : 0, "mae" : 0, "p10" : 0, "p50" : 0, "p90" : 0
		},
		336 : {
				"mse" : 0, "mae" : 0, "p10" : 0, "p50" : 0, "p90" : 0
		},
		720: {
				"mse" : 0, "mae" : 0, "p10" : 0, "p50" : 0, "p90" : 0
		}}

	with torch.no_grad():
		for input, target in tqdm(dataloader, desc=f"Epoch: Validating"):
			if isinstance(target, torch.Tensor):
				targets = target.to(device) 
			else:
				targets = [t for t in target]

			outputs = model(input.to(device))
            
			targets = (targets,) if not isinstance(targets, tuple) else targets


			# for each prediciton length we calculate the metrics
			for target, output in zip(targets, outputs.values()):
				length = output.size(1)

				preds_dict[length]["mse"] = preds_dict[length]["mse"] + F.mse_loss(output, target)
				preds_dict[length]["mae"] = preds_dict[length]["mae"] + F.l1_loss(output, target)
				#preds_dict[length]["smape"] = preds_dict[length]["smape"] + helpers.symmetric_mean_absolute_percentage_error(output, target) #TODO correct the formula
				preds_dict[length]["p10"] = preds_dict[length]["p10"] + percentile(output, target, 0.1)
				preds_dict[length]["p50"] = preds_dict[length]["p50"] + percentile(output, target, 0.5)
				preds_dict[length]["p90"] = preds_dict[length]["p90"] + percentile(output, target, 0.9)

	preds_dict[length]["mse"] = preds_dict[length]["mse"] /  len(dataloader)
	preds_dict[length]["mae"] = preds_dict[length]["mae"]  /  len(dataloader)
	#preds_dict[length]["smape"] = preds_dict[length]["smape"] /  len(dataloader)
	preds_dict[length]["p10"] = preds_dict[length]["p10"]  / len(dataloader)
	preds_dict[length]["p50"] = preds_dict[length]["p50"]  / len(dataloader)
	preds_dict[length]["p90"] = preds_dict[length]["p90"] / len(dataloader)

	
	return preds_dict
