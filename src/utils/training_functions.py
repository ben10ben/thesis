import torch
from utils import helpers
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_eval(model, dataloader, device):
	model.eval()
	preds_dict = {
		96 : {"mse" : 0},
		192 : {"mse" : 0},
		336 : {"mse" : 0},
		720: {"mse" : 0}}

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

	preds_dict[length]["mse"] = preds_dict[length]["mse"] /  len(dataloader)
	print(F"Validation metrics: {preds_dict[96]}")
	return preds_dict


def train_one_epoch(epoch, model, device, dataloader_train, dataloader_validation, optimizer, scheduler, writer, checkpoint_path=None, save_model=True, validate=True):
	global_step = 0
	total_loss = 0
	best_val_loss = 99999999
	best_model = None
	
	for epoch in range(1, epoch + 1):
		total_loss = 0
		model.train()
		for input, target in tqdm(dataloader_train, desc=f"Epoch: {epoch}"):
			optimizer.zero_grad()
			if len(target) == 4:
				loss = model(input.to(device), (target[0].to(device), target[1].to(device), target[2].to(device), target[3].to(device)))
			else:
				loss = model(input.to(device), target.to(device))
			
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

			global_step+=1

		scheduler.step()
		
		writer.add_scalar('Loss/train', (total_loss/len(dataloader_train)), epoch)

		if validate==True:
			eval_metrics_dict = fast_eval(model, dataloader_validation, device)
		else:
			eval_metrics_dict = None
		# safe best model measured on validation datasplit
		val_loss = eval_metrics_dict[96]["mse"].item()
		writer.add_scalar('Loss/validation', (val_loss), epoch)

		if  val_loss < best_val_loss:
			best_model = model
			best_val_loss = val_loss
		if save_model==True:
			helpers.create_checkpoint(model, optimizer, scheduler, epoch, loss, global_step, checkpoint_path)

	writer.close()

	return eval_metrics_dict, model
#	if best_model != None:
#		return eval_metrics_dict, best_model
#	else:
#		return eval_metrics_dict, model