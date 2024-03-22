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
	print(F"Validation metrics: {preds_dict}")
	return preds_dict


def train_one_epoch(epoch, model, device, dataloader_train, dataloader_validation, optimizer, scheduler, writer):
	global_step = 0
	model.train()
	total_loss = 0
	
	for input, target in tqdm(dataloader_train, desc=f"Epoch: {epoch}"):
		optimizer.zero_grad()
		if len(target) == 4:
			loss = model(input.to(device), (target[0].to(device), target[1].to(device), target[2].to(device), target[3].to(device)))
		else:
			loss = model(input.to(device), target.to(device))
		
		loss.backward()
		optimizer.step()
		total_loss += loss.item()

		writer.add_scalar('train_loss', loss, global_step)
		writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], global_step)
		lr =  optimizer.param_groups[0]['lr']
		global_step+=1

	print(f'Epoch {epoch}, MSE-Loss: {total_loss / (len(dataloader_train) * 4)}, LR: {lr}')

	scheduler.step()
	writer.close()
	#if epoch % 5 == 0:
	#	helpers.create_checkpoint(model, optimizer, scheduler, epoch, loss, global_step, "trial")
	eval_metrics_dict = fast_eval(model, dataloader_validation, device)

	return eval_metrics_dict