import torch
from utils import helpers
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_eval(model, dataloader):
	model.eval()
	preds_dict = {
		96 : {
				"mse" : 0, "mae" : 0, "smape":0, "p10" : 0, "p50" : 0, "p90" : 0
		},
		192 : {
				"mse" : 0, "mae" : 0, "smape":0, "p10" : 0, "p50" : 0, "p90" : 0
		},
		336 : {
				"mse" : 0, "mae" : 0, "smape":0, "p10" : 0, "p50" : 0, "p90" : 0
		},
		720: {
				"mse" : 0, "mae" : 0, "smape":0, "p10" : 0, "p50" : 0, "p90" : 0
		}}

	with torch.no_grad():
		for input, target in tqdm(dataloader, desc=f"Epoch: Validating"):
		
			# targets are saved as list, send each to device
			targets = (target[0].to(device), target[1].to(device), target[2].to(device), target[3].to(device))
			outputs = model(input.to(device))
		
		# for each prediciton length we calculate the metrics
		for target, output in zip(targets, outputs.values()):
			preds_dict[output.size(1)]["mse"] = preds_dict[output.size(1)]["mse"] + helpers.mean_squared_error(output, target)
			preds_dict[output.size(1)]["mae"] = preds_dict[output.size(1)]["mae"] + helpers.mean_absolute_error(output, target)
			preds_dict[output.size(1)]["smape"] = preds_dict[output.size(1)]["smape"] + helpers.symmetric_mean_absolute_percentage_error(output, target)
			preds_dict[output.size(1)]["p10"] = preds_dict[output.size(1)]["p10"] + helpers.percentile(output, target, 0.1)
			preds_dict[output.size(1)]["p50"] = preds_dict[output.size(1)]["p50"] + helpers.percentile(output, target, 0.5)
			preds_dict[output.size(1)]["p90"] = preds_dict[output.size(1)]["p90"] + helpers.percentile(output, target, 0.9)

	#print("Pred_len 96 MSE: ", preds_dict[96]["mse"])
	
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
	if epoch % 5 == 0:
		helpers.create_checkpoint(model, optimizer, scheduler, epoch, loss, global_step, "trial")
	eval_metrics_dict = fast_eval(model, dataloader_validation)
	return eval_metrics_dict