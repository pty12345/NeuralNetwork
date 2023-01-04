# https://github.com/imperial-qore/TranAD

import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint, pformat
# from beepy import beep

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'LPC_AD' in args.model or 'Attention' in args.model else w.view(-1))

	return torch.stack(windows)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []

	datasets = ['SMD', 'SWaT', 'WADI', 'MSL']
	assert dataset in datasets
 
	it = iter(machine_dict[dataset])
	m = next(it)
	for file in ['train', 'test', 'labels']:
		file = m + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
  
	while True:
		try:
			m = next(it)
			for i, file in enumerate(['train', 'test', 'labels']):
				file = m + file
				loader[i] = np.concatenate((loader[i], np.load(os.path.join(folder, f'{file}.npy'))), axis=0)
		except StopIteration:
			break
  
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(args.less_ratio, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def get_config(model):
    assert model.name in ['LPC_AD', 'TranAD', 'OmniAnomaly', 'USAD']
    config = {
		'lr':model.lr,
	}
    return config

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	weight_decay = 0 if modelname == 'L' else 1e-5
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
    # data's shape: (num, window, feats)
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'OmniAnomaly' in model.name:
		DEVICE = model.device
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				d = d
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds, hidden = [], torch.empty((1, 1)).to(DEVICE)
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		DEVICE = model.device
		if training:
			for d in data:
				d = d.to(DEVICE)
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else (len(data) // 5)

		drop_last = True if training else False
		dataloader = DataLoader(dataset, batch_size = bs, drop_last=drop_last)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		DEVICE = model.device
		if training:
			for d, _ in dataloader:
       			# (batch_size, len, feats) 
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2).to(DEVICE)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			model.eval()
			with torch.no_grad():
				loss_list, z_list = [], []
				for d, _ in dataloader:
					num = d.shape[0]
					window = d.permute(1, 0, 2).to(DEVICE)
					elem = window[-1, :, :].view(1, num, feats)
					z = model(window, elem)
					if isinstance(z, tuple): z = z[1]
     
					z_list.append(z)
					loss_list.append(l(z, elem)[0])
				loss_list = torch.cat(loss_list, dim=0)
				z_list = torch.cat(z_list, dim=1)
				return loss_list.cpu().detach().numpy(), z_list.cpu().detach().numpy()[0]
	elif 'LPC_AD' in model.name:
		if(not training): # reconstruct data
			data = data[0:data.shape[0]:model.window2]
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data) // 5
		drop_last = True if training else False
		dataloader = DataLoader(dataset, batch_size = bs, drop_last=drop_last)
		l1s = []
		if(training):
			# d: (batch_size, window, channel)
			DEVICE = model.device		
			for d, _ in dataloader:
				optimizer.zero_grad()
				x_repeat = d.repeat(model.repeat, 1, 1).to(DEVICE)
				''' forwarding '''
				x1, x2, px2, z1, z2, predict_z2 = model(x_repeat)
				''' calculate loss '''
				l1 = l(x_repeat[:, :x1.size()[1], :], x1)
				l2 = l(x_repeat[:, x1.size()[1]:, :], x2)
				l3 = l(x_repeat[:, x1.size()[1]:, :], px2)

				# # accumulate four loss items
				loss = torch.mean(l1) + torch.mean(l2) + torch.mean(l3)
				penalty = None
				if model.weight_decay > 0.000001:
					penalty = Regularization(model=model, weight_decay=model.weight_decay, p=2).to(device=DEVICE)
				if isinstance(penalty, torch.nn.Module):
					loss = loss + penalty(model)
				l1s.append(loss.item())

				
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()	
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), model.lr
		else:
			DEVICE = model.device
			l1s, y_pred = [], []
			model.eval()
			with torch.no_grad():
				for d, _ in dataloader:
					''' forwarding '''
					x1, x2, px2, z1, z2, predict_z2 = model(d.to(DEVICE))
					y_pred.append(px2)
	
					''' calculate loss '''
					# l1 = l(d[:, :x1.size()[1], :].to(DEVICE), x1)
					# l2 = l(d[:, x1.size()[1]:, :].to(DEVICE), x2)
					loss = l(d[:, x1.size()[1]:, :].to(DEVICE), px2)
					l1s.append(loss)
    
			l1s = torch.cat(l1s).cpu().detach().numpy().reshape(-1, feats)
			y_pred = torch.cat(y_pred).cpu().detach().numpy().reshape(-1, feats)
			if y_pred.shape[0] != dataO.shape[0]:
				y_pred = np.delete(y_pred, [-1], axis=0)
				l1s = np.delete(l1s, [-1], axis=0)
			# except:
			# 	print(y_pred.shape)
			# 	print(dataO.shape)
			# 	exit(0)
			# assert l1s.shape[0] == dataO.shape[0]
			return l1s, y_pred
      
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'LPC_AD'] or 'TranAD' in model.name: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	if(model.name in ['LPC_AD']):
		initRandomSeeds(seed=model.seed)
	### Training phase
	if args.model in ['LPC_AD', 'TranAD', 'OmniAnomaly', 'USAD'] and model.device != 'cpu':
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = "0"
		model, optimizer = MigrateToGPU(model, optimizer)
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}, with {model.device}{color.ENDC}')
		print(f'{color.BLUE}Training dataset scale : {trainO.shape}{color.ENDC}')
		num_epochs = 5; e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		if not args.unsave:
			save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()	
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	print(f'{color.BLUE}Testing dataset scale : {testO.shape}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

	### Plot curves
	if not args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		# print('l : ', l.shape)	
		# print('ls : ', ls.shape)
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
		result = pd.DataFrame([result])
		df = pd.concat([df, result], ignore_index=True)
	# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
	# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	# print('loss : ', loss.shape)
	# print('label : ', labels.shape)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	print(df)
	pprint(result)
 
	folder = os.path.join('result', args.model, args.dataset)
	os.makedirs(folder, exist_ok=True)
	# model config
	config = get_config(model)
	config.update({'Train dataset ratio':1 if args.less == False else args.less_ratio})
 
	# save config & result
	file = os.path.join(folder, 'result.txt')
	with open(file, 'a') as f:
		f.write(pformat(config)); f.write('\n')
		f.write(pformat(result)); f.write('\n\n')
	# pprint(getresults2(df, result))
	# beep(4)
