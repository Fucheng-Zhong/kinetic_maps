import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch, os, json, time
from torch.utils.data import DataLoader, random_split
from astropy.table import Table
from torch.optim.lr_scheduler import StepLR
from preprocess import normalization, denormalization


torch.set_default_dtype(torch.float32)

if not os.path.exists('models'):
    os.mkdir('models')

def save_config_as_json(config, path):
    filename = os.path.join(path, 'model_config.json')
    with open(filename, 'w') as json_file:
        json.dump(config, json_file, indent=2)

def load_config_from_json(path):
    filename = os.path.join(path, 'model_config.json')
    with open(filename, 'r') as json_file:
        config = json.load(json_file)
    return config


# data loader
class MyDataset(Dataset):
    def __init__(self, data):
        self.maps, self.paras = torch.from_numpy(data['maps']).to(torch.float32), torch.from_numpy(data['paras']).to(torch.float32)
        self.maps, self.paras = normalization(self.maps, self.paras) # preprocess
        print('Maps shape=', self.maps.shape, self.paras.shape)
        print('Paras type=', self.maps.dtype, self.paras.dtype)

    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        maps, paras = self.maps[idx].clone().detach().float(), self.paras[idx].clone().detach().float()
        return {'maps':maps, 'paras':paras}


# the network
class Network(nn.Module):
    def __init__(self, input_channel=2, input_dim=60, output_dim=3):
        super(Network, self).__init__()
        # cnn layer encoder
        kernel_size = 3
        stride = 2
        flattened_size = 256  # after 4 conv layers and pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Linear(flattened_size, output_dim),
            )
    def forward(self, maps):
        x = self.encoder(maps)
        return x



class Pipeline():
    """
    Initialize, one should set the size and channel of input
    """
    def __init__(self, cfg={}):
        self.cfg = {'model_name': 'test_CNN',
                    'input_size': 60,
                    'input_channel': 2,
                    'output_size': 2,
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'epochs': 5,
                    'step_size':10,
                    'gamma': 0.5,
                    'weight_decay': 0,
                    'train_ratio': 0.8,
                    'num_workers': 0,
                    }
        self.cfg.update(cfg)
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', self.device)
        if not os.path.exists('output'):
            print('Creating the output path:', 'output')
            os.makedirs('output')


    def Init(self, cfg_path=None):
        if cfg_path:
            self.cfg = load_config_from_json(cfg_path)
        self.model = Network(input_channel=self.cfg['input_channel'], input_dim=self.cfg['input_size'], output_dim=self.cfg['output_size'])
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg['learning_rate'], weight_decay=self.cfg['weight_decay']) 
        self.scheduler = StepLR(self.optimizer, step_size=self.cfg['step_size'], gamma=self.cfg['gamma'], verbose=True)
        self.history = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'time', 'learning rate'])
        self.path_name = os.path.join('models', self.cfg['model_name'])
        if not os.path.exists(self.path_name):
            print('Creating the model path:', self.path_name)
            os.makedirs(self.path_name)


    # save the training infomation
    def logging(self, info):
        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(info['epoch']+1, self.cfg['epochs'], info['train_loss'], info['val_loss']))
        self.history = pd.concat([self.history, pd.DataFrame([info])], ignore_index=True)
        self.history.to_csv(self.path_name + '/training_log.csv', index=False) 
        # save the best one  checkpoint
        if info['val_loss'] <= min(self.history['val_loss'].values):
            torch.save(self.model.state_dict(), self.path_name + "/network_paras.pth")
            print('save the best checkpoint of ', self.path_name)


    def load_data(self, data):
        if isinstance(data, str):
            data = Table.read(data, format='fits')
        dataset = MyDataset(data)
        train_size = int(self.cfg['train_ratio'] * len(dataset))
        valid_size = len(dataset) - train_size
        self.train_dataset, self.valid_dataset = random_split(dataset, [train_size, valid_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=self.cfg['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=self.cfg['num_workers'])
        print('Training set size:', len(self.train_loader), 'Validation set size:', len(self.valid_loader))

    # train function
    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc='Training'):
            self.optimizer.zero_grad()
            maps_input = batch['maps'].to(self.device)
            paras_true = batch['paras'].to(self.device)
            paras_pred = self.model(maps_input)
            loss = self.criterion(paras_pred, paras_true)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * maps_input.size(0)
        train_loss /= len(train_loader.dataset)
        return train_loss
    
    # validate/predict function
    def valid(self, valid_loader, load_model=False):
        if load_model: # load from weights
            self.model.load_state_dict(torch.load(self.path_name + "/network_paras.pth"))
            self.model = self.model.to(self.device)
        self.model.eval()
        valid_loss = 0
        predictions = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='Validation'):
                maps_input = batch['maps'].to(self.device)
                paras_true = batch['paras'].to(self.device)
                paras_pred = self.model(maps_input)
                loss = self.criterion(paras_true, paras_pred)
                valid_loss += loss.item() * maps_input.size(0)
                predictions.append(paras_pred.cpu().numpy())
        valid_loss = valid_loss / len(valid_loader.dataset)
        predictions = np.concatenate(predictions, axis=0)
        return valid_loss, predictions
    
    #=== training loop
    def train_loop(self):
        for epoch in range(self.cfg['epochs']):
            start_time = time.time()
            train_loss = self.train(self.train_loader)
            valid_loss, output = self.valid(self.valid_loader)
            end_time = time.time()
            # save the history
            info = {'epoch':epoch, 'train_loss':train_loss, 'val_loss':valid_loss, 'time':end_time-start_time, 'learning rate': self.scheduler.get_lr()[0]}
            self.logging(info)
            self.scheduler.step()
            print('learning rate', self.scheduler.get_lr())
        print('Training finished.')
        print('The best validation loss:', min(self.history['val_loss'].values))
        print('The model is saved in:', self.path_name)
        save_config_as_json(self.cfg, self.path_name)

    # reconstruct the images
    def predict(self, data, output_file='output/pred.fits'):
        if not isinstance(data, MyDataset):
            data = MyDataset(data)
        pred_dataset = data
        pred_loader = DataLoader(pred_dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=self.cfg['num_workers'])
        loss, output = self.valid(pred_loader, load_model=True)
        pred = denormalization(output)
        results = Table(pred)
        results.write(output_file, format='fits',overwrite=True)
        return results

