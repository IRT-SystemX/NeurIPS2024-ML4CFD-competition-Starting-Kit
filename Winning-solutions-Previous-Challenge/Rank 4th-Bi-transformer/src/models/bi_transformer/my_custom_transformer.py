import time
import random
import datetime as dt
from typing import List, Union

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from lips.dataset.scaler.standard_scaler_iterative import StandardScalerIterative


# K-means clustering
# using a uniform at random initialisation
def centroid_uniform_initialisation(X:Tensor, k: int=1000):
    samples = np.random.choice(a=X.shape[0], replace=False, size=k)
    return X[samples,:]

# using the k-means++ initialisation
def kmeans_plusplus_initialisation(X: torch.Tensor, k: int):
    n_samples = X.shape[0]
    # Step 1: Choose one center uniformly at random from among the data points.
    centroids = X[torch.randint(0, n_samples, (1,))]

    for _ in range(k - 1):
        # Step 2: For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
        distances = torch.cdist(X, centroids, p=2)
        min_distances = torch.min(distances, dim=1)[0]

        # Step 3: Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
        probabilities = min_distances / torch.sum(min_distances)
        new_centroid_idx = torch.multinomial(probabilities, 1)

        # Add the new centroid to the list of centroids
        centroids = torch.cat([centroids, X[new_centroid_idx]], dim=0)

    return centroids

def k_means(X:Tensor, k: int=1000, n_iter: int=100, method: str='kmeans_uniform'):
    if method == 'kmeans_uniform':      centroids = centroid_uniform_initialisation(X,k)
    elif method == 'kmeans_pp':         centroids = kmeans_plusplus_initialisation(X,k)
    else:                               raise ValueError('Invalid method for centroid initialisation')

    for i in range(n_iter):
        # compute the distance of each point to the centroids
        distances = torch.cdist(X, centroids, p=2)

        # define the index associated to each centroid
        cluster_idx = torch.argmin(distances, dim=1)

        # compute the new centroids
        new_centroids = torch.stack([X[cluster_idx == i].mean(0) for i in range(k)])

        # end when we have converged
        if torch.all(centroids == new_centroids):
            break

        centroids = new_centroids
    return cluster_idx, centroids


def skeleton_sampling(X:Tensor, k: int=1000, method: str='kmeans_uniform', n_iter: int=100):
    if method == 'kmeans_uniform':
        _, centroids = k_means(X, k=k, n_iter=n_iter, method=method)
        #clustroid_idx = np.random.choice(a=X.shape[0], size=k, replace=False)
    elif method == 'kmeans_pp':
        # extracting a skeleton from the cloudpoint
        _, centroids = k_means(X, k=k, n_iter=n_iter, method=method)
    elif method == 'kmeans_pp_init':
        centroids = kmeans_plusplus_initialisation(X, k)
    elif method == 'random':
        clustroid_idx = np.random.choice(a=X.shape[0], size=k, replace=False)
        return clustroid_idx
    else: raise ValueError('Invalid method for skeleton sampling, must be one of kmeans_uniform, kmeans_pp, kmeans_pp_init, random')

    clustroid_idx = []
    for point in centroids:
        clustroid_idx.append(torch.argmin(((X - point)**2).mean(dim=1)).item())

    return clustroid_idx

def data_batching(X: Tensor, batch_size: int=7):
    n = X.shape[0]
    q,r = X.shape[0]//batch_size, X.shape[0]%batch_size
    split_X = []
    for i in range(X.shape[0]//batch_size):
        split_X.append(X[batch_size*i:batch_size*(i+1),:])
    if r > 0:
        split_X.append(torch.cat([X[q*batch_size:,:], split_X[-1][-(batch_size-r):,:]]))
    return split_X, r

def smoothL1(pred, target, keptcomponent=False):
    x = pred - target
    y = torch.minimum(x.abs(), x * x)
    if keptcomponent:
        return y.mean(0)
    else:
        return y.mean()

def smoothSoftmax(x):
    s = torch.nn.functional.softmax(x, dim=1)
    ss = torch.nn.functional.relu(x)
    num = ss * 0.1 + s
    denom = num.sum(1).unsqueeze(-1)
    return num / denom

class MLP(torch.nn.Module):
    def __init__(self, layers_sizes):
        super(MLP, self).__init__()
        layers = [torch.nn.Linear(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes)-1)]
        self.mlp = torch.nn.Sequential()
        for i, layer in enumerate(layers):
            self.mlp.append(layer)
            if i < len(layers) - 1:
                self.mlp.append(torch.nn.ReLU())
    def forward(self, x):
        return self.mlp(x)


class AttentionBlock(torch.nn.Module):
    def __init__(self, sIN, sOUT, yDIM=7, sPROJ=None):
        super(AttentionBlock, self).__init__()

        if sPROJ is None:
            sPROJ = sOUT

        self.k = torch.nn.Linear(yDIM, sPROJ, bias=False)
        self.q = torch.nn.Linear(sIN, sPROJ, bias=False)
        self.v = torch.nn.Linear(yDIM, sPROJ, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        YK = self.k(y)
        XQ = self.q(x)
        YV = self.v(y)

        M = torch.matmul(XQ, YK.t())/np.sqrt(YK.shape[1])
        M = smoothSoftmax(M)
        output = torch.matmul(M, YV)

        return output

class TransformerBlock(torch.nn.Module):
    def __init__(self, sIN, sOUT, yDIM=7, sPROJ=None, layers: List[int] = [8, 64, 64, 8]):
        super(TransformerBlock, self).__init__()
        self.att = AttentionBlock(sIN, sOUT, yDIM, sPROJ)
        self.mlp = MLP(layers)
        self.layer_norm = LayerNorm(sOUT)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z1 = self.layer_norm(x)
        w = self.layer_norm(y)
        z1 = self.att(z1, w)
        z1 = z1 + x

        z2 = self.layer_norm(z1)
        z2 = self.mlp(z2)
        z2 = z2 + z1

        return z2

class Ransformer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Ransformer, self).__init__()

        self.k = kwargs['k']

        self.encoder = torch.nn.Sequential(
            Linear(7, 64),
            nn.ReLU(),
            Linear(64, 64),
            nn.ReLU(),
            Linear(64, 32)
        )

        self.transf1 = TransformerBlock(32, 32, yDIM=32, layers=[32, 64, 64, 64, 32])
        self.transf2 = TransformerBlock(32, 32, yDIM=32, layers=[32, 64, 64, 64, 32])
        self.transf3 = TransformerBlock(32, 32, yDIM=32, layers=[32, 64, 64, 64, 32])

        self.decoder = torch.nn.Sequential(
            Linear(32, 64),
            nn.ReLU(),
            Linear(64, 64),
            nn.ReLU(),
            Linear(64, 32),
            nn.ReLU(),
            Linear(32, 16),
            nn.ReLU(),
            Linear(16, 4)
        )

    def forward(self, x, y):
        # encoding x and y
        x_sample = x[torch.randint(0, x.shape[0], (self.k,))].clone()
        y = torch.cat([y, x_sample], dim=0)

        x_enc = self.encoder(x)
        y_enc = self.encoder(y)

        x_out = self.transf1(x_enc, y_enc)
        y_out = self.transf1(y_enc, y_enc)
        
        x_out = self.transf2(x_out, y_out)
        y_out = self.transf2(y_out, y_out)

        x_out = self.transf3(x_out, y_out)

        out = self.decoder(x_out)
        return out


class AugmentedSimulator():
    def __init__(self,benchmark,**kwargs):
        np.random.seed(42)
        torch.manual_seed(42)

        self.name = "AirfRANSSubmission"
        chunk_sizes=benchmark.train_dataset.get_simulations_sizes()
        scalerParams={"chunk_sizes":chunk_sizes}
        self.scaler = StandardScalerIterative(**scalerParams)

        self.model = None
        self.hparams = kwargs
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda:
            print('Using GPU')
        else:
            print('Using CPU')

        self.model = Ransformer(**self.hparams)

    def process_dataset(self, dataset, training: bool) -> List[Data]:
        coord_x=dataset.data['x-position']
        coord_y=dataset.data['y-position']
        surf_bool=dataset.extra_data['surface']
        position = np.stack([coord_x,coord_y],axis=1)

        nodes_features,node_labels=dataset.extract_data()
        if training:
            print("Normalize train data")
            nodes_features, node_labels = self.scaler.fit_transform(nodes_features, node_labels)
            print("Transform done")
        else:
            print("Normalize not train data")
            nodes_features, node_labels = self.scaler.transform(nodes_features, node_labels)
            print("Transform done")

        torchDataset=[]
        nb_nodes_in_simulations = dataset.get_simulations_sizes()
        start_index = 0
        # check alive
        t = dt.datetime.now()

        print("Creating the dataset & computing the skeleton...")
        for nb_nodes_in_simulation in tqdm(nb_nodes_in_simulations):
            #still alive?
            if dt.datetime.now() - t > dt.timedelta(seconds=60):
                print("Still alive - index : ", end_index)
                t = dt.datetime.now()
            end_index = start_index+nb_nodes_in_simulation
            simulation_positions = torch.tensor(position[start_index:end_index,:], dtype = torch.float) 
            simulation_features = torch.tensor(nodes_features[start_index:end_index,:], dtype = torch.float) 
            simulation_labels = torch.tensor(node_labels[start_index:end_index,:], dtype = torch.float) 
            simulation_surface = torch.tensor(surf_bool[start_index:end_index])

            # creating the skeleton
            clustroid_idx = skeleton_sampling(simulation_positions, k=self.hparams['k'], method=self.hparams['skeleton_method'], n_iter=self.hparams['skeleton_n_iter'])
            skeleton_features = torch.clone(simulation_features[clustroid_idx,:])
            skeleton_pos = torch.clone(simulation_positions[clustroid_idx,:])

            sampleData=Data(pos=simulation_positions,
                            x=simulation_features, 
                            y=simulation_labels,
                            surf = simulation_surface.bool(),
                            skeleton_features = skeleton_features,
                            skeleton_pos = skeleton_pos)
            torchDataset.append(sampleData)
            start_index += nb_nodes_in_simulation
        print("Dataset created.")
        return torchDataset

    def train(self,train_dataset, save_path=None):
        train_dataset = self.process_dataset(dataset=train_dataset,training=True)
        print("Start training")
        model = global_train(self.device, train_dataset, self.model, self.hparams,criterion = 'L1Smooth')
        print("Training done")

    def predict(self,dataset,**kwargs):
        print(dataset)
        test_dataset = self.process_dataset(dataset=dataset,training=False)
        self.model.eval()
        avg_loss_per_var = np.zeros(4)
        avg_loss = 0
        avg_loss_surf_var = np.zeros(4)
        avg_loss_vol_var = np.zeros(4)
        avg_loss_surf = 0
        avg_loss_vol = 0
        iterNum = 0

        predictions=[]
        with torch.no_grad():
            # print(len(test_dataset))
            for i, data in tqdm(enumerate(test_dataset)):
                # print(i)
                data_clone = data.clone()
                data_clone = data_clone.to(self.device)

                X = torch.arange(data_clone.x.shape[0]).reshape(-1,1)
                batch_indices, r = data_batching(X, batch_size = self.hparams["batch_size"])

                out = torch.empty(size=(0,4)).to(self.device)
                for i in range(len(batch_indices)):
                    batch_id = batch_indices[i]
                    data_batch = Data(pos = data_clone.pos[batch_id.squeeze()], \
                                                      x = data_clone.x[batch_id.squeeze()], \
                                                      y = data_clone.y[batch_id.squeeze()], \
                                                      surf = data_clone.surf[batch_id.squeeze()], \
                                                      skeleton_features = data_clone.skeleton_features, \
                                                      skeleton_pos = data_clone.skeleton_pos)
                    batch_out = self.model(data_batch.x, data_batch.skeleton_features)
                    out = torch.cat([out, batch_out], dim=0)
                
                # removing the duplicates from the output
                if r>0:
                    out = out[:-(self.hparams["batch_size"]-r),:]

                targets = data.y.to(self.device)
                loss_criterion = nn.MSELoss(reduction = 'none')

                loss_per_var = loss_criterion(out, targets).mean(dim = 0)
                loss = loss_per_var.mean()
                loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
                loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
                loss_surf = loss_surf_var.mean()
                loss_vol = loss_vol_var.mean()  

                avg_loss_per_var += loss_per_var.cpu().numpy()
                avg_loss += loss.cpu().numpy()
                avg_loss_surf_var += loss_surf_var.cpu().numpy()
                avg_loss_vol_var += loss_vol_var.cpu().numpy()
                avg_loss_surf += loss_surf.cpu().numpy()
                avg_loss_vol += loss_vol.cpu().numpy()  
                iterNum += 1

                out = out.cpu().data.numpy()
                prediction = self._post_process(out)
                predictions.append(prediction)
        print("Results for test")
        print(avg_loss/iterNum, avg_loss_per_var/iterNum, avg_loss_surf_var/iterNum, avg_loss_vol_var/iterNum, avg_loss_surf/iterNum, avg_loss_vol/iterNum)
        predictions= np.vstack(predictions)
        predictions = dataset.reconstruct_output(predictions)
        return predictions

    def _post_process(self, data):
        try:
            processed = self.scaler.inverse_transform(data)
        except TypeError:
            processed = self.scaler.inverse_transform(data.cpu())
        return processed


def global_train(device, train_dataset, network, hparams, criterion = 'L1Smooth', reg = 1):
    model = network.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = hparams['lr'])
    start = time.time()

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    epoch_nb = 0

    # If we don't have a subsampling value, we use the whole dataset
    if hparams['subsampling'] == "None":
        print(f"No subsampling, batch size of {hparams['batch_size']}")
        train_dataset_sampled = []
        for data in train_dataset:
            data_clone = data.clone()
            X = torch.arange(data_clone.x.shape[0]).reshape(-1,1)
            batch_indices, r = data_batching(X, batch_size = hparams["batch_size"])

            for batch_id in batch_indices:
                new_data = Data(pos = data_clone.pos[batch_id.squeeze()], \
                                x = data_clone.x[batch_id.squeeze()], \
                                y = data_clone.y[batch_id.squeeze()], \
                                surf = data_clone.surf[batch_id.squeeze()], \
                                skeleton_features = data_clone.skeleton_features, \
                                skeleton_pos = data_clone.skeleton_pos)
                train_dataset_sampled.append(new_data)
        train_loader = DataLoader(train_dataset_sampled, batch_size = hparams["dataloader_batch_size"], shuffle = True, drop_last=False)
        del(train_dataset_sampled)

        total_steps = len(train_loader) * hparams['nb_epochs']
    else:
        print(f"Subsampling with {hparams['subsampling']} samples per simulation and {hparams['batch_size']} batch size")
        iterations_per_simulation = hparams["subsampling"]//hparams["batch_size"]
        if hparams["subsampling"]%hparams["batch_size"] != 0: iterations_per_simulation += 1
        total_steps = len(train_dataset) * iterations_per_simulation
        total_steps = total_steps//hparams["dataloader_batch_size"]
        if total_steps%hparams["dataloader_batch_size"] != 0: total_steps += 1
        total_steps = total_steps*hparams['nb_epochs']

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = hparams['lr'],
        total_steps = total_steps,
    )

    print(f"nb_epochs: {hparams['nb_epochs']}")
    print(f"number of simulations: {len(train_dataset)}")
    print(f"scheduler total_steps: {total_steps}")

    for epoch in pbar_train:
        epoch_nb += 1
        print('Epoch: ', epoch_nb)

        # If we have a subsampling value, we sample a new random dataset at each epoch
        if hparams['subsampling'] != "None":
            train_dataset_sampled = []
            for data in train_dataset:
                data_sampled = data.clone()
                idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
                idx = torch.tensor(idx)

                data_sampled.pos = data_sampled.pos[idx]
                data_sampled.x = data_sampled.x[idx]
                data_sampled.y = data_sampled.y[idx]
                data_sampled.surf = data_sampled.surf[idx]

                X = torch.arange(data_sampled.x.shape[0]).reshape(-1,1)
                batch_indices, r = data_batching(X, batch_size = hparams["batch_size"])

                for batch_id in batch_indices:
                    new_data = Data(pos=data_sampled.pos[batch_id.squeeze()], \
                                    x=data_sampled.x[batch_id.squeeze()], \
                                    y=data_sampled.y[batch_id.squeeze()], \
                                    surf=data_sampled.surf[batch_id.squeeze()], \
                                    skeleton_features=data_sampled.skeleton_features, \
                                    skeleton_pos=data_sampled.skeleton_pos)
                    train_dataset_sampled.append(new_data)
            train_loader = DataLoader(train_dataset_sampled, batch_size = hparams["dataloader_batch_size"], shuffle = True, drop_last=False)
            del(train_dataset_sampled)

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train_model(device, model, train_loader, optimizer, lr_scheduler, criterion, reg=reg)
        if criterion == 'MSE_weighted':
            train_loss = reg*loss_surf + loss_vol
        del(train_loader)

        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)

    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)

    return model


def train_model(device, model, train_loader, optimizer, scheduler, criterion='L1Smooth', reg: Union[int, None]=1.0):
    model.train()
    avg_loss_per_var = torch.zeros(4, device = device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device = device)
    avg_loss_vol_var = torch.zeros(4, device = device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iterNum = 0

    # defining the loss function
    loss_criterion = nn.MSELoss(reduction = 'none')
    if criterion == 'MSE':
        loss_criterion = nn.MSELoss(reduction = 'none')
    elif criterion == 'MAE':
        loss_criterion = nn.L1Loss(reduction = 'none')
    elif criterion == 'L1Smooth':
        loss_criterion = smoothL1

    for i, data in tqdm(enumerate(train_loader)):
        data_clone = data.clone()
        data_clone = data_clone.to(device)   
        optimizer.zero_grad()

        out = model(data_clone.x, data_clone.skeleton_features)
        targets = data_clone.y
        
        loss_per_var = loss_criterion(out, targets).mean(dim = 0)
        total_loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        if (reg is not None):            
            (loss_vol + reg*loss_surf).backward()           
        else:
            total_loss.backward()
        
        optimizer.step()
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol 
        iterNum += 1

    return  avg_loss.cpu().data.numpy()/iterNum,            \
            avg_loss_per_var.cpu().data.numpy()/iterNum,    \
            avg_loss_surf_var.cpu().data.numpy()/iterNum,   \
            avg_loss_vol_var.cpu().data.numpy()/iterNum,    \
            avg_loss_surf.cpu().data.numpy()/iterNum,       \
            avg_loss_vol.cpu().data.numpy()/iterNum
