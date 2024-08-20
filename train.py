import wandb
import torch
from softlabel import YieldPredictor
from data_utils import my_ds, get_raw_feature, get_data_spilt,load_data
import random
import pickle
import os
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import  mean_absolute_error, mean_squared_error,r2_score,accuracy_score,f1_score
import numpy as np
from data import rxn
import itertools
import pdb
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image

class my_ds(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray,margin):
        """
        X: Total feature length 2705 aimnet 50 context_fp 324 mordred 1295
        Y: yields"""
        self.X = X
        self.Y = Y
        self.margin = margin


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        y_scale = y * std + mean
        # soft_label = np.zeros(3)
        # should consider the margin case where the y_scale is close to the boundary, assign a margin class to it
        if y_scale <= mean - std / 2: # low yield
            soft_label = np.array([0, 0, 1])

        elif y_scale >  mean - std / 2 and y_scale < mean + std / 2 : # medium yield
            soft_label = np.array([0, 1, 0])

        elif y_scale >= mean + std / 2 : # high yield
            soft_label = np.array([1, 0, 0])

        return x, y, soft_label

 
def get_raw_feature(ids,standardize=True):
    X, Y = [], []
    for i,id in enumerate(ids):
        # context = rxns.get_context_one_hot(id)
        # fp = rxns.get_fp(id)
        fp = rxns.get_fp_fast(id)
        # print(len(fp))
        # context_fp0 = list(np.array(context + fp)) # 3807
        # aimnet0 = rxns.get_qm_fast(id)  #51
        # mordred0 = rxns.get_mordred(id)  #4944
        # aev0 = rxns.get_aev_fast(id) # 5148
        # x = aimnet0 + mordred0 + aev0 + context_fp0 # 13950
        x = fp
        X.append(x)
        y = rxns.get_yield(id)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)

def get_data_spilt():
    with open('./data/datav6_internal/normal_ids.pkl', 'rb') as file:
        normal_ids = pickle.load(file)
    with open('./data/datav6_internal/train_uncertain_ids.pkl', 'rb') as file:
        uncertain_ids = pickle.load(file)
    with open('./data/datav6_internal/test_clean_ids.pkl', 'rb') as file:
        test_ids = pickle.load(file)
    with open('./data/datav6_internal/test_uncertain_ids.pkl', 'rb') as file:
        test_u_ids = pickle.load(file)
    
    tmp = {}
    tmp['normal'] = normal_ids
    tmp['uncertain'] = uncertain_ids
    tmp['test'] = test_ids
    tmp['test_u'] = test_u_ids
    
    return tmp

def load_data(data, get_feature,config):

    normal,uncertain,test,test_u = data['normal'],data['uncertain'],data['test'],data['test_u']
    # combine normal and uncertain and split to train and valid and cali
    data = normal

    test_u = list(test_u)
    test = list(test)
    # test = list(test) + test_u[:int(len(test_u)*0.2)]
    test_u = test_u[int(len(test_u)*0.2):]

    train_ids = set(random.sample(list(data), int(len(data) * 0.9)))
    val_ids = data - train_ids
    
    X_train, Y_train = get_feature(train_ids)
    global mean
    mean = 64.05
    global std
    std = 22.46
    X_valid,Y_valid = get_feature(val_ids)
    # X_cal,Y_cal = get_feature(cali)
    X_test,Y_test = get_feature(test)
    X_test_u,Y_test_u = get_feature(test_u)
    # X_test_u,Y_test_u = get_feature(test_u)
    Y_train = (Y_train - mean)/std
    Y_test = (Y_test - mean)/std
    # Y_cal = (Y_cal - mean)/std
    Y_test_u = (Y_test_u - mean)/std
    Y_valid = (Y_valid - mean)/std
    train_loader = DataLoader(my_ds(X_train, Y_train,config.margin), batch_size=config.batch_size)
    valid_loader = DataLoader(my_ds(X_valid, Y_valid,config.margin), batch_size=config.batch_size)
    # cal_loader = DataLoader(my_ds(X_cal, Y_cal), batch_size=len(X_cal))
    test_loader = DataLoader(my_ds(X_test, Y_test,config.margin), batch_size=config.batch_size)
    test_loader_u = DataLoader(my_ds(X_test_u, Y_test_u,config.margin), batch_size=config.batch_size)
    return train_loader,valid_loader,test_loader,test_loader_u


def find_closest(predicted_tensor, actual_tensor):
    """Find the closest values in predicted_tensor to those in actual_tensor."""
    # Calculate the absolute differences between the actual values and each of the predictions
    diffs = torch.abs(predicted_tensor - actual_tensor)
    # Find the indices of the closest predictions
    closest_idxs = diffs.argmin(dim=1)
    # Gather the closest predictions using the indices
    closest_predictions = predicted_tensor[torch.arange(predicted_tensor.size(0)), closest_idxs]
    return closest_predictions


def ranked_accuracy(Ys, Y_preds, weights, mask=None):
    for y, y_pred,weight in zip(Ys, Y_preds,weights):
        weight = [w.detach().cpu() for w in weight]
        closest_predictions = torch.tensor([y_hat[0]*w[0] + y_hat[1]*w[1] + y_hat[2]*w[2] for y_hat,w in zip(y_pred,weight)])
        closest_predictions = torch.tensor([y_hat[0].detach().cpu().numpy() for y_hat in y_pred])
        closest_predictions = np.array([process_Y(y_hat) for y_hat in closest_predictions]) 
        acc = accuracy_score(y.cpu().detach().numpy(), closest_predictions)
    return acc



def ranked_R2(Ys, Y_preds,weights): #add mask, for those with mask = 0, we don't consider them
    r2_scores = []
    for y, y_pred,weight in zip(Ys, Y_preds,weights):
        weight = [w.detach().cpu() for w in weight]
        closest_predictions = torch.tensor([y_hat[0]*w[0] + y_hat[1]*w[1] + y_hat[2]*w[2] for y_hat,w in zip(y_pred,weight)])
        # closest_predictions = np.array([y_hat[0].detach().cpu().numpy() for y_hat in y_pred])
        r2 = r2_score(y.cpu().detach().numpy(), closest_predictions)
    r2_scores.append(r2)
    
    # Calculate the average R2 score for all samples
    avg_r2_score = np.mean(r2_scores)
    return avg_r2_score

def ranked_mae(Ys, Y_preds,weights):
    # Ys = torch.tensor(Ys.squeeze(), dtype=torch.float32)
    mae = []
    for y, y_pred,weight in zip(Ys, Y_preds,weights):
        weight = [w.detach().cpu() for w in weight]
        closest_predictions = torch.tensor([y_hat[0]*w[0] + y_hat[1]*w[1] + y_hat[2]*w[2] for y_hat,w in zip(y_pred,weight)])
        mae.append(mean_absolute_error(y.cpu().detach().numpy(), closest_predictions.cpu().detach().numpy()))
    mae = np.mean(mae)
    return mae

def ranked_rmse(Ys, Y_preds,weights):
    # Ys = torch.tensor(Ys.squeeze(), dtype=torch.float32)
    rmse = []
    for y, y_pred,weight in zip(Ys, Y_preds,weights):
        weight = [w.detach().cpu() for w in weight]
        closest_predictions = torch.tensor([y_hat[0]*w[0] + y_hat[1]*w[1] + y_hat[2]*w[2] for y_hat,w in zip(y_pred,weight)])
        rmse.append(mean_squared_error(y.cpu().detach().numpy(), closest_predictions.cpu().detach().numpy(), squared=False))
    rmse = np.mean(rmse)
    return rmse

# def ranked_R2(Ys, Y_preds, weights, mask=None): 
#     r2_scores = []
#     for y, y_pred, weight in zip(Ys, Y_preds, weights):
#         # Detach weights and convert to numpy
#         weight = [w.detach().cpu().numpy() for w in weight]
        
#         # Find the index of the largest weight
#         max_weight_idx = [np.argmax(w for w in weight)]
#         # pdb.set_trace()
#         # Select the prediction with the largest weight
#         closest_prediction = [y_p[max_weight_idx].detach().cpu().numpy() for y_p in y_pred]
        
#         # Apply mask if provided
#         if mask is not None and mask == 0:
#             continue
#         # Calculate R2 score
#         r2 = r2_score(y.cpu().detach().numpy(), np.array(closest_prediction).reshape(-1,1))
#     r2_scores.append(r2)
    
#     # Calculate the average R2 score for all samples
#     avg_r2_score = np.mean(r2_scores)
#     return avg_r2_score

# def ranked_mae(Ys, Y_preds, weights, mask=None):
#     mae = []
#     for y, y_pred, weight in zip(Ys, Y_preds, weights):
#         # Detach weights and convert to numpy
#         weight = [w.detach().cpu().numpy() for w in weight]
        
#         # Find the index of the largest weight
        
#         max_weight_idx = [np.argmax(w for w in weight)]
#         # Select the prediction with the largest weight
#         closest_prediction = [y_p[max_weight_idx].detach().cpu().numpy() for y_p in y_pred]
        
#         # Apply mask if provided
#         if mask is not None and mask == 0:
#             continue
        
#         # Calculate MAE
#         mae.append(mean_absolute_error(y.cpu().detach().numpy(),np.array(closest_prediction).reshape(-1,1)))
#     mae = np.mean(mae)
#     return mae

# def ranked_rmse(Ys, Y_preds, weights, mask=None):
#     rmse = []
#     for y, y_pred, weight in zip(Ys, Y_preds, weights):
#         # Detach weights and convert to numpy
#         weight = [w.detach().cpu().numpy() for w in weight]
        
#         # Find the index of the largest weight
#         max_weight_idx = [np.argmax(w for w in weight)]
        
#         # Select the prediction with the largest weight
#         # closest_prediction = y_pred[max_weight_idx].detach().cpu().numpy()
#         closest_prediction = [y_p[max_weight_idx].detach().cpu().numpy() for y_p in y_pred]
        
#         # Apply mask if provided
#         if mask is not None and mask == 0:
#             continue
        
#         # Calculate RMSE
#         rmse.append(mean_squared_error(y.cpu().detach().numpy(), np.array(closest_prediction).reshape(-1,1), squared=False))
#     rmse = np.mean(rmse)
#     return rmse

def density_plot(x, file_name):
    # x = x * std + mean
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x[:, 0].cpu().detach().numpy().squeeze(), fill=True, label='high', color='r', alpha=0.5, bw_adjust=0.5)
    sns.kdeplot(x[:, 1].cpu().detach().numpy().squeeze(), fill=True, label='medium', color='g', alpha=0.5, bw_adjust=0.5)
    sns.kdeplot(x[:, 2].cpu().detach().numpy().squeeze(), fill=True, label='low', color='b', alpha=0.5, bw_adjust=0.5)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Reset buffer's position to the beginning

    # Convert buffer to PIL Image
    image = Image.open(buf)
    
    # Log the image to wandb
    wandb.log({"density_plot_" + file_name: wandb.Image(image)})
    
    # Clear the current plot
    plt.close()


def custom_loss(pred_yield,y_true):
    mse_loss_fn = torch.nn.MSELoss()
    high_threshold = (mean + std / 2)/100
    low_threshold = (mean - std / 2)/100
    # Basic MSE loss for all predictions
    yield_high = pred_yield[:, 0]
    yield_medium = pred_yield[:, 1]
    yield_low = pred_yield[:, 2]

    loss_high = mse_loss_fn(yield_high, y_true)
    loss_medium = mse_loss_fn(yield_medium, y_true)
    loss_low = mse_loss_fn(yield_low, y_true)
    
    # Initialize penalties to zero
    penalty_high = torch.zeros_like(yield_high)
    penalty_medium = torch.zeros_like(yield_medium)
    penalty_low = torch.zeros_like(yield_low)

    # Apply penalties based on the predictions
    # If high MLP predicts below the high threshold, apply penalty
    penalty_high[yield_high < high_threshold] = (high_threshold - yield_high[yield_high < high_threshold])**2
    
    # If medium MLP predicts outside the medium range, apply penalty
    penalty_medium[(yield_medium <= low_threshold) | (yield_medium >= high_threshold)] = mse_loss_fn(yield_medium[(yield_medium <= low_threshold) | (yield_medium >= high_threshold)], y_true[(yield_medium <= low_threshold) | (yield_medium >= high_threshold)])
    
    # If low MLP predicts above the low threshold, apply penalty
    penalty_low[yield_low > low_threshold] = (yield_low[yield_low > low_threshold] - low_threshold)**2

    # Calculate total loss as sum of MSE loss and penalties
    # total_loss = loss_high + loss_medium + loss_low + penalty_high.sum() + penalty_medium.sum() + penalty_low.sum()
    total_loss = penalty_high.mean() + penalty_medium.mean() + penalty_low.mean()
    return total_loss


def margin_loss(y_cls, soft_label, y_pred, y,epoch,epsilon,episilon_0):
    mask = soft_label == 1
    y_hat = y_pred[mask]
    # Ensure y_cls[mask] is between 0 and 1
    y_cls_prob = y_cls[mask]


    # pdb.set_trace()
    # Add a small epsilon to avoid the exact 1 value which might cause overflow
    loss_1 = torch.clamp(1- y_cls_prob - epsilon, min=0) * torch.abs((y_hat - y))


    mask_0 = soft_label == 0
    y_hat_0 = y_pred[mask_0]
    y_cls_prob_0 = y_cls[mask_0]
    y_true_0 = y.repeat(1,3)[mask_0]
    # epsilon_0 = 0.3 #
    loss_0 = torch.clamp(y_cls_prob_0 - episilon_0, min=0) * torch.abs(y_hat_0 - y_true_0)
    loss = loss_1.mean() + epoch/500 * loss_0.mean()

    return loss.mean()




# def margin_loss(y_cls, soft_label, y_pred, y):
#     # mask = soft_label == 1
#     # y_hat = y_pred[mask]
#     # # Ensure y_cls[mask] is between 0 and 1
#     # pdb.set_trace()
#     # y_cls_prob = y_cls[mask]
#     margin = 0.2
#     adjusted_preds= y * std + mean
#     adjusted_preds_h= y_pred[:,0] * std + mean
#     adjusted_preds_h=  adjusted_preds

#     epsilon_h = torch.where(adjusted_preds_h <  mean + std / 2 ,0,margin)

#     adjusted_preds_m = y_pred[:, 1] * std + mean
#     adjusted_preds_m=  adjusted_preds
#     epsilon_m = torch.where(
#         ((adjusted_preds_m > 20) & (adjusted_preds_m < mean - std / 2)) |
#         ((adjusted_preds_m > mean + std / 2) & (adjusted_preds_m < 85)),
#         0,  
#         margin)
    
#     adjusted_preds_l = y_pred[:, 2] * std + mean
#     adjusted_preds_l=  adjusted_preds
#     epsilon_l = torch.where(adjusted_preds_l > mean - std / 2, 1,margin)

#     loss = torch.clamp(1- y_cls[:,0] - epsilon_h, min=0) * torch.abs((y_pred[:,0] - y)) \
#          + torch.clamp(1- y_cls[:,1] - epsilon_m, min=0) * torch.abs((y_pred[:,1] - y)) \
#          + torch.clamp(1- y_cls[:,2] - epsilon_l, min=0) * torch.abs((y_pred[:,2] - y))

#     # pdb.set_trace()
#     return loss.mean()

def train(train_loader, model, optimizer,scheduler,scale,tau,epoch,epsilon,epsilon_0,device):
    model.train()
    total_loss = 0
    # total_counts = 0
    # mse_loss_fn = torch.nn.MSELoss(reduction='none')
    # bce_loss_fn = torch.nn.BCELoss(reduction='none')
    scale_weight = 1.05
    for i, (X, Y,softlabel) in enumerate(train_loader):
        scale *= scale_weight
        X = X.to(device).float()
        Y = Y.to(device).reshape(-1, 1).float()
        softlabel = softlabel.to(device).float().reshape(-1,3)
        Y = Y.view(-1,1)
        optimizer.zero_grad()
        Y_pred,Y_cls,_,_,reaction_encoding= model(X,hard = False,tau = tau)

        loss = margin_loss(Y_cls, softlabel,Y_pred,Y,epoch,epsilon,epsilon_0) 
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    epoch_loss = total_loss / len(train_loader)
    train_r2, train_mae, train_rmse,_,_,_,_,_,_,_,_,_,_,_,_,_= validate(train_loader,
                                               model,
                                               test = False,
                                               device=device)
    return (train_r2, train_mae, train_rmse,epoch_loss,Y_pred,Y,softlabel,Y_cls,reaction_encoding)

def validate(valid_loader, model, test, device):
    model.eval()
    Ys, Y_preds, weights = [], [], []
    high, medium, low = [], [], []
    y_high, y_medium, y_low = [], [], []
    with torch.no_grad():
        for i, (X, Y, softlabel) in enumerate(valid_loader):
            X = X.to(device).float()
            Y = Y.to(device).reshape(-1, 1).float()

            Y_l = [process_Y(y * std + mean) for y in Y.cpu().numpy()]
            # pdb.set_trace()
            Y_thresh = Y * std + mean
            Y_thresh = Y_thresh.repeat(1,3,1).cpu()
            Y_pred, weight, _, _ ,reaction_encoding= model(X, hard=False, tau=config.tau)
            out = Y_pred.cpu() * std + mean
            # pdb.set_trace()

            pdb.set_trace()
            high_mask = out > mean + std / 2
            high_indices = torch.nonzero(high_mask, as_tuple=True)[0]
            medium_mask = (out <= mean + std / 2) & (out >= mean - std / 2)
            medium_indices = torch.nonzero(medium_mask, as_tuple=True)[0]
            low_mask = out < mean - std / 2
            low_indices = torch.nonzero(low_mask, as_tuple=True)[0]


            Ys_high = Y[high_indices] * std + mean
            Y_preds_high = out[high_mask]
            Ys_medium = Y[medium_indices] * std + mean
            Y_preds_medium = out[medium_mask]
            Ys_low = Y[low_indices] * std + mean
            Y_preds_low = out[low_mask]


            Ys.append(Y.cpu() * std + mean)

            high.append(Ys_high)
            medium.append(Ys_medium)
            low.append(Ys_low)
            y_high.append(Y_preds_high)
            y_medium.append(Y_preds_medium)
            y_low.append(Y_preds_low)

            Y_preds.append(out)
            weights.append(weight)
        
        # # Concatenate all results
        # Ys = torch.cat(Ys, dim=0)
        # Y_preds = torch.cat(Y_preds, dim=0)
        # weights = torch.cat(weights, dim=0)
        # pdb.set_trace()
        # Calculate overall metrics
        r2 = ranked_R2(Ys, Y_preds, weights)
        mae = ranked_mae(Ys, Y_preds, weights)
        rmse = ranked_rmse(Ys, Y_preds, weights)
        acc = ranked_accuracy(Y_l, Y_preds, weights)
        print("acc: ",acc)
        
# Assuming high, y_high, medium, y_medium, low, y_low are tensors

        # Flatten the tensors (if needed)
        high = torch.cat(high, dim=0)
        y_high = torch.cat(y_high, dim=0)
        medium =  torch.cat(medium, dim=0)
        y_medium = torch.cat(y_medium, dim=0)
        low = torch.cat(low, dim=0)
        y_low =  torch.cat(y_low, dim=0)

        # Calculate metrics for high yield range
        r2_high = r2_score(high.cpu().numpy(), y_high.cpu().numpy())
        mae_high = mean_absolute_error(high.cpu().numpy(), y_high.cpu().numpy())
        rmse_high = mean_squared_error(high.cpu().numpy(), y_high.cpu().numpy(), squared=False)

        # Calculate metrics for medium yield range
        r2_medium = r2_score(medium.cpu().numpy(), y_medium.cpu().numpy())
        mae_medium = mean_absolute_error(medium.cpu().numpy(), y_medium.cpu().numpy())
        rmse_medium = mean_squared_error(medium.cpu().numpy(), y_medium.cpu().numpy(), squared=False)

        # Calculate metrics for low yield range
        r2_low = r2_score(low.cpu().numpy(), y_low.cpu().numpy())
        mae_low = mean_absolute_error(low.cpu().numpy(), y_low.cpu().numpy())
        rmse_low = mean_squared_error(low.cpu().numpy(), y_low.cpu().numpy(), squared=False)
        
        if test:
            # Add any test-specific operations here if needed
            pass
    
    # Return overall metrics and per-range metrics
    return (r2, mae, rmse, r2_high, mae_high, rmse_high, r2_medium, mae_medium, rmse_medium, r2_low, mae_low, rmse_low, out, Y, weight,reaction_encoding)
# def validate(valid_loader, model, test,device):
#     model.eval()
#     Ys, Y_preds,weights = [], [], []
#     high,medium,low = [],[],[]
#     with torch.no_grad():
#         for i, (X, Y,softlabel) in enumerate(valid_loader):
#             X = X.to(device).float()
#             Y = Y.to(device).reshape(-1, 1).float()
#             Y_pred,weight,_,_ = model(X,hard = False,tau = config.tau)
#             out = Y_pred.cpu()*std + mean
#             Ys.append(Y.cpu()*std + mean)
#             # print(out)
#             # pdb.set_trace()
#             Y_preds.append(out)
#             weights.append(weight)
#         if test:
#             # print("grundTh: ",Ys)
#             # print("pred: ",Y_preds)
#             pass
    
#     r2 = ranked_R2(Ys, Y_preds,weights)
#     mae = ranked_mae(Ys, Y_preds,weights)
#     rmse = ranked_rmse(Ys, Y_preds,weights)

#     return (r2, mae, rmse,out,Y,weight)


torch.cuda.empty_cache()
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

reaxys_mean = 64.05
reaxys_median = 68.00
reaxys_std = 22.46
# preparing data
global mean
global std
mean = reaxys_mean
std = reaxys_std

folder = "./data/datav6_internal"
rxns = rxn(folder)
ids = rxns.all_idx2()
print(f'Total number of reactions: {len(ids)}')
for i, id in enumerate(ids):
    fp = rxns.get_fp_fast(id)
    aev = rxns.get_aev_fast(id)
    qm = rxns.get_qm_fast(id)
    steric = rxns.get_steric_desps(id)
    mordred = rxns.get_mordred_fast(id)
    yld = rxns.get_yield(id)
    print(yld)
    print(len(fp), len(aev), len(qm), len(steric), len(mordred))
    # print(aev)
    # print(qm)
    # print(steric)
    break
train_ids0 = pickle.load(open(os.path.join(folder, 'normal_ids.pkl'), 'rb'))
train_uncertain_ids = pickle.load(open(os.path.join(folder, 'train_uncertain_ids.pkl'),'rb'))
train_ids0 = train_ids0.union(train_uncertain_ids)
test_ids = pickle.load(open(os.path.join(folder, 'test_clean_ids.pkl'), 'rb'))


hyperparameter_grid = {
    "output_size": [1], 
    "margin": [15], # control the margin of the soft label
    "feature_dim": [3072],
    "label_dim": [1],  # "label_dim": [256, 512, 1024, 2048],
    "learning_rate": [5e-4],  
    "epochs": [1],
    "hidden_size": [512],  # "hidden_size": [512, 1024, 2048, 4096
    "threshold": [0.2],
    "layers": [1,2,3],
    "batch_size": [128],  
    "scale": [1e4],
    "tau": [3], # control the temperature of gumbel softmax
    "checkpt_path": ["./final_models"],
    "dropout": [0.2],
    "seed": [0],
    "epsilon": [0.2],
    "epsilon_0": [0.1]
}
results = {}
def process_Y(Y):
    Y = np.array(Y)
    Y = np.clip(Y, 0, 100)
    if Y > reaxys_mean + reaxys_std/2:
        return [1,0,0]
    elif Y < reaxys_mean - reaxys_std/2:
        return [0,0,1]
    else:
        return [0,1,0]
    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for config in itertools.product(*hyperparameter_grid.values()):
    config_dict = dict(zip(hyperparameter_grid.keys(), config))
    metrics = {
        "test_r2": [],
        "test_mae": [],
        "test_rmse": [],
        "test_r2_u": [],
        "test_mae_u": [],
        "test_rmse_u": []
    }
    for i in range(5):
    # Initialize wandb with the current configuration
        wandb.init(project="softlabel_tuning", entity="nd", config=config_dict)
        config = wandb.config

        all_data = get_data_spilt()
        train_loader, valid_loader,test_loader,test_loader_u = load_data(all_data,get_raw_feature,config)
        model = YieldPredictor(config.feature_dim,
                            config.label_dim,
                            config.hidden_size,
                            config.threshold,
                            config.layers,
                            config.dropout,
                            config.output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        lambda_lr = lambda step: 0.1 if step >= 50 else 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_lr)
        bad_counter = 0
        best = 8848
        patience = 20

        for epoch in range(config.epochs):
            train_r2, train_mae, train_rmse,epoch_loss,y_pred,Y,softlabel,Y_cls,reaction_encoding= train(train_loader,
                                                                                        model,
                                                                                        optimizer,
                                                                                        scheduler,
                                                                                        scale = config.scale,
                                                                                        tau = config.tau,
                                                                                        epoch = epoch,
                                                                                        epsilon= config.epsilon,
                                                                                        epsilon_0 = config.epsilon_0,
                                                                                        device = device)
            

            wandb.log({"train_r2": train_r2, "train_mae": train_mae, "train_rmse": train_rmse, "epoch_loss": epoch_loss})
            valid_r2, valid_mae, valid_rmse,_,_,_,_,_,_,_,_,_,val_out,val_y,val_weight,_ = validate(valid_loader,
                                                                                model,
                                                                                test = False,
                                                                                device=device)
            wandb.log({"valid_r2": valid_r2, "valid_mae": valid_mae,"valid_rmse": valid_rmse})
            
            if valid_rmse < best:
                best = valid_rmse
                bad_counter = 0
                best_weights = model.state_dict()
                torch.save(model.state_dict(), os.path.join(config.checkpt_path, "softlabel_best_model_' + {config.name} + '.pt"))
            else:
                bad_counter += 1

            if bad_counter >= patience:
                print("Early stopping triggered at epoch:", epoch)
                model.load_state_dict(best_weights)
                break
            


        
        density_plot(val_out,'val_'+ str(i))
        density_plot(y_pred,'train_'+ str(i))
        # density_plot(val_y,"true")
        test = True
        test_r2, test_mae, test_rmse,r2_high, mae_high, rmse_high, r2_medium, mae_medium, rmse_medium, r2_low, mae_low, rmse_low,_,_,_,_= validate(test_loader,
                                                model,
                                                test,
                                                device)
        print(f"Test R2: {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}",'r2_high:',r2_high,'mae_high:',mae_high,'rmse_high:',rmse_high,'r2_medium:',r2_medium,'mae_medium:',mae_medium,'rmse_medium:',rmse_medium,'r2_low:',r2_low,'mae_low:',mae_low,'rmse_low:',rmse_low)
        test_r2_u, test_mae_u, test_rmse_u,r2_high, mae_high, rmse_high, r2_medium, mae_medium, rmse_medium, r2_low, mae_low, rmse_low,_,_,_,_ = validate(test_loader_u,
                                                    model,
                                                    test,
                                                    device)
        print(f"Test R2 U: {test_r2_u:.4f}, Test MAE: {test_mae_u:.4f}, Test RMSE: {test_rmse_u:.4f}")
        print("_____________________________________________________________________________________________")
        batch = config.batch_size
        all_embeddings =  reaction_encoding.reshape(-1, 1024).cpu().detach().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        data = tsne.fit_transform(all_embeddings)
        C = np.argmax(Y.cpu().detach().numpy, -1) 


        fig=plt.figure(figsize=(10,10),facecolor='white') #3,3
        plt.scatter(data[:,0], data[:,1], s=20)  #1

        # fig=plt.figure(figsize=(3,3),facecolor='white') #3,3
        # plt.tick_params(labelsize=7)
        # plt.scatter(data[:,0], data[:,1], s=1,c=color)  #1
        plt.savefig('t-SNE_visualization.png')
        wandb.log({"test_r2": test_r2, "test_mae": test_mae, "test_rmse": test_rmse})
                # Append the results from this run
        metrics["test_r2"].append(test_r2)
        metrics["test_mae"].append(test_mae)
        metrics["test_rmse"].append(test_rmse)
        metrics["test_r2_u"].append(test_r2_u)
        metrics["test_mae_u"].append(test_mae_u)
        metrics["test_rmse_u"].append(test_rmse_u)

    # Store the mean and standard deviation for each metric
    results[str(config_dict)] = {metric: (np.mean(values), np.std(values)) for metric, values in metrics.items()}
    wandb.finish()


for config, stats in results.items():
    print(config)
    for metric, (mean, std) in stats.items():
        print(f"{metric}: Mean = {mean}, Std = {std}")