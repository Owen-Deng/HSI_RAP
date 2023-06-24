import logging
import math
import os
import time
import warnings
from multiprocessing import Process, Queue
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, 
                             cohen_kappa_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils import data

import models


def set_logger():
    log_path = './output/logs'
    os.makedirs(log_path, exist_ok=True)
    computer_name = os.environ.get("COMPUTERNAME")
    file_path = time.strftime(f"{log_path}/%y-%m-%d-{computer_name}.log")
    f_h = logging.FileHandler(file_path,mode='a',encoding='utf-8')
    s_h = logging. StreamHandler()
    formater = logging.Formatter('%(asctime)s %(levelname)s %(module)s %(lineno)d %(funcName)s\t%(message)s','%m%d %H:%M:%S')
    f_h.setFormatter(formater)
    s_h.setFormatter(formater)
    
    logg = logging.getLogger()
    logg.addHandler(f_h)
    logg.addHandler(s_h)
    
    logg.setLevel(logging.INFO)
    return logg


def load_mat(mat_path):
    if 'Chikusei' in mat_path:
        import hdf5storage
        hdfile = hdf5storage.loadmat(mat_path)
        mat = hdfile
        if 'GT' in mat:
            mat['GT'] = mat['GT'][0][0][0]
    else:
         mat = sio.loadmat(mat_path)
    
    return mat

def load_dataset(dataset_name,root_path,path_hsi = "",name_hsi = "",path_gt = "",name_gt = ""):
    if dataset_name !="":
        folder_path = Path(root_path,dataset_name)
        if not folder_path.exists():
            raise FileNotFoundError(folder_path)

        file_list = []
        for mat_file in folder_path.glob('*.mat'):
            file_list.append([str(mat_file),os.path.getsize(mat_file)])
            
        file_list.sort(key = lambda x: x[1])
        hsi = load_mat(file_list[-1][0])
        
        for vals in hsi.values():
            if isinstance(vals,np.ndarray):
                hsi = vals
                break
        
        gt = load_mat(file_list[0][0])
        for vals in gt.values():
            if isinstance(vals,np.ndarray):
                gt = vals
                break
    else:
        hsi = load_mat(path_hsi).get(name_hsi)
        gt = load_mat(path_gt).get(name_gt)
    return hsi,gt


def generate_patches(hsi,gt,patch_size):
    hsi_bands = hsi.shape[-1]
    pad_size = math.floor(patch_size / 2)
    hsi_pad = np.pad(hsi, ((pad_size, pad_size),(pad_size, pad_size), (0, 0)), 'reflect')
    #hsi_pad = np.pad(hsi, ((pad_size, pad_size),(pad_size, pad_size), (0, 0)), 'constant', constant_values=0)
    

    nonezero_indexes = np.where(gt != 0)
    hsi_patches = np.empty((len(nonezero_indexes[0]),patch_size, patch_size,hsi_bands), dtype='float32')
    for x, y, idx in zip(nonezero_indexes[0], nonezero_indexes[1], range(len(nonezero_indexes[0]))):
        hsi_patches[idx] = hsi_pad[x:x+patch_size, y:y+patch_size, :]

    return hsi_patches


def normalization(data):
    voxel = data.reshape(-1,data.shape[2])
    scaler = StandardScaler()
    voxel = scaler.fit_transform(voxel)
    return voxel.reshape(data.shape)


def apply_pca_producer(q: Queue,X, numComponents):
    newX = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[-1]))
    pca = PCA(n_components=numComponents,whiten=True)
    q.put(0)
    ret_X = pca.fit_transform(newX)
    ret_X = np.reshape(ret_X, (*X.shape[:2],-1))
    q.put(ret_X)

def apply_pca(X, numComponents):
    #set_start_method('spawn')
    while True:
        q=Queue()
        p1=Process(target=apply_pca_producer,args=(q,X,numComponents))
        p1.start()
        q.get()
        retry = 15
        while retry > 12 and q.empty():
            time.sleep(0.1)
            retry -= 0.1
        
        while retry > 0 and q.empty():
            time.sleep(0.5)
            retry -= 0.5
        
        if q.empty():
            if p1.is_alive():
                p1.kill()
        else:
            ret_X = q.get()
            return ret_X


def load_data(hsi_ns,label_ds,num_train_data,num_valid_data):
    shape_hsi = hsi_ns.shape
    num_class = len(set(label_ds))
    x_data_t = np.zeros(0,dtype='float32').reshape(0,shape_hsi[1],shape_hsi[2],shape_hsi[3])
    y_data_t = np.zeros(0,dtype='int64')
    x_data_v = np.zeros(0,dtype='float32').reshape(0,shape_hsi[1],shape_hsi[2],shape_hsi[3])
    y_data_v = np.zeros(0,dtype='int64')
    all_train_indexes = []
    for i in range(num_class):
        indexes = np.where(label_ds == i)
        
        partidxes_t = indexes[0][:num_train_data]
        partidxes_v = indexes[0][num_train_data:num_train_data+num_valid_data]
        
        x_data_t = np.append(x_data_t,hsi_ns.take(partidxes_t,axis=0),axis=0)
        y_data_t = np.append(y_data_t,label_ds.take(partidxes_t))
        all_train_indexes.extend(partidxes_t)
        x_data_v = np.append(x_data_v,hsi_ns.take(partidxes_v,axis=0),axis=0)
        y_data_v = np.append(y_data_v,label_ds.take(partidxes_v))
    logging.info(f'train indexes: {all_train_indexes}')
    return x_data_t,y_data_t,x_data_v,y_data_v


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat == y
    return float(torch.sum(cmp.type(y.dtype)))

def train(net, train_iter, valid_iter,train_round,args:dict):
    start_time = time.time()
    num_epochs = args['num_epoch']
    lr = args['started_lr']
    device = args['device']
    aug_prob = args.get('aug_prob',{})
    for key in aug_prob:
        if aug_prob[key] != 1:
            del aug_prob[key]
    aug_prob = np.array(list(aug_prob.keys()))
    fixed_lr = args.get('fixed_lr',False)
    test = args.get('test_during_training',True)
    optimizer_name = args.get('optimizer','Adam')
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,weight_decay=args['optimizer_weight_decay'])
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=lr,weight_decay=args['optimizer_weight_decay'])
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    if not fixed_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, lr*0.01)
    loss = nn.CrossEntropyLoss()
    test_acc = 0
    net.train()
    metric = Accumulator(3) 
    pbar = tqdm.tqdm(range(num_epochs))
    max_patch_size = train_iter.dataset.tensors[0].shape[-2]
    cur_patch_size = args.get('cur_patch_size',0)
    if cur_patch_size==0:
        cur_patch_size = max_patch_size
    
    cur_offset = (max_patch_size - cur_patch_size ) // 2
    test_acc_list= []
    amp = args.get('amp',False)
    if amp:
        scaler = GradScaler()
    warnings.filterwarnings('ignore')
    for epoch in pbar:
        metric.reset()
        for X, y in train_iter:
            optimizer.zero_grad()

            if cur_offset != 0:
                X = X[:,:,cur_offset:cur_patch_size+cur_offset,cur_offset:cur_patch_size+cur_offset,:]
                
            if amp:
                with autocast():
                    y_hat = net(X)
                    l = loss(y_hat, y)
                scaler.scale(l).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
            
            metric.add(l * X.shape[0], accuracy(y_hat,y), X.shape[0])
        

        if not fixed_lr:
            scheduler.step()
        
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
    
        if test:
            if valid_iter != None and (epoch+1) % 100 == 0 :
                if args.get('eval_epoch_acc',False):
                    test_acc = test_accuracy(net,valid_iter,device,True,0,cur_patch_size)
                    test_acc_list.append(test_acc)
                elif epoch != (num_epochs -1):
                    test_acc = test_accuracy(net,valid_iter,device,True,5,cur_patch_size)
                
                pbar.set_description(str(test_acc)+'|'+str(cur_patch_size))
        
    pbar.clear()    
    used_time = time.time()-start_time
    warnings.filterwarnings('default')
    logging.info(f'loss {train_l:.3f}, train acc {train_acc:.3f}, valid acc {test_acc:.3f}, time:{used_time:.1f} {np.mean(cur_patch_size)}')
    return epoch,cur_patch_size,test_acc_list


def augment_tradition(x,y):
    aug_x = np.empty((0,*x.shape[1:]),dtype=np.float32)
    aug_y = np.empty(0,dtype=np.int64)
    
    for idx in range(x.shape[0]):
        x_tmp = np.copy(x[idx])
        x_tmp = np.reshape(x_tmp,(1,*x_tmp.shape))
        x_data_t_flr = np.flip(x_tmp,axis=1)#左右翻转
        x_data_t_fud = np.flip(x_tmp,axis=2)#上下翻转
        x_data_t_rot90 = np.rot90(x_tmp,1,(1,2))#旋转
        x_data_t_rot180 = np.rot90(x_tmp,2,(1,2))
        x_data_t_rot270 = np.rot90(x_tmp,3,(1,2))
        x_data_t_trans = np.transpose(x_tmp,(0,2,1,3))#主对角线翻转
        x_data_t_trans_2 = np.flip(x_data_t_rot90,2)#副对角线翻转
        aug_x = np.concatenate((aug_x,x_data_t_flr,x_data_t_fud,
                                x_data_t_trans,x_data_t_trans_2,
                                x_data_t_rot90 ,x_data_t_rot180,x_data_t_rot270),
                                axis=0)
        tmp_y = y[idx].reshape(1)
        aug_y = np.concatenate((aug_y,tmp_y,
                                tmp_y,tmp_y,
                                tmp_y,tmp_y,tmp_y,tmp_y),axis=0)

    return aug_x,aug_y


def get_dataloader(train_x,train_y,valid_x,valid_y,batch_size,device):

    x_data_t = torch.from_numpy(train_x)
    y_data_t = torch.from_numpy(train_y)
    x_data_t = x_data_t.view(-1,1,x_data_t.shape[1],x_data_t.shape[2],x_data_t.shape[3])
    data_set_t = data.TensorDataset(x_data_t.to(device,non_blocking = True), y_data_t.to(device,non_blocking = True))
    dataloder_t = data.DataLoader(data_set_t, batch_size, shuffle=True,drop_last=False,pin_memory=False,num_workers=0)
    if valid_y.shape[0] > 0:
        x_data_v = torch.from_numpy(valid_x)
        y_data_v = torch.from_numpy(valid_y)
        x_data_v = x_data_v.view(-1,1,x_data_v.shape[1],x_data_v.shape[2],x_data_v.shape[3])
        data_set_v = data.TensorDataset(x_data_v.to(device,non_blocking = True), y_data_v.to(device,non_blocking = True))
        dataloder_v = data.DataLoader(data_set_v, batch_size, shuffle=True,drop_last=False,pin_memory=False,num_workers=0)
        return dataloder_t,dataloder_v

    return dataloder_t, None


def load_test_data(hsi_ns,label_ds,num_train_valid,device,batch_size):  # @save
    shape_hsi = hsi_ns.shape
    num_class = len(set(label_ds))
    x_data = np.zeros(0,dtype='float32').reshape(0,shape_hsi[1],shape_hsi[2],shape_hsi[3])
    y_data = np.zeros(0,dtype='int64')
    for i in range(num_class):
        indexes = np.where(label_ds == i)
        partidxes = indexes[0][num_train_valid:]
        x_data = np.append(x_data,hsi_ns.take(partidxes,axis=0),axis=0)
        y_data = np.append(y_data,label_ds.take(partidxes))
    
    # x_data = torch.from_numpy(x_data)
    # y_data = torch.from_numpy(y_data)
    logging.info(f'testing data shape: {x_data.shape}, {y_data.shape}')

    return x_data,y_data


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
def test_accuracy(net,test_iter,device,OnlyOA=False,num_batch = 0,patch_size = 0):
    total_preds = torch.zeros(0,device = device,dtype=torch.int64)
    total_trues = torch.zeros(0,dtype=torch.int64)
    net.eval()
    batch_count = 0
    data_patch_size = test_iter.dataset.tensors[0].shape[-2]
    if patch_size == 0:
        patch_size = data_patch_size
    
    datas = {}
    with torch.no_grad():
        for X,y in test_iter:
            X = X.to(device,non_blocking = True)

            cur_offset = (data_patch_size - patch_size ) // 2
            if cur_offset != 0:
                X = X[:,:,cur_offset:patch_size+cur_offset,cur_offset:patch_size+cur_offset,:]
                
            #X = X.to(device)
            #y = y.to(device)
            preds = net(X).argmax(axis=1)
            total_preds = torch.cat((total_preds,preds))
            total_trues = torch.cat((total_trues,y))
            batch_count += 1
            if num_batch > 0 and batch_count >= num_batch:
                break
    
    net.train()
    total_preds = total_preds.cpu().numpy()
    total_trues = total_trues.numpy()
    if OnlyOA:
        return accuracy_score(total_trues,total_preds)
    
    AAs = []
    c_m = confusion_matrix(total_trues,total_preds)
    for i in range(c_m.shape[0]):
        total_count = np.sum(c_m[i,:])
        AAs.append(0 if total_count == 0 else c_m[i,i]/total_count)
    
    return accuracy_score(total_trues,total_preds),cohen_kappa_score(total_trues,total_preds),AAs


def  save_classification_map(model,hsi_n,g_gt,dataset_name,fname):
    x_data = torch.from_numpy(hsi_n)
    x_data = x_data.view(-1,1,*x_data.shape[1:])
    x_data = x_data.to(torch.float32)
    
    data_set = data.TensorDataset(x_data)
    batch_size = 512
    test_set = data.DataLoader(data_set, batch_size, shuffle=False,pin_memory=True)
    total_preds = np.zeros(0,dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for (X,) in test_set:
            X = X.cuda()
            preds = model(X).argmax(axis=1)
            total_preds = np.append(total_preds,preds.cpu().detach().numpy())
    model.train()
    data_len = g_gt.shape[0]*g_gt.shape[1]
    nonezero_idxes = np.nonzero(g_gt.reshape(data_len))
    predict_map = np.zeros(data_len)
    for i,j in zip(nonezero_idxes[0],range(len(total_preds))):
        predict_map[i] = total_preds[j]+1

    predict_map = predict_map.reshape(g_gt.shape[0],g_gt.shape[1])

    gt_path = Path('./output/figures/',dataset_name,'ground_truth.jpg')
    os.makedirs(gt_path.parent,exist_ok=True)
    gt_path = str(gt_path)
    matplotlib.use('Agg')
    if not os.path.exists(gt_path):
        plt.matshow(g_gt,cmap=plt.cm.nipy_spectral)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(fname=gt_path,bbox_inches='tight',dpi=600,pad_inches=0)
        plt.close()

    map_path = Path('./output/figures/',dataset_name,fname+'.jpg')
    os.makedirs(map_path.parent,exist_ok=True)
    map_path = str(map_path)
    plt.matshow(predict_map,cmap=plt.cm.nipy_spectral)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fname=map_path,bbox_inches='tight',dpi=600,pad_inches=0)
    plt.close()


def get_model(model_name,num_band,num_class,device,aug_name,augs,augargs,patch_size=-1,dropout=0.5):
    model = getattr(models,model_name)
    if model_name == "HSI_CR":
        model = model(num_band,num_class,aug_name,augs,augargs,patch_size,dropout,device)
    else:
        model = model(num_band,num_class,patch_size)
    model.to(device,non_blocking = True)
    return model


def shuffle_hsi(hsi_n,label_d,seed=0):
    rd_seed = 0
    if seed == 0:
        rd_seed = np.random.randint(65535)
    else:
        rd_seed = seed
    hsi_ns = np.copy(hsi_n)
    label_ds = np.copy(label_d)

    np.random.seed(rd_seed)
    np.random.shuffle(hsi_ns)
    np.random.seed(rd_seed)
    np.random.shuffle(label_ds)
    return hsi_ns,label_ds


def init():
    #matplotlib.use('TkAgg')
    set_logger()
    np.set_printoptions(linewidth=1000,precision=5)

   
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu') 


if __name__ == '__main__':
    pass