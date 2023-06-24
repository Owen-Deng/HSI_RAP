import logging
import math
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data

import emp
import settings
import utils as ut


def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.
    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels
    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)

    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def Gnoisegen(x, snr):
    Nx = len(x)  # 求出信号的长度
    noise = np.random.randn(Nx)# 用randn产生正态分布随机数 
    signal_power = np.sum(x*x)/Nx# 求信号的平均能量
    noise_power = np.sum(noise*noise)/Nx# 求信号的平均能量
    noise_variance = signal_power/(math.pow(10., (snr/10)))#计算噪声设定的方差值
    noise = math.sqrt(noise_variance/noise_power)*noise# 按照噪声能量构成相应的白噪声
    return noise


def add_noise(hsi):
    o_shape=hsi.shape
    hsi = hsi.astype(np.float64)
    hsi = hsi.reshape(-1,o_shape[-1])
    for i in range(hsi.shape[0]):
        factor = (1.25-0.75) * np.random.random() + 0.75
        hsi[i] *= factor
        noise = Gnoisegen(hsi[i],25)
        hsi[i] += noise
    
    return hsi.reshape(o_shape)


def preprocess_data(dataset_name,num_pca,num_emp,global_setting,trainings,load_origin=False):
    spectral_var = global_setting.get('spectral_var',False)
    
    cache_data = Path(DATASET_FOLDER,dataset_name,f'{num_pca}_{num_emp}.npz')
    loaded = False
    if not spectral_var and not load_origin and cache_data.exists():
        hsi_pca, label_list, gt = np.load(str(cache_data)).values()

        if hsi_pca.shape[-1] == num_pca*(num_emp*2+1):
            logging.info(f'load cached data: {hsi_pca.shape}, {label_list.shape}, {gt.shape}')
            loaded = True
        else:
            logging.info('data incorrect, loading original data.')
    
    if not loaded:
        hsi, gt = ut.load_dataset(dataset_name,DATASET_FOLDER)
        if spectral_var:
            hsi = add_noise(hsi)
        hsi_len = hsi.shape[0] * hsi.shape[1]
        label_list = gt.reshape(hsi_len).astype(np.int64)
        
        label_list = np.delete(label_list,np.where(label_list== 0)) #删除没有标记的样本
        label_list = label_list - 1  #标签要从0开始
        
        hsi_norm = ut.normalization(hsi)
        hsi_pca = ut.apply_pca(hsi_norm,num_pca)
        if num_emp > 0:
            hsi_pca = emp.build_emp(base_image=hsi_pca, num_openings_closings=num_emp)
        logging.info(f'load original data: {hsi_pca.shape}, {label_list.shape}, {gt.shape}')
        if not spectral_var:
            np.savez(str(cache_data),hsi_pca,label_list,gt)
    
    patch_size = 0
    for key in trainings:
        if trainings[key]['patch_size'] > patch_size:
            patch_size = trainings[key]['patch_size']
    
    hsi_patches = ut.generate_patches(hsi_pca,gt,patch_size)
    return hsi_patches, label_list, hsi_pca,gt


def get_spilted_data(hsi_patches, label_list, gt, dataset_name,seed,g_hyperpara):
    test_batch_size = g_hyperpara['test_batch_size'].get(dataset_name,g_hyperpara['default_test_batch_size'])
    
    if g_hyperpara.get('disjoint',False):
        gt_nonezero = np.nonzero(gt)
        hsi_patches_unflatten = np.zeros((gt.shape[0],gt.shape[1],*hsi_patches.shape[1:]),dtype=np.float32)
        for idx in range(gt_nonezero[0].shape[0]):
            hsi_patches_unflatten[gt_nonezero[0][idx],gt_nonezero[1][idx]] = hsi_patches[idx]
        
        gt_train, gt_test = sample_gt(gt,0.5,'disjoint')
        
        gt_train_nonezero = np.nonzero(gt_train)
        gt_test_nonezero = np.nonzero(gt_test)
        
        hsi_patches_train = np.empty((gt_train_nonezero[0].shape[0],*hsi_patches.shape[1:]),dtype=np.float32)
        for idx in range(gt_train_nonezero[0].shape[0]):
            hsi_patches_train[idx] =  hsi_patches_unflatten[gt_train_nonezero[0][idx],gt_train_nonezero[1][idx]]
        
        hsi_patches_test = np.empty((gt_test_nonezero[0].shape[0],*hsi_patches.shape[1:]),dtype=np.float32)
        for idx in range(gt_test_nonezero[0].shape[0]):
            hsi_patches_test[idx] =  hsi_patches_unflatten[gt_test_nonezero[0][idx],gt_test_nonezero[1][idx]]
        
        label_list_train = gt_train.flatten().astype(np.int64)
        label_list_train = np.delete(label_list_train,np.where(label_list_train== 0)) #删除没有标记的样本
        label_list_train = label_list_train - 1  #标签要从0开始
        
        label_list_test = gt_test.flatten().astype(np.int64)
        label_list_test = np.delete(label_list_test,np.where(label_list_test== 0)) #删除没有标记的样本
        label_list_test = label_list_test - 1  #标签要从0开始
        
        hsi_shuffled, label_shuffled = ut.shuffle_hsi(hsi_patches_train, label_list_train,seed)
        train_x,train_y,valid_x,valid_y = ut.load_data(hsi_shuffled, label_shuffled,g_hyperpara['num_train_data'], g_hyperpara['num_valid_data'])
        test_x, test_y = hsi_patches_test,label_list_test
        
    else:
        hsi_shuffled, label_shuffled = ut.shuffle_hsi(hsi_patches, label_list,seed)

        test_x,test_y = ut.load_test_data(hsi_shuffled, label_shuffled, g_hyperpara['num_train_data']+g_hyperpara['num_valid_data'], g_hyperpara['device'],test_batch_size)

        train_x,train_y,valid_x,valid_y = ut.load_data(hsi_shuffled, label_shuffled,g_hyperpara['num_train_data'], g_hyperpara['num_valid_data'])
        
    return train_x,train_y,valid_x,valid_y, test_x,test_y


def get_test_dataloader(test_x,test_y,batch_size):
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    test_x = test_x.view(-1,1,test_x.shape[1],test_x.shape[2],test_x.shape[3])
    data_set = data.TensorDataset(test_x, test_y)
    return data.DataLoader(data_set, batch_size, shuffle=True)


def get_augmented_dataloader(train_x,train_y,valid_x,valid_y,args):
    aug_func_name = args.get('aug_func',None)
    batch_size = args['batch_size']

    if aug_func_name != None:
        if isinstance(aug_func_name,list):
            aug_func_names = aug_func_name
        else:
            aug_func_names = [aug_func_name]
        
        augment_x = np.zeros((0,*train_x.shape[1:]),dtype='float32')
        augment_y = np.zeros(0,dtype='int64')
        
        for func_name in aug_func_names:
            aug_func = getattr(ut,func_name)
            tmpx,tmpy = aug_func(train_x,train_y)
            augment_x = np.concatenate([augment_x,tmpx],axis=0)
            augment_y = np.concatenate([augment_y,tmpy],axis=0)
        
        train_x = np.concatenate([train_x,augment_x],axis=0)
        train_y = np.concatenate([train_y,augment_y],axis=0)

    train_iter,validation_iter = ut.get_dataloader(train_x,train_y,valid_x,valid_y,batch_size,args['device'])
    return train_iter,validation_iter


def train_single(train_iter,validation_iter,test_iter,train_round,args):
    num_band = train_iter.dataset.tensors[0].shape[-1]
    patch_size = 0
    
    model_name = args.get('model','HSI_CR')

    aug_prob = args.get('aug_prob',{})
    for key in aug_prob:
        if aug_prob[key] != 1:
            del aug_prob[key]
    augs = list(aug_prob.keys())
    net = ut.get_model(model_name,num_band, args['num_class'],args['device'],args.get('aug_name',''),augs,args.get('aug_args',{}),args['patch_size'],args.get('dropout',0.5))

    epoch = 0
    train_start = time.time()
    test_start = time.time()
    epoch,ret_ps,test_acc_list = ut.train(net, train_iter, test_iter,train_round,args)
    if args['eval_epoch_acc']:
        return test_acc_list,0,[],net, epoch,test_start-train_start,0
    else:
        try:
            OA, Kappa, AA = ut.test_accuracy(net, test_iter,args['device'],patch_size=ret_ps)
        except KeyboardInterrupt:
            logging.info('KeyboardInterrupt, stop testing accuracy...')
            return 0,0,[],net,epoch,0,0
        test_end = time.time()
        return OA,Kappa,AA,net, epoch,test_start-train_start,test_end-test_start
 

def train_dataset(dataset_name,g_hyperpara,trainings:dict):
    hsi_patches, label_list,hsi_pca, gt = preprocess_data(dataset_name,g_hyperpara['num_pca'],g_hyperpara['num_emp'],g_hyperpara,trainings)
    num_class = len(set(label_list))
    time_str = time.strftime('%y%m%d_%H%M%S')
    ret_dict={}
    seed_i = g_hyperpara.get('seed',0)
    if seed_i == 0:
        seed_i = SEED
    
    try:
        for round in range(g_hyperpara['train_round']):
            seed = seed_i + round*10
            logging.info(f' Train round {round} started, seed: {seed}'.center(80,"="))  
            start_round_time = time.time()
            train_x,train_y,valid_x,valid_y,test_x,test_y = get_spilted_data(hsi_patches, label_list,gt,dataset_name,seed,g_hyperpara)
                
            for aug_name,args in trainings.items():
                start_aug_time = time.time()
   
                args['amp'] = g_hyperpara.get('amp',False)
                patch_size = args['patch_size']
                offset = (train_x.shape[1] - patch_size )// 2

                new_train_x = train_x[:,offset:offset+patch_size,offset:offset+patch_size,:]
                new_test_x = test_x[:,offset:offset+patch_size,offset:offset+patch_size,:]
                    
                new_valid_x = valid_x[:,offset:offset+patch_size,offset:offset+patch_size,:]
                test_batch_size = g_hyperpara['test_batch_size'].get(dataset_name,g_hyperpara['default_test_batch_size'])
                test_iter = get_test_dataloader(new_test_x,test_y,test_batch_size)
                train_iter, validation_iter = get_augmented_dataloader(new_train_x,train_y,new_valid_x,valid_y,args)                        
                data_aug_time = time.time()- start_aug_time
                t_shape_x = list(train_iter.dataset.tensors[0].shape)
                if validation_iter == None:
                    v_shape_x = 'None'
                else:
                    v_shape_x = list(validation_iter.dataset.tensors[0].shape)
                logging.info(f' {aug_name} use {data_aug_time:.1f} s, train: {str(t_shape_x)}, valid: {str(v_shape_x)} '.center(80,"-"))
                folder_name = time_str + f'_{aug_name}'
                args['num_class'] = num_class
                OA,Kappa,AA,net,epoch,train_time,test_time = train_single(train_iter,validation_iter,test_iter,round,args)
                
                if aug_name in ret_dict:
                    ret_dict[aug_name][0].append(OA)
                    ret_dict[aug_name][1].append(Kappa)
                    ret_dict[aug_name][2].append(AA)
                    ret_dict[aug_name][3].append(train_time)
                    ret_dict[aug_name][4].append(test_time)
                else:
                    ret_dict[aug_name] = [[OA],[Kappa],[AA],[train_time],[test_time]]
                
                
                c_map = g_hyperpara.get('c_map',False)
                if c_map:
                    new_hsi_patches = hsi_patches[:,offset:offset+patch_size,offset:offset+patch_size,:]
                    ut.save_classification_map(net,new_hsi_patches,gt,dataset_name,f'{aug_name}_{OA*100:.2f}')
                used_aug_time = time.time()-start_aug_time
                
                if g_hyperpara['eval_epoch_acc']:
                    logging.info(f' {aug_name}, OA: {OA}, time {used_aug_time:.1f} '.center(80,"-"))
                else:
                    logging.info(f' {aug_name}, OA: {OA*100:.2f}, Kappa: {Kappa*100:.2f}, AA: {np.mean(AA)*100:.2f}, time {used_aug_time:.2f}, train: {train_time:.2f},test: {test_time:.2f}'.center(80,"-"))
                
            used_round_time = time.time() - start_round_time
            logging.info(f' Train round {round} ended, used {used_round_time:.1f} sec '.center(80,"="))
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt, exit...')

    log_str = output_statistics(ret_dict, g_hyperpara)
    logging.info(log_str)
    return log_str
    
    
def output_statistics(ret_dict:dict, g_hyperpara):
    ret_dict = ljust_dict_key(ret_dict)
    ret_str = []
    OAs, AAs, Kappas = [],[],[]
    train_times,test_times = [],[]
    out_str = "Class"
    for key,value in ret_dict.items():
        if g_hyperpara['eval_epoch_acc']:
            acc_arr = np.array(value[0])
            mean_OA = np.mean(acc_arr,axis=0)
            ret_str.append(f'{key}: OA {mean_OA}')
        else:
            mean_OA = round(np.mean(value[0])*100,2)
            std_OA = round(np.std(value[0]) *100,2)
            mean_Kappa = round(np.mean(value[1]) *100,2)
            std_Kappa = round(np.std(value[1]) *100,2)
            mean_AA = [round(x,2) for x in (np.mean(value[2],axis=0)*100).tolist()]
            mean_train_time = round(np.mean(value[3]),2)
            mean_test_time = round(np.mean(value[4]),2)
            ret_str.append(f'{key}: OA {mean_OA:.2f}±{std_OA:.2f}, Kappa {mean_Kappa:.2f}±{std_Kappa:.2f}, AA {mean_AA}, train/test time: {mean_train_time:.2f}/{mean_test_time:.2f}')
            OAs.append(f'{mean_OA:.2f}±{std_OA:.2f}')
            AAs.append(mean_AA)
            Kappas.append(f'{mean_Kappa:.2f}±{std_Kappa:.2f}')
            train_times.append(f'{mean_train_time:.2f}')
            test_times.append(f'{mean_test_time:.2f}')
        
        out_str += '\t'+key
    
    out_str += '\n'
    
    for i in range(len(AAs[0])):
        tmp = []
        for j in range(len(ret_dict.keys())):
             tmp.append(f'{AAs[j][i]:.2f}')
        out_str += str(i)+'\t'+'\t'.join(tmp) + '\n'
    
    out_str += f'OA\t' + '\t'.join(OAs) + '\nAA' 
    for i in range(len(AAs)):
        out_str+= f'\t{np.mean(AAs[i]):.2f}±{np.std(AAs[i]):.2f}'
    out_str += f'\nKappa\t' + '\t'.join(Kappas)
    out_str += f'\ntrain\t' + '\t'.join(train_times)
    out_str += f'\ntest\t' + '\t'.join(test_times)
    return out_str


def ljust_dict_key(indict):
    max_len = 0
    for key in indict:
        t_len = len(key)
        if t_len> max_len:
            max_len = t_len
    new_trainings = {}
    for key in indict:
        new_key = key.ljust(max_len)
        new_trainings[new_key] = indict[key]
    return new_trainings


if __name__ == '__main__':
    whole_start = time.time()
    ut.init()
    if platform.system().lower() == 'windows':
        DATASET_FOLDER = 'D:/Datasets/'
        WIN = True
    else:
        DATASET_FOLDER = '../Datasets/'
        WIN = False
        
    dataset_list = ['KSC','PU', 'IP', 'BW', 'PC']

    global_settings = {
        '4_samples_7pc_3emp':settings.get_hyperpara(num_train_data=4,num_pca=7,num_emp=3,train_round=10,c_map=True,amp=False,spectral_var=False),
    }
    
    trainings = settings.compare
    
    SEED = 32668
    logging.info(f'gloabl seed: {SEED}')
    if torch.cuda.is_available():
        cudnn.benchmark = True
    g_device = ut.try_gpu()
    len_settings = len(list(global_settings.keys()))
    results = {}
    for global_setting_key in global_settings:
        
        if len_settings != 1:
            logging.info(f' Global setting {global_setting_key} Started '.center(80,"#"))
        
        eval_epoch_acc = global_settings[global_setting_key].get('eval_epoch_acc',False)
        for key in trainings:
            trainings[key]['eval_epoch_acc'] = eval_epoch_acc
        
        for dataset_name in dataset_list:
            logging.info(f' Dataset {dataset_name} Started '.center(80,"#"))
            start_time = time.time()
            try:
                log_str = train_dataset(dataset_name,global_settings[global_setting_key],trainings)
            except Exception as ex:
                logging.error('Exception: '+ str(ex))
            else:
                if dataset_name in results:
                    results[dataset_name].append(global_setting_key + '\n' + log_str)
                else:
                    results[dataset_name] = [global_setting_key + '\n' + log_str]
            used_time = time.time() - start_time
            logging.info(f' Dataset {dataset_name} Ended, Used {used_time:.1f} seconds '.center(80,"#"))
        
        if len_settings != 1:
            logging.info(f' Global setting {global_setting_key} Ended '.center(80,"#"))
    
    summar = 'Summarization:\n'
    for key, value in results.items():
        summar += key + '\n'
        for line in value:
            summar += line + '\n'
    logging.info(summar)
    whole_end = time.time() - whole_start
    logging.info(f'Took {whole_end:.0f} seconds in total.')
