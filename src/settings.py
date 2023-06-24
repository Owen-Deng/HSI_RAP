from utils import try_gpu

g_device = try_gpu()
def get_hyperpara(num_train_data = 4,num_valid_data = 0,num_pca=7,
                  num_emp = 3,train_round = 8,test_batch_size ={'ksc':256,'paviau':128},
                  default_test_batch_size = 128,g_device = g_device,g_del_other = False,eval_epoch_acc= False,disjoint=False,seed = 0,amp=False,c_map=False,
                  spectral_var = False):
    return {
        'num_train_data':num_train_data,
        'num_valid_data' : num_valid_data,
        'num_pca':num_pca,
        'num_emp':num_emp,
        'train_round':train_round,
        'test_batch_size':test_batch_size,
        'default_test_batch_size':default_test_batch_size,
        'device': g_device,
        'g_del_other': g_del_other,
        'eval_epoch_acc':eval_epoch_acc,
        'disjoint':disjoint,
        'seed':seed,
        'amp':amp,
        'c_map':c_map,
        'spectral_var' :spectral_var
    }

compare = {
    # 'M3D-DCNN':{
    #     'started_lr':0.01,
    #     'fixed_lr':True,
    #     'num_epoch' :1300,
    #     'batch_size':40,
    #     'device':g_device,
    #     'patch_size':7,
    #     'optimizer':'Adagrad',
    #     'optimizer_weight_decay':0.01,
    #     'model': 'HeEtAl',
    #     'test_during_training':False,
    #     'aug_func':'augment_tradition'
    # },
    # '3-D CNN':{
    #     'started_lr':0.01,
    #     'fixed_lr':True,
    #     'num_epoch' :1300,
    #     'batch_size':100,
    #     'device':g_device,
    #     'patch_size':5,
    #     'optimizer':'sgd',
    #     'optimizer_weight_decay':0.0005,
    #     'model': 'HamidaEtAl',
    #     'test_during_training':False,
    #     'aug_func':'augment_tradition'
    # },
    # 'HSI-CNN':{
    #     'started_lr':0.1,
    #     'fixed_lr':True,
    #     'num_epoch' :1300,
    #     'batch_size':100,
    #     'device':g_device,
    #     'patch_size':3,
    #     'optimizer':'SGD',
    #     'optimizer_weight_decay':0.009,
    #     'model': 'LuoEtAl',
    #     'test_during_training':False,
    #     'aug_func':'augment_tradition'
    # },
    # 'NoAug':
    #     {'dropout':0.5,
    #     'started_lr':0.0003,
    #     'num_epoch' :1300,
    #     'batch_size':1024,
    #     'device':g_device,
    #     'model':'HSI_CR',
    #     'patch_size':13,
    #     'test_during_training':False,
    #     },
    # 'RandAugment':
    #     {'dropout':0.5,
    #     'started_lr':0.0003,
    #     'num_epoch' :1300,
    #     'batch_size':1024,
    #     'device':g_device,
    #     'model':'HSI_CR',
    #     'aug_name': 'RandAugment',
    #     'aug_prob':{},
    #     'aug_args':{},
    #     'patch_size':13,
    #     'test_during_training':False,
    #     },
    # 'ADA':
    #     {'dropout':0.5,
    #     'started_lr':0.0003,
    #     'num_epoch' :1300,
    #     'batch_size':1024,
    #     'device':g_device,
    #     'model':'HSI_CR',
    #     'aug_name': 'RandomAugmentPipeADA',
    #     'aug_prob':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1,'rotate90':1,'noise':1},
    #     'aug_args':{},
    #     'patch_size':13,
    #     'test_during_training':False,
    #     },
    #  'RAP':
    #     {'dropout':0.5,
    #     'started_lr':0.0003,
    #     'num_epoch' :1300,
    #     'batch_size':1024,
    #     'device':g_device,
    #     'model':'HSI_CR',
    #     'aug_name': 'RandomAugmentPipe',
    #     'aug_prob':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1},
    #     'aug_args':{'shear_max':0.2,'xint_max':0.1,'scale_std':0.1,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.3},
    #     'patch_size':13,
    #     'test_during_training':False,
    #     },       
    'Extended-RAP':
        {'dropout':0.5,
        'started_lr':0.0003,
        'num_epoch' :1300,
        'batch_size':1024,
        'device':g_device,
        'model':'HSI_CR',
        'aug_name': 'RandomAugmentPipe',
        'aug_prob':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1,'yflip':1,'transpose':1,'shearx':1,'sheary':1,'srotate':1},
        'aug_args':{'shear_max':0.2,'xint_max':0.1,'scale_std':0.1,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.3},
        'patch_size':13,
        'test_during_training':False,
        }
}




best_and_default_aug_paras = {
    'best_aug_para':
        {'dropout':0.5,
        'started_lr':0.0003,
        'num_epoch' :1300,
        'batch_size':256,
        'device':g_device,
        'aug_prob':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1,'yflip':1,'transpose':1,'shearx':1,'sheary':1,'srotate':1},
        'aug_args':{'shear_max':0.2,'xint_max':0.1,'scale_std':0.1,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.3},
        'auto_aug':1,
        'patch_size':13},
    
    'default_aug_para':
        {'dropout':0.5,
        'started_lr':0.0003,
        'num_epoch' :1300,
        'batch_size':256,
        'device':g_device,
        'aug_prob':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1,'yflip':1,'transpose':1,'shearx':1,'sheary':1,'srotate':1},
        'aug_args':{},
        'auto_aug':1,
        'patch_size':13
        }
}