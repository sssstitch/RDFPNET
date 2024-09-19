import os
import argparse
import torch
import random
import numpy as np

from lib import Transformations, build_dataset, DATA

DATASETS = ['Liver-Disorders' 'Shedden_2008' 'Cervical Cancer' 'Parkinson Dataset' 'Hepatic Encephalopathy']

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='result/RDFPNET/default')
    parser.add_argument("--dataset", type=str, default='Hepatic Encephalopathy')
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=200)
    parser.add_argument("--save", action='store_true', help='whether to save model')
    parser.add_argument("--catenc", action='store_true',
                        help='whether to use catboost encoder for categorical features')
    args = parser.parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    cfg = {
        "model": {
            "prenormalization": True,
            'kv_compression': None,
            'kv_compression_sharing': None,
            'token_bias': True
        },
        "training": {
            "max_epoch": 500,
            "optimizer": "adamw",
        }
    }

    return args, cfg


# 随机数种子
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



device = torch.device('cuda')
args, cfg = get_training_args()
seed_everything(args.seed)

#数据加载
assert args.dataset in DATASETS
T_cache = False
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)
dataset = build_dataset(DATA / args.dataset, transformation, T_cache)

#模型准备
d_out = dataset.n_classes or 1
n_num_features = dataset.n_num_features
cardinalities = dataset.get_category_sizes('train')
n_categories = len(cardinalities)
n_features = n_num_features + n_categories
if args.catenc:
    n_categories = 0
cardinalities = None if n_categories == 0 else cardinalities

# 模型超参数
kwargs = {
    'd_numerical': n_num_features,
    'd_out': d_out,
    'categories': cardinalities,
    **cfg['model']
}
default_model_configs = {
    'ffn_dropout': 0., 'attention_dropout': 0.3, 'residual_dropout': 0.,
    'n_layers': 2, 'n_heads': 32, 'd_token': 256,
    'init_scale': 0.01,
}
default_training_configs = {
    'lr': 1e-1*0.5,
    'weight_decay': 0.,
}

# 更新模型参数
kwargs.update(default_model_configs)
cfg['training'].update(default_training_configs)


# batch size
batch_size = 128
val_batch_size = 64




