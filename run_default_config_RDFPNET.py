import os
import time
import json
import numpy as np
import warnings



import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder

from model import RDFPNET
from lib import prepare_tensors, make_optimizer
from default_config import dataset, device, cfg, args, batch_size, val_batch_size, kwargs


#plt.rcParams['font.sans-serif'] = ['SimSun']
warnings.filterwarnings("ignore")




def record_exp(args, final_score, best_score, **kwargs):
    results = {'config': args, 'final': final_score, 'best': best_score, **kwargs}
    with open(f"{args['output']}/results.json", 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}
# 分类特征变为数值特征
if args.catenc and dataset.X_cat is not None:
    cardinalities = dataset.get_category_sizes('train')
    enc = CatBoostEncoder(
        cols=list(range(len(cardinalities))),
        return_df=False
    ).fit(dataset.X_cat['train'], dataset.y['train'])
    for k in ['train', 'val', 'test']:
        dataset.X_num[k] = np.concatenate([enc.transform(dataset.X_cat[k]).astype(np.float32), dataset.X_num[k]],
                                          axis=1)


X_num, X_cat, ys = prepare_tensors(dataset, device=device)

if args.catenc:
    X_cat = None

#特征排序
mif_cache_dir = 'cache/mif'
if not os.path.isdir(mif_cache_dir):
    os.makedirs(mif_cache_dir)
mif_cache_file = f'{mif_cache_dir}/{args.dataset}.npy'
if os.path.exists(mif_cache_file):
    mi_scores = np.load(mif_cache_file)
else:
    mi_func = mutual_info_regression if dataset.is_regression else mutual_info_classif
    mi_scores = mi_func(dataset.X_num['train'], dataset.y['train'])
    np.save(mif_cache_file, mi_scores)
mi_ranks = np.argsort(-mi_scores)
X_num = {k: v[:, mi_ranks] for k, v in X_num.items()}
sorted_mi_scores = torch.from_numpy(mi_scores[mi_ranks] / mi_scores.sum()).float().to(device)



# 可视化互信息分数 "----------------------------------------------------------------------------------------"
# plt.figure(figsize=(12, 6))
# bars = plt.bar(range(len(sorted_mi_scores)), sorted_mi_scores.cpu().numpy(), edgecolor='black', linewidth=1)  # 设置边框颜色为黑色
# #plt.title('Feature Importance Using Mutual Information')
# plt.xlabel('排序特征索引',fontsize=45)
# plt.ylabel('归一化信息值',fontsize=45)
# plt.grid(True, linewidth=0.3)

# plt.xticks(fontsize=50,fontproperties='Times New Roman')
# plt.yticks(fontsize=50,fontproperties='Times New Roman')
# plt.tick_params(axis='x', direction='in', width=1)
# plt.tick_params(axis='y', direction='in', width=1)
# plt.gca().set_axisbelow(True)

# output_path = 'D:/Desktop/feature_importance.png'
# plt.savefig(output_path, dpi=1000, bbox_inches='tight')
# plt.show()

# 显示排序后的特征索引和对应的互信息值
# print('Sorted Feature Indices:')
# print(mi_ranks)
# print('Sorted Mutual Information Values:')
# print(mi_scores[mi_ranks])




# 更新训练参数
cfg['training'].update({
    "batch_size": batch_size,
    "eval_batch_size": val_batch_size,
    "patience": args.early_stop
})

# 数据加载
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}



# 模型
model = RDFPNET(**kwargs).to(device)

# 优化器
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
optimizer = make_optimizer(
    cfg['training']['optimizer'],
    (
        [
            {'params': parameters_with_wd},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
    ),
    cfg['training']['lr'],
    cfg['training']['weight_decay'],
)


if torch.cuda.device_count() > 1:
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)

# 损失函数
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)


def apply_model(x_num, x_cat=None):
    return model(x_num, x_cat)

@torch.inference_mode()
def evaluate(parts):
    model.eval()
    predictions = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        infer_time = 0.
        predictions[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            start = time.time()
            predictions[part].append(apply_model(x_num, x_cat))
            infer_time += time.time() - start
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
        if part == 'test':
            print('test time: ', infer_time)
    prediction_type = None if dataset.is_regression else 'logits'
    return dataset.calculate_metrics(predictions, prediction_type)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



#训练
metric = 'roc_auc' if dataset.is_binclass else 'score'
init_score = evaluate(['test'])['test'][metric]
print(f'Test score before training: {init_score: .4f}')

losses, val_metric, test_metric = [], [], []
n_epochs = 500
warm_up = 10
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs - warm_up)
max_lr = cfg['training']['lr']
report_frequency = len(ys['train']) // batch_size // 1


loss_holder = AverageMeter()
best_score = -np.inf
final_test_score = -np.inf
best_test_score = -np.inf
running_time = 0.


no_improvement = 0
EARLY_STOP = args.early_stop

for epoch in range(1, n_epochs + 1):
    model.train()
    if warm_up > 0 and epoch <= warm_up:
        lr = max_lr * epoch / warm_up
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()
    for iteration, batch in enumerate(train_loader):
        x_num, x_cat, y = (
            (batch[0], None, batch[1])
            if len(batch) == 2
            else batch
        )

        start = time.time()
        optimizer.zero_grad()
        loss = loss_fn(apply_model(x_num, x_cat), y.squeeze())
        loss.backward()
        optimizer.step()
        running_time += time.time() - start
        loss_holder.update(loss.item(), len(ys))
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss_holder.val:.4f} (avg_loss) {loss_holder.avg:.4f}')
    losses.append(loss_holder.avg)
    loss_holder.reset()
    scores = evaluate(['val', 'test'])
    val_score, test_score = scores['val'][metric], scores['test'][metric]
    val_metric.append(val_score), test_metric.append(test_score)
    print(f'Epoch {epoch:03d} | Test score: {test_score:.4f} | ACC: {scores["test"]["accuracy"]:.4f}  F1-Score: {scores["test"]["weighted avg"]["f1-score"]:.4f}  ROC AUC: {scores["test"]["roc_auc"]:.4f}  Recall: {scores["test"]["weighted avg"]["recall"]:.4f} | Support: {scores["test"]["weighted avg"]["support"]} | Score: {scores["test"]["score"]:.4f}',end='')
    #print(f'Epoch {epoch:03d} ||| RMSE: {scores["test"]["rmse"]:.4f}',end='')


    if val_score > best_score:
        best_score = val_score
        final_test_score = test_score
        print(' <<< BEST VALIDATION EPOCH')
        no_improvement = 0
        if args.save:
            torch.save(model.state_dict(), f"{args.output}/pytorch_model.pt")
    else:
        no_improvement += 1
    if test_score > best_test_score:
        best_test_score = test_score

    if no_improvement == EARLY_STOP:
        break
        

record_exp(
    vars(args), final_test_score, best_test_score,
    losses=str(losses), val_score=str(val_metric), test_score=str(test_metric),
    cfg=cfg, time=running_time,
)
