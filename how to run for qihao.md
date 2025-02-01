1 先自己跑跑看看代码是啥样子，模型结构是啥样子；
2 看看论文里面，比如ml-100k的结果 '~/log/final_res'，先看看这个log文件在哪里，然后看看log文件里面的超参数是啥，自己改成这个超参数，看看能不能跑出结果出来；
    - 自己的log在：~/log/ 下面；

### 实验结果：
https://eve0wcbe0c.feishu.cn/sheets/LYZZsuDXqhewzAtYQ6KcbOI5n9d

@ qihao
### Run APDF
NOTE：apdf不需要专门eval_privacy，因为run的时候就自动eval了；
#### 思路1: 直接end2end训练
- pretrain=True, finetune=False；lam_eu=0.5，lam_pu=0.5；
- 虽然叫pretrain，但其实是，先训练3epoch推荐模型，将emb warm up（训练好emb）；3epoch之后，同时进行estimator和推荐模型的对抗训练；
- 目前这个方式需要在每个client训练estimator，太慢了，试试优化fed_train_estimator
```
conda activate torchcu090
cd /home/zhaoxuhao/FedRec/APDF
python run_grid_search.py --GNN [False] or [True] --num_round 200 --dataset ml-1m --earlystop 10 --pretrain True --finetune False # GNN=False for applying apdf to fedncf; GNN=true for applying it to fedgnn;
```
or
```
tmux attach -t 0
python run_grid_search.py --GNN [False] or [True] --num_round 200 --dataset ml-1m --earlystop 10 --pretrain True --finetune False
```

#### 思路2: 先pretrain再finetune
- 这样的好处是，
    - 先pretrain让模型到一个较好的推荐结果左右；再利用estimator finetune的时候，推荐效果不会太差；
    - 因为对抗训练很花时间（因为目前是所有的client都要先train estimator，这个太慢了），所以先pretrain的话，训练速度会快很多；
- end2end训练的坏处就是：estimator对抗训练的时候，有可能estimator的privacy_loss太容易优化，导致optimizer去训练pri_loss，而不训练rec_loss，使推荐模型陷入sub_optimal；
- pretrain的时候，用run_experiment.py，将lam_eu和pu设置为0；训练到最优性能的80%左右即可；
- finetune的时候，差不多50个epoch就够了？
```
conda activate torchcu090
cd /home/zhaoxuhao/FedRec/APDF
python run_experiment.py --GNN [False] or [True] --num_round xx --dataset ml-1m --earlystop 10 --pretrain True --finetune False --save_model True --save_name xxx.pkl
python run_experiment.py --GNN [False] or [True] --num_round yy --dataset ml-1m --earlystop 10 --pretrain False --finetune True
```
for example,
python run_experiment.py --num_round 45 --save_model True --save_name APDF-pretran-ml1m.pkl --device_id 1 --dataset ml-1m --GNN False --earlystop 10 --pretrain True --finetune False

@ xuhao
### train fedncf
```
cd ~/FedRec/FedNCF
python run_grid_search.py --GNN False --num_round 200 --dataset ml-1m --earlystop 10 --save_model True --save_name FedNCF
```

### train fedgnn
```
python run_grid_search.py --GNN True --num_round 150 --dataset ml-1m --earlystop 10 --save_model True --save_name FedGNN
```


@ xuhao
### eval privacy for fedncf, fedgnn, ppfedncf, ppfedgnn ...
```
cd /home/zhaoxuhao/FedRec/FedNCF
# 根据eval_privacy.py里面的todos修改文件名
python eval_privacy.py --dataset ml-1m --attack_mode u_emb+i_emb >> log/ppattack_finalres/ml1m-xxx.txt # xxx是模型名
```