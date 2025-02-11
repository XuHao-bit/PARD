is_GNN=True
# is_GNN=False
# dataset='ml-100k'
# dataset='ali-ads'
dataset='ml-1m'
# pre_name='APDF-FedNCF-pretrain15-ml100k-v2.pkl'
# pre_name='APDF-FedNCF-pretrain15-ml100k.pkl+lr0.5+eta80'
# pre_name='apdf-pre2-titan.pkl'
pre_name='APDF-FedGNN-pretran40-ml1m.pkl+lr0.1+eta80.0'
# pre_name='APDF-FedNCF-pretrain13-ml1m.pkl+lr0.5+eta80'
ratio=1.
is_esti_local=True
device='0'

# python run_experiment.py --clients_sample_ratio $ratio --GNN $is_GNN --num_round 40 --NAME $pre_name --dataset $dataset --earlystop 15 --pretrain False --finetune True

python run_grid_search.py --GNN $is_GNN --num_round 15 --NAME $pre_name --dataset $dataset --earlystop 20 --pretrain False --finetune True --is_esti_local $is_esti_local --device_id $device