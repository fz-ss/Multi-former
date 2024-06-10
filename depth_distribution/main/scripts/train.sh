# python -m torch.distributed.launch --nproc_per_node=2 --master_port='29501' train.py 

python -m torch.distributed.launch --nproc_per_node=2 train.py  #--nproc_per_node:并行的GPU数量
