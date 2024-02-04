python train.py     --epochs 35     --lr 5e-4     --batch_size 256     --hidden1 64     --hidden2 16     --hidden_decode1 512     --network_type DTI     --data_path '../data/DTI/fold1'     --input_type one_hot   

python train.py     --epochs 15     --lr 5e-4     --batch_size 256     --hidden1 64     --hidden2 16     --hidden_decode1 512     --network_type DDI     --data_path '../data/DDI/fold1'    --input_type one_hot   --dropout=0.5

python train.py     --epochs 15     --lr 5e-4     --batch_size 256     --hidden1 64     --hidden2 16     --hidden_decode1 512     --network_type PPI     --data_path '../data/PPI/fold1'     --input_type one_hot

python train.py \
    --epochs 5 \
    --lr 5e-4 \
    --batch_size 256 \
    --hidden1 64 \
    --hidden2 16 \
    --hidden_decode1 512 \
    --network_type GDI \
    --data_path '../data/GDI/fold1'\
    --input_type one_hot   
    --dropout 0.5


