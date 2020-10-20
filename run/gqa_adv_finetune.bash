# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=/storage/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa.py \
    --train train,valid --valid testdev \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA /pretrain/model \
    --batchSize 32 --optim bert --lr 1e-5 --epochs 4 \
    --tqdm --output $output ${@:3} \
    --adv_training \
    --adv_modality ["text"] \
    --adv_lr_txt 1e-2 \
    --adv_lr_img 1e-2 \
    --adv_steps 3 \
    --adv_init_mag 0 \
    --norm_type l2 \
    --adv_max_norm 0 \
    --adv_kl_weight 1.5 
