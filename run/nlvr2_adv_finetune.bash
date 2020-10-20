# The name of this experiment.
name=$2

# Save logs and models under snap/nlvr2; Make backup.
output=/storage/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See run/Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/nlvr2.py \
    --train train --valid valid \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT /pretrain/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output ${@:3} \
    --adv_training \
    --adv_modality ["text"] \
    --adv_lr_txt 1e-3 \
    --adv_lr_img 1e-3 \
    --adv_steps 3 \
    --adv_init_mag 0 \
    --norm_type l2 \
    --adv_max_norm 0 \
    --adv_kl_weight 1.5 

