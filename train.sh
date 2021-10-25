





# Uformer
nohup python3 ./train.py --arch Uformer --batch_size 16 --gpu '0' \
    --train_ps 128 --train_dir ../datasets/LOL_patches --env Uformer_1_128\
    --val_dir ../datasets/LOL/valid --embed_dim 32 --warmup --nepoch=500 &

    
# UNet
# python3 ./train.py --arch UNet --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

#stage0
nohup python3 ./train_0.py --arch Uformer --batch_size 16 --gpu '0' \
    --train_ps 256 --train_dir /home/mist/lowlight/datasets/lol_R/train --env 32_1023_4_0 \
    --val_dir /home/mist/lowlight/datasets/lol_R/valid --embed_dim 32 --warmup --nepoch=500  &


# #stage1
# nohup python3 ./train_1.py --arch Uformer --batch_size 3 --gpu '0' \
#     --train_ps 256 --train_dir /home/mist/lowlight/datasets/lol/train --env 32_1022_3_1 \
#     --val_dir /home/mist/lowlight/datasets/lol/valid --embed_dim 32 --warmup --nepoch=500 &

# #stage1_resume
# nohup python3 ./train_1_resume.py --arch Uformer --batch_size 16 --gpu '0' \
#     --train_ps 64 --train_dir /home/mist/lowlight/datasets/lol_stage1/train --env 32_1006_1_2 \
#     --val_dir /home/mist/lowlight/datasets/lol_stage1/valid --embed_dim 32  --nepoch 150  --lr_initial 0.000001 &
