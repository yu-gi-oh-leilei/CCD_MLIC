python -m torch.distributed.launch --nproc_per_node=2 main_mlc.py  --world-size 1 --rank 0 \
-a 'CCD-R101-448' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/ResNet101_448_COCO/bs128worktest" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp \
--ema-decay 0.9997 \
--gpus 0,1


python -m torch.distributed.launch --nproc_per_node=2 main_mlc.py  --world-size 1 --rank 0 \
-a 'CCD-R101-448' \
--dataset_dir '/media/data2/MLICdataset/VOC2007/' \
--backbone resnet101 --dataname voc2007 --batch-size 128 --print-freq 400 \
--output "./output/ResNet101_448_VOC2007/bs128worktest" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 20 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 0 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1

# COCO2014
python -m torch.distributed.launch --nproc_per_node=4 main_mlc.py  --world-size 1 --rank 0 \
-a 'CCD-R101-448' \
--dataset_dir '/media/data1/maleilei/dataset/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128worktest" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 5 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1


# 
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/data1/maleilei/dataset/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128worktest" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 5 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1,2,3




torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/mlldiskSSD/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128work6" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 --loss focal \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 5 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1,2,3


# debug
python -m torch.distributed.launch --nproc_per_node=4

torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/mlldiskSSD/VOC2007/' \
--backbone resnet101 --dataname voc2007 --batch-size 32 --print-freq 400 \
--output "./output/ResNet101_448_VOC2007/bs128worktest" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 --loss lsloss \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 20 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 0 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1




torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/mlldiskSSD/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128work12" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 --loss lsloss \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-4 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 7 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1,2,3



torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/mlldiskSSD/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128work10" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 --loss lsloss \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-4 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 5 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1,2,3


torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128work_q2l" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 --loss lsloss \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-4 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 0 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1,2,3


torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc_resume.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128work_q2l3" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 --loss lsloss \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-4 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 5 \
--ema-decay 0.9997 \
-sybn \
--resume '/media/data2/maleilei/CCD__final/module_model_highest.pth.tar' \
--gpus 0,1,2,3


# 
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128work_onlystage1" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 --loss lsloss \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-4 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 80 \
--ema-decay 0.9997 \
-sybn \
--gpus 0,1,2,3


# 
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py \
-a 'CCD-R101-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset/COCO2014' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output "./output/CCD_ResNet101_448_COCO/bs128_concat_select" \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 1 --gamma_neg 1 --dtgfl --loss_clip 0 --loss lsloss \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-4 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 2048 --feat_dim 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp --stop_epoch 7 \
--ema-decay 0.9997 \
--feat_fuse concat \
-sybn \
--gpus 0,1,2,3