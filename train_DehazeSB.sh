### /data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_simple
### /data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_new_v2_delete_bluesky

python train.py \
--dataroot /data/lanyunwei/my_experiment_results/datasey \
--mode dehazesb --model dehazesb \
--dataset_mode unpaired \
--batch_size 4 \
--checkpoints_dir /output/dehaze_SB_ckpt/ \
--netG DCSNet_unetgan \
--prompt_pretrain_dir /code/pretrained/best_prompt_round.pth

