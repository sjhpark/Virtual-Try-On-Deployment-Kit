python train_PFAFN_e2e.py --name PFAFN_e2e   \
--PFAFN_warp_checkpoint 'checkpoints/PFAFN_stage1/PFAFN_warp_epoch_201.pth'  \
--PBAFN_warp_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_gen_epoch_101.pth'  \
--resize_or_crop None --verbose --tf_log --batchSize 1 --num_gpus 1 --label_nc 14 --launcher pytorch










