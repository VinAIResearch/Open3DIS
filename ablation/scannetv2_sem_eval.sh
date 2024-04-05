CUDA_VISIBLE_DEVICES=0 python scannetv2_sem_eval.py \
--exp version1_5x2 \
--feature_stage computed_feature1 \
--predsem_path ../../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval/version1 \
--gtsem_path ../../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval/pcl \
--positive_thresh 0.6