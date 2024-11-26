get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

TEST_PATH="/data1/baihy/datasets/SJKdata/luoshedaqiao_rectified"

COARSE_CONFIG="sjk_luoshe_coarse"
CONFIG="sjk_luoshe_refine"

max_block_id=24  # i.e. x_dim * y_dim * z_dim - 1
port=4041


# train coarse global gaussian model
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python train_large.py --config config/$COARSE_CONFIG.yaml

# obtain data partitioning
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python data_partition.py \
#     --config config/$CONFIG.yaml > part_log1.txt
