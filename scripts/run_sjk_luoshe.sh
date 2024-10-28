get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

TEST_PATH="data/luoshedaqiao_forward"

COARSE_CONFIG="sjk_luoshe_coarse"
CONFIG="sjk_luoshe_refine"

max_block_id=24  # i.e. x_dim * y_dim * z_dim - 1
port=4041


# train coarse global gaussian model
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python train_large.py --config config/$COARSE_CONFIG.yaml


# train CityGaussian

# obtain data partitioning
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python data_partition.py \
#     --config config/$CONFIG.yaml > part_log1.txt



# optimize each block, please adjust block number according to config

# left_lists="1 2 3 4 5"
# for num in $left_lists; do


# for num in $(seq 0 $max_block_id); do
#     while true; do
#         gpu_id=$(get_available_gpu)
#         if [[ -n $gpu_id ]]; then
#             echo "GPU $gpu_id is available. Starting training block '$num'"
#             CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $num --port $port &
#             # Increment the port number for the next run
#             ((port++))
#             # Allow some time for the process to initialize and potentially use GPU memory
#             sleep 30
#             break
#         else
#             echo "No GPU available at the moment. Retrying in 2 minute."
#             sleep 60
#         fi
#     done
# done

# wait


# merge the blocks
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python merge.py \
#     --config config/$CONFIG.yaml > merge_log.txt



# rendering and evaluation, add --load_vq in rendering if you want to load compressed model
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python render_large.py \
#   --config config/$CONFIG.yaml \
#   --custom_test $TEST_PATH > render_log.txt



# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python metrics_large.py -m output/$CONFIG -t "luoshedaqiao_forward"


gpu_id=2
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python render_video.py -m output/$CONFIG



