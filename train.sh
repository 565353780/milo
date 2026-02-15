DATA_FOLDER="${HOME}/chLi/Dataset/GS/haizei_1_v4"

ITERATIONS=30000

cd milo

CUDA_VISIBLE_DEVICES=3 \
  python train.py \
  -s ${DATA_FOLDER}/colmap_normalized/ \
  -m ${DATA_FOLDER}/milo/ \
  --images masked_images \
  -r 1 \
  --imp_metric "indoor" \
  --rasterizer "radegs" \
  --white_background \
  --log_interval 200 \
  --mesh_config "default"

CUDA_VISIBLE_DEVICES=3 \
  python mesh_extract_sdf.py \
  -s ${DATA_FOLDER}/colmap_normalized/ \
  -m ${DATA_FOLDER}/milo/ \
  --images masked_images \
  -r 1 \
  --rasterizer "radegs"
