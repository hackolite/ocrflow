python3.8 inference.py \
"./imgs/*" \
./configs/ssd300_vgg16_pascal-voc-07-12.json  \
model.h5 \
--label_maps=./pad_project/label_maps.txt \
--confidence_threshold=0.9 \
--num_predictions=10
