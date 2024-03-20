python3.8 train.py \
./configs/ssd300_vgg16_pascal-voc-07-12.json \
./pad_project/images \
./pad_project/labels \
--training_split=./pad_project/train.txt \
--validation_split=./pad_project/val.txt \
--label_maps=./pad_project/label_maps.txt \
--learning_rate=0.001 \
--epochs=50 \
--batch_size=16 \
--shuffle=True \
--augment=False \
--output_dir=./pad_model_ssd
