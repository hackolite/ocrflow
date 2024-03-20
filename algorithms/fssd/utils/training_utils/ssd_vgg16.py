import os
from losses import SSD_LOSS
from utils import data_utils
from networks import SSD_VGG16
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam

from data_generators import SSD_DATA_GENERATOR
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, LearningRateScheduler
from tensorflow.keras.applications.vgg16 import preprocess_input


def ssd_vgg16(config, args, callbacks):
    training_config = config["training"]
    with open(args.label_maps, "r") as label_map_file:
        label_maps = [i.strip("\n") for i in label_map_file.readlines()]
        #label_maps = ["text"] 
    
    training_samples = data_utils.get_samples_from_split(
        split_file=args.training_split,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )

    if args.validation_split is not None:
        validation_samples = data_utils.get_samples_from_split(
            split_file=args.validation_split,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir
        )

    training_data_generator = SSD_DATA_GENERATOR(
        samples=training_samples,
        config=config,
        label_maps=label_maps,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=preprocess_input
    )

    if args.validation_split is not None:
        print("-- validation split specified")
        validation_data_generator = SSD_DATA_GENERATOR(
            samples=validation_samples,
            config=config,
            label_maps=label_maps,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            augment=False,
            process_input_fn=preprocess_input
        )

    loss = SSD_LOSS(
        alpha=training_config["alpha"],
        min_negative_boxes=training_config["min_negative_boxes"],
        negative_boxes_ratio=training_config["negative_boxes_ratio"]
    )

    if training_config["optimizer"]["name"] == "adam":
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=args.learning_rate,
            beta_1=training_config["optimizer"]["beta_1"],
            beta_2=training_config["optimizer"]["beta_2"],
            epsilon=training_config["optimizer"]["epsilon"],
            decay=training_config["optimizer"]["decay"]
        )
    elif training_config["optimizer"]["name"] == "sgd":
        optimizer = SGD(
            learning_rate=args.learning_rate,
            momentum=training_config["optimizer"]["momentum"],
            decay=training_config["optimizer"]["decay"],
            nesterov=training_config["optimizer"]["nesterov"]
        )
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=args.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=0.0
        )

    model = SSD_VGG16(
        config=config,
        label_maps=label_maps,
        is_training=True
    )

    if args.show_network_structure:
        model.summary()

    model.compile(
        optimizer=optimizer,
        loss=loss.compute
    )

    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint), "checkpoint does not exist"
        model.load_weights(args.checkpoint, by_name=True)

    model.fit(
        x=training_data_generator,
        validation_data=validation_data_generator if args.validation_split is not None else None,
        batch_size=args.batch_size,
        validation_batch_size=args.batch_size,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        callbacks=callbacks,
    )

    model.save_weights(os.path.join(args.output_dir, "model.h5"))
