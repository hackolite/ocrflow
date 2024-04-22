import cv2
import os
import json
import argparse
import numpy as np
from glob import glob
from utils import inference_utils

parser = argparse.ArgumentParser(
    description='run inference on an input image.')
parser.add_argument('images', type=str,
                    help='glob string for list of images.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('weights', type=str, help='path to the weight file.')
parser.add_argument('--label_maps', type=str, help='path to label maps file.')
parser.add_argument('--confidence_threshold', type=float,
                    help='the confidence score a detection should match in order to be counted.', default=0.01)
parser.add_argument('--num_predictions', type=int,
                    help='the number of detections to be output as final detections', default=10)
args = parser.parse_args()

# assert os.path.exists(args.input_image), "config file does not exist"
assert os.path.exists(args.config), "config file does not exist"
assert args.num_predictions > 0, "num_predictions must be larger than zero"
assert args.confidence_threshold > 0, "confidence_threshold must be larger than zero."
assert args.confidence_threshold <= 1, "confidence_threshold must be smaller than or equal to 1."
with open(args.config, "r") as config_file:
    config = json.load(config_file)

input_size = config["model"]["input_size"]
model_config = config["model"]

if model_config["name"] == "ssd_vgg16":
    model, process_input_fn, label_maps = inference_utils.ssd_vgg16(config, args)
elif model_config["name"] == "ssd_mobilenetv1":
    model, process_input_fn, label_maps = inference_utils.ssd_mobilenetv1(config, args)
elif model_config["name"] == "ssd_mobilenetv2":
    model, process_input_fn, label_maps = inference_utils.ssd_mobilenetv2(config, args)
elif model_config["name"] == "tbpp_vgg16":
    model, process_input_fn, label_maps = inference_utils.tbpp_vgg16(config, args)
else:
    print("has not been implemented yet")
    exit()

model.load_weights(args.weights)


for idx, input_image in enumerate(list(glob(args.images))):
    image = cv2.imread(input_image)  # read image in bgr format
    # image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    image = np.array(image, dtype=np.float)
    image = np.uint8(image)

    display_image = image.copy()
    image_height, image_width, _ = image.shape
    height_scale, width_scale = input_size/image_height, input_size/image_width

    image = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = process_input_fn(image)

    image = np.expand_dims(image, axis=0)
    y_pred = model.predict(image)

    for i, pred in enumerate(y_pred[0]):
        classname = label_maps[int(pred[0]) - 1].upper()
        confidence_score = pred[1]

        score = confidence_score * 100
        print(classname, score)

        if  1==1:
            xmin = max(int(pred[2] / width_scale), 1)
            ymin = max(int(pred[3] / height_scale), 1)
            xmax = min(int(pred[4] / width_scale), image_width-1)
            ymax = min(int(pred[5] / height_scale), image_height-1)

            cv2.putText(
                display_image,
                classname,
                (int(xmin), int(ymin)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (100, 100, 255),
                1,
                2
            )

            cv2.rectangle(
                display_image,
                (xmin, ymin),
                (xmax, ymax),
                (255, 0, 0),
                2
            )

    print("\n")

    cv2.imwrite(str(idx)+"_output.jpg", display_image)
