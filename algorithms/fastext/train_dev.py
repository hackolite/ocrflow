from __future__ import  absolute_import
import os
import sys, traceback
#import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

matplotlib.use('agg')




def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    #if opt.load_path:
    trainer.load("./fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth")
    print('load pretrained model from %s' % opt.load_path)

    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
            trainer.reset_meters()
            for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
                try:
                    scale = at.scalar(scale)
                    img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                    trainer.train_step(img, bbox, label, scale)
                    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                    gt_img = visdom_bbox(ori_img_,at.tonumpy(bbox_[0]),at.tonumpy(label_[0]))

                    #plot predicti bboxes
                    #_bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                    #pred_img = visdom_bbox(ori_img_, at.tonumpy(_bboxes[0]), at.tonumpy(_labels[0]).reshape(-1), at.tonumpy(_scores[0]))

                except Exception as e:
                    print("error")
                    traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
    train()