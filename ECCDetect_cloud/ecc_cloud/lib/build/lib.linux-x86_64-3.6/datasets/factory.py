# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
#from datasets.coco import coco
from datasets.imagenet_vid import imagenet_detect
#from datasets.vg import vg

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


        
# set up image net vid.
for split in ['train', 'val', 'test']:
    name = 'imagenet_vid_{}'.format(split)
    devkit_path =  "/home/wgq/rfcn/ILSVRC/"
    __sets[name] = (lambda split=split, devkit_path=devkit_path: imagenet_detect(split,devkit_path, 'VID'))

# set up image net det.
for split in ['train', 'val', 'test']:
    name = 'imagenet_det_{}'.format(split)
    devkit_path = "/home/wgq/IL/ILSVRC2015/"
    #data_path = os.path.join('data', 'ILSVRC')
    __sets[name] = (lambda split=split, devkit_path=devkit_path: imagenet_detect(split,devkit_path, 'DET'))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
