import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import lmdb
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
import os
import sys
import pickle
import numpy as np

import glob

glob.glob('/nfs/users/ext_ankan.bhunia/Handwritten_data/CVL/cvl-database-1-1/*/words/*/*tif')