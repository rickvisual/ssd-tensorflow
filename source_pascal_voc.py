#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   30.08.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import lxml.etree
import random
import math
import cv2
import os

import numpy as np

from utils import Label, Box, Sample, Size
from utils import rgb2bgr, abs2prop
from glob import glob
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Labels
#-------------------------------------------------------------------------------
label_defs = [
    # Label('aeroplane',   rgb2bgr((0,     0,   0))),
    # Label('bicycle',     rgb2bgr((111,  74,   0))),
    # Label('bird',        rgb2bgr(( 81,   0,  81))),
    # Label('boat',        rgb2bgr((128,  64, 128))),
    # Label('bottle',      rgb2bgr((244,  35, 232))),
    # Label('bus',         rgb2bgr((230, 150, 140))),
    # Label('car',         rgb2bgr(( 70,  70,  70))),
    # Label('cat',         rgb2bgr((102, 102, 156))),
    # Label('chair',       rgb2bgr((190, 153, 153))),
    # Label('cow',         rgb2bgr((150, 120,  90))),
    # Label('diningtable', rgb2bgr((153, 153, 153))),
    # Label('dog',         rgb2bgr((250, 170,  30))),
    # Label('horse',       rgb2bgr((220, 220,   0))),
    # Label('motorbike',   rgb2bgr((107, 142,  35))),
    # Label('person',      rgb2bgr(( 52, 151,  52)))
    # Label('pottedplant', rgb2bgr(( 70, 130, 180))),
    # Label('sheep',       rgb2bgr((220,  20,  60))),
    # Label('sofa',        rgb2bgr((  0,   0, 142))),
    # Label('train',       rgb2bgr((  0,   0, 230))),
    # Label('tvmonitor',   rgb2bgr((119,  11,  32)))
    Label('face', rgb2bgr((255,0,0)))
    ]

#-------------------------------------------------------------------------------
class PascalVOCSource:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.num_classes   = len(label_defs)
        self.colors        = {l.name: l.color for l in label_defs}
        self.lid2name      = {i: l.name for i, l in enumerate(label_defs)}
        self.lname2id      = {l.name: i for i, l in enumerate(label_defs)}
        self.num_train     = 0
        self.num_valid     = 0
        self.num_test      = 0
        self.train_samples = []
        self.valid_samples = []
        self.test_samples  = []

    #---------------------------------------------------------------------------
    def __build_sample_list(self, root,image_dir,info_dir):
        """
        Build a list of samples for the VOC dataset (either trainval or test)
        """
        samples     = []
        file = open(root+info_dir ,'r')
        All = file.read()
        file.close()
        lines = All.split('\n')
        print(len(lines))
        #-----------------------------------------------------------------------
        # Process each annotated sample
        #-----------------------------------------------------------------------
        i=0
        while i<(len(lines)-1)/10:
            #---------------------------------------------------------------
            # Get the file dimensions
            #---------------------------------------------------------------
            filename = root+image_dir+lines[i]
            img = cv2.imread(filename)
            imgsize = Size(img.shape[1], img.shape[0])
            #---------------------------------------------------------------
            # Get boxes for all the objects
            #---------------------------------------------------------------
            boxes    = []
            i+=1
            num_objects  = int(lines[i])
            if not num_objects:
                i+=2
                continue
            i+=1
            for obj in range(num_objects):
                #-----------------------------------------------------------
                # Get the properties of the box and convert them to the
                # proportional terms
                #-----------------------------------------------------------
                label = 'face'
                xmin = int(lines[i+obj].split()[0])-int(lines[i+obj].split()[2])/2
                xmax = int(lines[i+obj].split()[0])+int(lines[i+obj].split()[2])/2
                ymin = int(lines[i+obj].split()[1])-int(lines[i+obj].split()[3])/2
                ymax = int(lines[i+obj].split()[1])+int(lines[i+obj].split()[3])/2
                center, size = abs2prop(xmin, xmax, ymin, ymax, imgsize)
                box = Box(label, self.lname2id[label], center, size)
                boxes.append(box)
            i+=num_objects
            sample = Sample(filename, boxes, imgsize)
            samples.append(sample)
        return samples

    #---------------------------------------------------------------------------
    def load_trainval_data(self, data_dir, valid_fraction):
        """
        Load the training and validation data
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """

        #-----------------------------------------------------------------------
        # Process the samples defined in the relevant file lists
        #-----------------------------------------------------------------------

        valid_samples = self.__build_sample_list(data_dir, '/WIDER_val/images/',
                                                 '/wider_face_split/wider_face_val_bbx_gt.txt')
        train_samples = self.__build_sample_list(data_dir,'/WIDER_train/images/',
                                                 '/wider_face_split/wider_face_train_bbx_gt.txt')


        #-----------------------------------------------------------------------
        # Final set up and sanity check
        #-----------------------------------------------------------------------
        self.valid_samples = valid_samples
        self.train_samples = train_samples

        if len(self.train_samples) == 0:
            raise RuntimeError('No training samples found in ' + data_dir)

        if valid_fraction > 0:
            if len(self.valid_samples) == 0:
                raise RuntimeError('No validation samples found in ' + data_dir)

        self.num_train = len(self.train_samples)
        self.num_valid = len(self.valid_samples)

    #---------------------------------------------------------------------------
    def load_test_data(self, data_dir):
        """
        Load the test data
        :param data_dir: the directory where the dataset's file are stored
        """
        root = data_dir + '/test/VOCdevkit/VOC2012'
        annot = self.__build_annotation_list(root, 'test')
        self.test_samples  = self.__build_sample_list(root, annot,
                                                      'test_VOC2012')

        if len(self.test_samples) == 0:
            raise RuntimeError('No testing samples found in ' + data_dir)

        self.num_test  = len(self.test_samples)

#-------------------------------------------------------------------------------
def get_source():
    return PascalVOCSource()
