import math

# Data import
import webknossos as wk
import numpy as np

# Empanada
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from empanada.data import VolumeDataset
from empanada.inference.engines import PanopticDeepLabRenderEngine3d
from empanada.inference import filters
from empanada.config_loaders import load_config
from empanada.inference.patterns import *

def get_data_from_bbox(mag_view, bbox) -> None:
    # Generate view from bounding box and read the zarr 
    view = mag_view.get_view(absolute_offset=bbox.topleft, 
                             size=bbox.size) # "absolute_offset" and "size" are in Mag(1)!
    data = view.read() # reads the actual data

    # return data chunk
    return data

def define_bbox_chunks(view, mag, bbox_size=5000):
    """Define bbox chunks for parallel import
    
    view: webknossos.dataset.view
            Bounding box to data in specific MagView
    mag: webknossos.geometry.mag
            Magnification level of data layer (usually highest mag)
    bbox_size: int
            Size of bbox chunks
    returns list of wk.BoundingBox objects
    """
    
    # Infer data set dimensions (in desired mag)
    dim = view.bounding_box.in_mag(mag).size
    
    # Offset
    x0 = view.bounding_box.in_mag(mag).topleft.x
    y0 = view.bounding_box.in_mag(mag).topleft.y

    # Determine number of chunks to split data into in all dimensions
    chunks_x, chunks_y, chunks_z = math.ceil(dim.x / bbox_size), math.ceil(dim.y / bbox_size), math.ceil(dim.z / 256)

    # Determine z size of bbox (x an y sizes are defined by bbox_size)
    size_z = min(dim.z, 256)

    # From # chunks, define bboxes to be used
    # Loop over x, y, z chunk indices, define bbox on multiples of bbox_size starting from topleft of og bbox
    bboxes = []
    for i in range(chunks_x):
        for j in range(chunks_y):
            if bbox_size*(i+1) >= dim.x: # check if bbox is larger than max x and adjust bbox dimension 
                bbox_size_x = dim.x - bbox_size*i
            else: # not exceeding stack dimensions, use regular bbox_size
                bbox_size_x = bbox_size
            
            if bbox_size*(j+1) >= dim.y: # check if bbox is larger than max y and adjust bbox dimension 
                bbox_size_y = dim.y - bbox_size*j
            else: # not exceeding stack dimensions, use regular bbox_size
                bbox_size_y = bbox_size
            
            for k in range(chunks_z):
                # Generate bbox
                bbox_small = wk.BoundingBox(topleft=(x0 + bbox_size*i, y0 + bbox_size*j, 256*k),
                                            size=(bbox_size_x, bbox_size_y, size_z))\
                                                .from_mag_to_mag1(from_mag=mag)        
                bboxes.append(bbox_small)   
                
    return bboxes