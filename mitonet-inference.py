"""MitoNet 3D inference script

relies on provided parameters being set in the script for now
"""
import logging
import pathlib
import os
import webknossos as wk

import mitonet_seg.importer as importer 
from mitonet_seg.utils import define_bbox_chunks, get_data_from_bbox
import mitonet_seg.exporter as exporter
from mitonet_seg.inferencer import inference_3d 

# script properties
DATASET_NAME = "20230626_RP_serial_warped_full_1x_mito_seg" # Dataset name as in WebKnossos
EM_LAYER = "color"
CONFIG = os.path.abspath("configs/MitoNet_v1.yaml") # MitoNet model configuration file
REMOTE = False # Set to "True" if importing data remotely (without filesystem mount)
USE_CPU = False # Use GPU, setting to "True" falls back to CPU (computations are much slower)
DOWNSAMPLE = True # Set to "True" to automatically downsample segmentations
DTYPE_SEG = 'uint16' # Datatype, standard is uint16

# Parameters that should likely be set once
MAG_X, MAG_Y, MAG_Z = 4, 4, 1 # Magnification level (x, y, z) in WebKnossos to be used for segmentation. Default is (4, 4, 1)
TOKEN = None # Generate from https://webknossos.tnw.tudelft.nl/auth/token
ORGANIZATION_ID = "hoogenboom-group" # "hoogenboom-group"
URL = "https://webknossos.tnw.tudelft.nl" # "https://webknossos.tnw.tudelft.nl" 
BASE_DIR = "/home/{USER}/sonic" # Mount location or "/long_term_storage" if directly running on sonic

def _main():
    MAG = wk.Mag(f"{MAG_X}-{MAG_Y}-{MAG_Z}") # Set magnification for WebKnossos
    if REMOTE:
        assert TOKEN, f"No WebKnossos token supplied, please generate from https://webknossos.tnw.tudelft.nl/auth/token"
        # Fetch dataset remotely
        dataset, mag_view, voxelsize = importer.import_wk_dataset_remote(TOKEN, 
                                                                         URL, 
                                                                         DATASET_NAME,
                                                                         ORGANIZATION_ID,
                                                                         MAG,
                                                                         EM_LAYER)    
    else:
        DIR_PATH = pathlib.Path("{BASE_DIR}/webknossos/binaryData/{ORGANIZATION_ID}/{DATASET_NAME}")
        # Fetch dataset locally
        dataset, mag_view, voxelsize = importer.import_wk_dataset_local(DIR_PATH,
                                                                        MAG,
                                                                        EM_LAYER)
    # Bounding box
    layer_bbox = dataset.get_layer(EM_LAYER).bounding_box   
    
    # Define chunks because array size may exceed vram
    view = mag_view.get_view(offset=layer_bbox.in_mag(MAG).topleft, size=layer_bbox.in_mag(MAG).size)
    bboxes = define_bbox_chunks(view, mag=MAG, bbox_size=10000)
    
    # Read data into memory
    data = importer.read_data(layer_bbox, MAG, bboxes, mag_view)
    
    # Run 3d inference (weirdly has to be on yz stack)
    mito_labels = inference_3d(CONFIG, data, mode='stack', qlen=1, nmax=100000, seg_thr=0.5, nms_thr=0.1, nms_kernel=3, 
                           iou_thr=0.25, ioa_thr=0.25, pixel_vote_thr=2, cluster_io_thr=0.75, min_size=200, 
                           min_span=1, downsample_f=1, one_view=True, fine_boundaries=False, use_cpu=USE_CPU, nworkers=1)    
    if REMOTE:
        layers_2_link = dataset.get_color_layers() + dataset.get_segmentation_layers()
        new_dataset_name = DATASET_NAME + '_mito_seg'
        new_layer_name = EM_LAYER + "_MitoNet"
        exporter.remote_export(url=URL, token=TOKEN, new_dataset_name=new_dataset_name, layer_name=new_layer_name, voxel_size=voxelsize, mag=MAG, 
                               bbox=layer_bbox, seg=mito_labels, layers_2_link=layers_2_link, dtype_per_layer=DTYPE_SEG, downsample=DOWNSAMPLE)
    else:
        exporter.local_export(dataset=dataset, layer_name=new_layer_name, mag=MAG, dtype_per_layer=DTYPE_SEG)
    
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )
    _main()