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
EM_LAYER = "color" # EM layer to predict mitochondria from
CONFIG = os.path.abspath("configs/FinetunedModel.yaml") # MitoNet model configuration file
REMOTE = False # Set to "True" if importing data remotely (without filesystem mount)
USE_CPU = False # Use GPU, setting to "True" falls back to CPU (computations are much slower)
DOWNSAMPLE = False # Set to "True" to automatically downsample segmentations
DTYPE_SEG = 'uint16' # Datatype, standard is uint16

NEW_LAYER_NAME = EM_LAYER + "_MitoNet_Finetuned"  # New predictions layer name
NEW_DATASET_NAME = DATASET_NAME + '_test' # If remote exporting

# Parameters that should likely be set once
MAG_X, MAG_Y, MAG_Z = 4, 4, 1 # Magnification level (x, y, z) in WebKnossos to be used for segmentation. Default is (4, 4, 1)
TOKEN = None # String, generate from https://webknossos.tnw.tudelft.nl/auth/token
ORGANIZATION_ID = "hoogenboom-group" # "hoogenboom-group"
URL = "https://webknossos.tnw.tudelft.nl" # "https://webknossos.tnw.tudelft.nl" 
BASE_DIR = f"/home/ajkievits/sonic" # Mount location or "/long_term_storage" if directly running on sonic

# Set GPU
if not USE_CPU:
    # set the environment variable 'CUDA_VISIBLE_DEVICES'
    # this sets which GPU can be seen by the program
    # Either set it to "0", "1", "2", "3"
    # or if you want to use multiple GPUs add commas between the numbers
    # e.g. "0,1"
    # the numbers correspond with those in the command 'nvidia-smi'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _main():
    MAG = wk.Mag(f"{MAG_X}-{MAG_Y}-{MAG_Z}") # Set magnification for WebKnossos
    if REMOTE:
        assert TOKEN, f"No WebKnossos token supplied, please generate from https://webknossos.tnw.tudelft.nl/auth/token"
        logging.info("Fetching dataset remotely")
        # Fetch dataset remotely
        dataset, mag_view, voxelsize = importer.import_wk_dataset_remote(TOKEN, 
                                                                         URL, 
                                                                         DATASET_NAME,
                                                                         ORGANIZATION_ID,
                                                                         MAG,
                                                                         EM_LAYER)    
    else:
        DIR_PATH = pathlib.Path(f"{BASE_DIR}/webknossos/binaryData/{ORGANIZATION_ID}/{DATASET_NAME}")
        # Fetch dataset locally
        logging.info("Fetching dataset locally")
        dataset, mag_view, voxelsize = importer.import_wk_dataset_local(DIR_PATH,
                                                                        MAG,
                                                                        EM_LAYER)
    # Bounding box
    layer_bbox = dataset.get_layer(EM_LAYER).bounding_box.align_with_mag(MAG)   
    
    view = mag_view.get_view(offset=layer_bbox.in_mag(MAG).topleft, size=layer_bbox.in_mag(MAG).size)
    if REMOTE:
        # Define chunks because full dataset size may not import
        bboxes = define_bbox_chunks(view, mag=MAG, bbox_size=5000)
    else:
        # Import in one go if locally (works)
        bboxes = layer_bbox.in_mag(MAG)
    
    # Read data into memory
    logging.info(f"Reading data from {DATASET_NAME} into memory")
    data = importer.read_data(layer_bbox, MAG, bboxes, mag_view)
    
    # Run 3d inference (weirdly has to be on yz stack)
    mito_labels = inference_3d(CONFIG, data, mode='stack', qlen=1, nmax=100000, seg_thr=0.5, nms_thr=0.1, nms_kernel=3, 
                           iou_thr=0.25, ioa_thr=0.25, pixel_vote_thr=2, cluster_io_thr=0.75, min_size=200, 
                           min_span=1, downsample_f=1, one_view=True, fine_boundaries=False, use_cpu=USE_CPU, nworkers=1)
    # Export   
    if REMOTE:
        # layers_2_link = dataset.get_color_layers() + dataset.get_segmentation_layers() BREAKS FOR SOME REASON
        layers_2_link = [dataset.get_layer(EM_LAYER)] # layers_2_link needs to be an iterable, only link EM layer in this case
        logging.info(f"Exporting remotely, layers to link: {layers_2_link}")
        remote_ds = exporter.remote_export(url=URL, token=TOKEN, new_dataset_name=NEW_DATASET_NAME, layer_name=NEW_LAYER_NAME, 
                                           voxel_size=voxelsize, mag=MAG, bbox=layer_bbox, seg=mito_labels, 
                                           layers_2_link=layers_2_link, dtype_per_layer=DTYPE_SEG, downsample=DOWNSAMPLE)
        logging.info(f"Succesfully uploaded {remote_ds.url}")
    else:
        logging.info(f"Exporting locally")
        exporter.local_export(dataset=dataset, layer_name=NEW_LAYER_NAME, mag=MAG, 
                              seg=mito_labels, bbox=layer_bbox, dtype_per_layer=DTYPE_SEG)
        
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )
    _main()