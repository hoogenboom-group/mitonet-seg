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

def import_wk_dataset_remote(TOKEN, 
                             url, 
                             dataset_name, 
                             organization_id, 
                             MAG,
                             layer="color") -> None:
    # use the context to get acces to your group
    with wk.webknossos_context(
        token=TOKEN,
        url=url
    ):

    # open remote dataset with dataset name, organization id and WebKnossos url 
        dataset = wk.Dataset.open_remote(
                dataset_name_or_url = dataset_name,
                organization_id = organization_id,
                webknossos_url = url)
        voxel_size = dataset.voxel_size

        EM = dataset.get_layer(layer) # Layer
        mag_view = EM.get_mag(MAG) # MagView
        
    # return data, voxel size
    return dataset, mag_view, voxel_size

def import_wk_dataset_local(dir_path,
                            MAG,
                            layer="color") -> None:
    # open local dataset in given directory 
    dataset = wk.Dataset.open(
        dataset_path = dir_path)
    voxel_size = dataset.voxel_size

    EM = dataset.get_layer(layer) # Layer
    mag_view = EM.get_mag(MAG) # MagView
        
    # return data, voxel size
    return dataset, mag_view, voxel_size

def get_data_from_bbox(mag_view, bbox) -> None:
    # Generate view from bounding box and read the zarr 
    view = mag_view.get_view(absolute_offset=bbox.topleft, 
                             size=bbox.size) # "absolute_offset" and "size" are in Mag(1)!
    data = view.read() # reads the actual data

    #return data chunk
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

def inference_3d(config, volume_data, mode='stack', qlen=3, nmax=20000, seg_thr=0.3, nms_thr=0.1, nms_kernel=3, 
                iou_thr=0.25, ioa_thr=0.25, pixel_vote_thr=2, cluster_io_thr=0.75, min_size=200, 
                min_span=2, downsample_f=1, one_view=True, fine_boundaries=False, use_cpu=True, nworkers=10):
                
    # read the model config file
    config = load_config(config)

    # set device and determine model to load
    device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")
    use_quantized = str(device) == 'cpu' and config.get('model_quantized') is not None
    model_key = 'model_quantized' if use_quantized  else 'model'
    
    if os.path.isfile(config[model_key]):
        model = torch.jit.load(config[model_key])
    else:
        model = torch.hub.load_state_dict_from_url(config[model_key])

    model = model.to(device)
    model.eval()

    # data
    volume = np.squeeze(volume_data)
    shape = volume.shape

    if mode == 'orthoplane':
        axes = {'xy': 0, 'xz': 1, 'yz': 2} # x, y, z
    else:
        axes = {'yz': 2}

    eval_tfs = A.Compose([
        A.Normalize(**config['norms']),
        ToTensorV2()
    ])

    trackers = {}
    class_labels = list(config['class_names'].keys())
    thing_list = config['thing_list']
    label_divisor = nmax

    # create a separate tracker for
    # each prediction axis and each segmentation class
    trackers = create_axis_trackers(axes, class_labels, label_divisor, shape)

    for axis_name, axis in axes.items():
        print(f'Predicting {axis_name} stack')
        stack = None

        # create the inference engine
        inference_engine = PanopticDeepLabRenderEngine3d(
            model, thing_list=thing_list,
            median_kernel_size=qlen,
            label_divisor=label_divisor,
            nms_threshold=nms_thr,
            nms_kernel=nms_kernel,
            confidence_thr=seg_thr,
            padding_factor=config['padding_factor'],
            coarse_boundaries=not fine_boundaries
        )

        # create a separate matcher for each thing class
        matchers = create_matchers(thing_list, label_divisor, iou_thr, ioa_thr)

        # setup matcher for multiprocessing
        queue = mp.Queue()
        rle_stack = []
        matcher_out, matcher_in = mp.Pipe()
        matcher_args = (
            matchers, queue, rle_stack, matcher_in,
            class_labels, label_divisor, thing_list
        )
        matcher_proc = mp.Process(target=forward_matching, args=matcher_args)
        matcher_proc.start()

        # make axis-specific dataset
        dataset = VolumeDataset(volume, axis, eval_tfs, scale=downsample_f)

        num_workers = nworkers
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            pin_memory=(device == 'gpu'), drop_last=False,
            num_workers=num_workers
        )

        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            size = batch['size']

            # pads and crops image in the engine
            # upsample output by same factor as downsampled input
            pan_seg = inference_engine(image, size, upsampling=downsample_f)

            if pan_seg is None:
                queue.put(None)
                continue
            else:
                pan_seg = pan_seg.squeeze().cpu().numpy()
                queue.put(pan_seg)

        final_segs = inference_engine.end(downsample_f)
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                pan_seg = pan_seg.squeeze().cpu().numpy()
                queue.put(pan_seg)

        # finish and close forward matching process
        queue.put('finish')
        rle_stack = matcher_out.recv()[0]
        matcher_proc.join()

        print(f'Propagating labels backward through the stack...')
        for index,rle_seg in tqdm(backward_matching(rle_stack, matchers, shape[axis]), total=shape[axis]):
            update_trackers(rle_seg, index, trackers[axis_name])

        finish_tracking(trackers[axis_name])
        for tracker in trackers[axis_name]:
            filters.remove_small_objects(tracker, min_size=min_size)
            filters.remove_pancakes(tracker, min_span=min_span)

    # create the final instance segmentations
    for class_id, class_name in config['class_names'].items():
        print(f'Creating consensus segmentation for class {class_name}...')
        class_trackers = get_axis_trackers_by_class(trackers, class_id)

        # merge instances from orthoplane inference if applicable
        if mode == 'orthoplane':
            if class_id in thing_list:
                consensus_tracker = create_instance_consensus(
                    class_trackers, pixel_vote_thr, cluster_iou_thr, one_view
                )
                filters.remove_small_objects(consensus_tracker, min_size=min_size)
                filters.remove_pancakes(consensus_tracker, min_span=min_span)
            else:
                consensus_tracker = create_semantic_consensus(class_trackers, pixel_vote_thr)
        else:
            consensus_tracker = class_trackers[0]

        dtype = np.uint32 if class_id in thing_list else np.uint8

        # decode and fill the instances
        consensus_vol = np.zeros(shape, dtype=dtype)
        fill_volume(consensus_vol, consensus_tracker.instances)

    print('Finished!')
    return consensus_vol