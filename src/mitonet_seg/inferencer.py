import numpy as np
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

import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def inference_3d(volume_data, config, mode='stack', qlen=3, nmax=20000, seg_thr=0.3, nms_thr=0.1, nms_kernel=3, 
                 iou_thr=0.25, ioa_thr=0.25, pixel_vote_thr=2, cluster_io_thr=0.75, min_size=200, 
                 min_span=2, downsample_f=1, one_view=True, fine_boundaries=False, use_cpu=False, nworkers=1):
                
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
    consensus_vol = np.expand_dims(consensus_vol, axis=0)
    print('Finished!')
    return consensus_vol