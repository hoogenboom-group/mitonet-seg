from tqdm import tqdm
import webknossos as wk
import numpy as np


def import_wk_dataset_remote(TOKEN, url, dataset_name, organization_id, MAG, layer="color") -> None:
    # use the context to get access to your group
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

def import_wk_dataset_local(dir_path, MAG, layer="color") -> None:
    # open local dataset in given directory 
    dataset = wk.Dataset.open(
        dataset_path = dir_path)
    voxel_size = dataset.voxel_size

    EM = dataset.get_layer(layer) # Layer
    mag_view = EM.get_mag(MAG) # MagView
        
    # return data, voxel size
    return dataset, mag_view, voxel_size

def read_data(bboxes, mag_view):
    # Read data from remote in chunks
    # By looping through bboxes
    for bbox_small in tqdm(bboxes,
                        desc="Reading data from bboxes",
                        total=len(bboxes),
                        unit="bbox"):
        # Fill slice in array with data from smaller bbox
        view_small = mag_view.get_view(absolute_offset=bbox_small.topleft, 
                                       size=bbox_small.size)
        data = view_small.read()
        yield bbox_small, data