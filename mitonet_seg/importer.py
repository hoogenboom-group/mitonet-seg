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

def read_data(layer_bbox, MAG, bboxes, mag_view): 
    # Infer data set dimensions (in desired mag)
    dim = layer_bbox.in_mag(MAG).size

    # Allocate memory
    data = np.zeros((1, dim.x, dim.y, dim.z), dtype=np.uint8)

    # Read data from remote in chunks
    # By looping through bboxes
    x_offset = bboxes[0].topleft[0]
    y_offset = bboxes[0].topleft[1]
    for bbox_small in tqdm(bboxes,
                        desc="Reading data from bboxes",
                        total=len(bboxes),
                        unit="bbox"):
        # Fill slice in array with data from smaller bbox
        view_small = mag_view.get_view(absolute_offset=bbox_small.topleft, 
                                    size=bbox_small.size)
        x_start = bbox_small.in_mag(MAG).topleft.x - x_offset
        x_end = x_start + bbox_small.in_mag(MAG).size.x
        y_start = bbox_small.in_mag(MAG).topleft.y - y_offset
        y_end = y_start + bbox_small.in_mag(MAG).size.y
        
        data[:, 
            x_start:x_end,
            y_start:y_end,
            :] = view_small.read()
    
    # return data
    return data