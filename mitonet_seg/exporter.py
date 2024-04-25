import webknossos as wk
from tempfile import TemporaryDirectory

def local_export(dataset, layer_name, mag, seg, bbox, dtype_per_layer="uint16"):
    """Local export of segmentation layer to WebKnossos (to existing dataset)"""
    # Make new "segmentation" layer
    segmentation_layer = dataset.add_layer(
            layer_name, 
            wk.SEGMENTATION_CATEGORY,
            dtype_per_layer=dtype_per_layer,
            compressed=True,
            largest_segment_id=None)
    mag = segmentation_layer.add_mag(mag, compress=True)
    
    # Add "Mag" and write segmentation data to "Mag"
    mag.write(data=seg,
              absolute_offset=bbox.topleft)
    segmentation_layer.refresh_largest_segment_id()

    # Downsample segmentation
    segmentation_layer.downsample(coarsest_mag=wk.Mag("128-128-1"),
                                  sampling_mode="constant_z")

def remote_export(url, token, new_dataset_name, layer_name, voxel_size, mag, 
                  bbox, seg, layers_2_link, dtype_per_layer="uint16", downsample=False):
    """Remote export of segmentation layer to WebKnossos (creates a new dataset)"""
    with wk.webknossos_context(
        url=url,
        token=token,
    ):
        with TemporaryDirectory() as tempdir:
            new_dataset = wk.Dataset(
                dataset_path=tempdir,
                name=new_dataset_name,
                voxel_size=voxel_size
            )
            warped_layer = new_dataset.add_layer(
                layer_name,
                wk.SEGMENTATION_CATEGORY,
                dtype_per_layer=dtype_per_layer,
                compressed=True
                )
            warped_layer.bounding_box = bbox.align_with_mag(mag) # Highest MAG
            warped_layer.add_mag(mag, compress=True).write(seg)
            
            # Downsample segmentation
            if downsample:  
                warped_layer.downsample(
                    coarsest_mag=wk.Mag("128-128-1"),
                    sampling_mode="constant_z"
                    )
                
            # Upload
            remote_ds = new_dataset.upload(
                layers_to_link=layers_2_link,
            )
    return remote_ds
        