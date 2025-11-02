import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import options
from tools import fusers_helper
from utils.dataset_utils import get_dataset
from utils.generic_utils import to_gpu, cache_model_outputs

"""
sys.argv=['test.py', '--name','HERO_MODEL', '--output_base_path', './output/Matterport3D', '--config_file', 'configs/models/pano_model.yaml', '--load_weights_from_checkpoint', 'weights/pano_model.ckpt', '--data_config', 'configs/data/matterport3d_debug.yaml',
     '--num_workers', '8','--batch_size', '1','--run_fusion','--depth_fuser','ours','--fuse_color','--dump_depth_visualization']
"""

sys.argv=['test.py', '--name','HERO_MODEL', '--output_base_path', './output/S2D3D', '--config_file', 'configs/models/pano_model.yaml', '--load_weights_from_checkpoint', 'weights/pano_model.ckpt', '--data_config', 'configs/data/S2D3D_default_test.yaml',
     '--num_workers', '8','--batch_size', '1','--run_fusion','--depth_fuser','ours','--fuse_color','--dump_depth_visualization']



option_handler = options.OptionsHandler()
option_handler.parse_and_merge_options()
option_handler.pretty_print_options()
print("\n")
opts = option_handler.options

# if no GPUs are available for us then, use the 32 bit on CPU
if opts.gpus == 0:
    print("Setting precision to 32 bits since --gpus is set to 0.")
    opts.precision = 32

opts.depth_fuser == "ours"
# get dataset
dataset_class, scans = get_dataset(opts.dataset, 
                    opts.dataset_scan_split_file, opts.single_debug_scan_id)

# path where results for this model, dataset, and tuple type are.
results_path = os.path.join(opts.output_base_path, opts.name, 
                                    opts.dataset, opts.frame_tuple_type)
for scan in tqdm(scans):
    fuser = fusers_helper.get_fuser(opts, scan)
    dataset = dataset_class(
            opts.dataset_path,
            split=opts.split,
            mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
            limit_to_scan_id=scan,
            include_full_res_depth=True,
            tuple_info_file_location=opts.tuple_info_file_location,
            num_images_in_tuple=None,
            shuffle_tuple=opts.shuffle_tuple,
            include_high_res_color=(
                            (opts.fuse_color and opts.run_fusion)
                            or opts.dump_depth_visualization
                        ),
            include_full_depth_K=True,
            skip_frames=opts.skip_frames,
            skip_to_frame=opts.skip_to_frame,
            image_width=opts.image_width,
            image_height=opts.image_height,
            pass_frame_id=True,
        )

    dataloader = torch.utils.data.DataLoader(
                                    dataset, 
                                    batch_size=opts.batch_size, 
                                    shuffle=False, 
                                    num_workers=opts.num_workers, 
                                    drop_last=False,
                                )


    for batch_ind, batch in enumerate(tqdm(dataloader)):
        # get data, move to GPU
        cur_data, src_data = batch
        cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
        src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

        depth_gt = cur_data["full_res_depth_b1hw"]
        #print(depth_gt.max())
        #print(depth_gt.min())
        #print(depth_gt.mean())
        color_frame = (cur_data["high_res_color_b3hw"] 
                    if  "high_res_color_b3hw" in cur_data 
                        else cur_data["image_b3hw"])
        fuser.fuse_frames(
                            #upsampled_depth_pred_b1hw, 
                            depth_gt,
                            cur_data["K_full_depth_b44"], 
                            cur_data["cam_T_world_b44"], 
                            color_frame
                    )

    mesh_output_dir = os.path.join(results_path, "meshes")                         
    fuser.export_mesh(
            os.path.join(mesh_output_dir, 
                f"{scan.replace('/', '_')}.ply"),
        )