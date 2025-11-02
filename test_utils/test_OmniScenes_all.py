""" 
    Predicts depth maps using a DepthModel model. Uses an MVS dataset from 
    datasets.
    
    All results will be stored at a base results folder (results_path) at:
        opts.output_base_path/opts.name/opts.dataset/opts.frame_tuple_type/

    frame_tuple_type is the type of image tuple used for MVS. A selection should 
    be provided in the data_config file you used. 

    By default will compute depth scores for each frame and provide both frame 
    averaged and scene averaged metrics. The script will save these scores (per 
    scene and totals) under:
        results_path/scores

    We've done our best to ensure that a torch batching bug through the matching 
    encoder is fixed for (<10^-4) accurate testing by disabling image batching 
    through that encoder. Run `--batch_size 4` at most if in doubt, and if 
    you're looking to get as stable as possible numbers and avoid PyTorch 
    gremlins, use `--batch_size 1`  for comparison evaluation.


    If you want to use this for speed, set --fast_cost_volume to True. This will
    enable batching through the matching encoder and will enable an einops 
    optimized feature volume.

    This script can also be used to perform a few different auxiliary tasks, 
    including:

    ### TSDF Fusion
    To run TSDF fusion provide the --run_fusion flag. You have two choices for 
    fusers
    1) '--depth_fuser ours' (default) will use our fuser, whose meshes are used 
        in most visualizations and for scores. This fuser does not support 
        color. You must use the provided version of skimage for our custom 
        measure.marching_cubes implementation that allows exporting a single 
        walled mesh.
    2) '--depth_fuser open3d' will use the open3d depth fuser. This fuser 
        supports color and you can enable this by using the '--fuse_color' flag. 
    
    By default, depth maps will be clipped to 3m for fusion and a tsdf 
    resolution of 0.04m3 will be used, but you can change that by changing both 
    '--max_fusion_depth' and '--fusion_resolution'

    You can optionnally ask for predicted depths used for fusion to be masked 
    when no vaiid MVS information exists using '--mask_pred_depths'. This is not 
    enabled by default.

    You can also fuse the best guess depths from the cost volume before the 
    U-Net that fuses the strong image prior. You can do this by using 
    --fusion_use_raw_lowest_cost.

    Meshes will be stored under results_path/meshes/mesh_options/

    ### Cache depths
    You can optionally store depths by providing the '--cache_depths' flag. 
    They will be stored at
        results_path/depths

    ### Quick viz
    There are other scripts for deeper visualizations of output depths and 
    fusion, but for quick export of depth map visualization you can use 
    '--dump_depth_visualization'. Visualizations will be stored at 
        results_path/viz/quick_viz/


    Example command to just compute scores and cache depths
        python test.py --name HERO_MODEL \
                    --output_base_path OUTPUT_PATH \
                    --config_file configs/models/hero_model.yaml \
                    --load_weights_from_checkpoint weights/hero_model.ckpt \
                    --data_config configs/data/scannet_default_test.yaml \
                    --num_workers 8 \
                    --cache_depths \
                    --batch_size 4;

    Example command to fuse depths to get meshes
        python test.py --name HERO_MODEL \
                    --output_base_path OUTPUT_PATH \
                    --config_file configs/models/hero_model.yaml \
                    --load_weights_from_checkpoint weights/hero_model.ckpt \
                    --data_config configs/data/scannet_default_test.yaml \
                    --num_workers 8 \
                    --run_fusion \
                    --batch_size 4;

    Example command to output quick depth visualizations
        python test.py --name HERO_MODEL \
                    --output_base_path OUTPUT_PATH \
                    --config_file configs/models/hero_model.yaml \
                    --load_weights_from_checkpoint weights/hero_model.ckpt \
                    --data_config configs/data/scannet_default_test.yaml \
                    --num_workers 8 \
                    --dump_depth_visualization \
                    --batch_size 4;

    Example command to fuse depths to get color meshes
        python test.py --name HERO_MODEL \
                    --output_base_path OUTPUT_PATH \
                    --config_file configs/models/hero_model.yaml \
                    --load_weights_from_checkpoint weights/hero_model.ckpt \
                    --data_config configs/data/scannet_default_test.yaml \
                    --num_workers 8 \
                    --run_fusion \
                    --depth_fuser open3d \
                    --fuse_color \
                    --batch_size 4;
"""


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiment_modules.depth_model import DepthModel
import options
from tools import fusers_helper
from utils.dataset_utils import get_dataset
from utils.generic_utils import to_gpu, cache_model_outputs
from utils.metrics_utils import ResultsAverager, compute_depth_metrics_batched
from utils.visualization_utils import quick_viz_export
import numpy as np

import modules.cost_volume as cost_volume


import os

def extract_scene_name(file_path):
    """
    从文件路径中提取场景名称（去除前缀和后缀），并确保返回路径以.txt结尾。
    """
    # 获取文件名（不包括路径）
    base_name = os.path.basename(file_path)
    
    # 假设场景名称是路径的一部分，去除前缀和后缀
    parts = base_name.split('_')
    if len(parts) >= 3:
        scene_name = '_'.join(parts[1:-2])  # 返回去除前后缀的部分
        return scene_name + '.txt'  # 确保返回的路径以.txt结尾
    
    return file_path  # 如果没有按预期格式，返回原始名称






def generate_bounds_for_scan_from_file(scan, file_path):
    """
    根据文件中的扫描名称查找边界值，并在每个方向上根据相机的最大值和最小值调整边界。
    
    参数:
    scan (str): 扫描的名称。
    file_path (str): 文本文件的路径，文件中包含场景名称和边界值。
    
    返回:
    dict: 包含xmin, xmax, ymin, ymax, zmin, zmax的边界字典。
    """
    scan_path = extract_scene_name(scan)
    file_path = os.path.join(file_path, scan_path)
    
    # 读取点云文件中的坐标
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # 读取每一行，取前三列（x, y, z）坐标数据
            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append((x, y, z))
    
    # 将坐标转化为numpy数组，方便后续计算
    points = np.array(points)
    
    # 计算x, y, z方向上的最小值和最大值
    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    
    # 根据要求调整边界
    return {
        "xmin": xmin-0.5 ,
        "xmax": xmax+0.5 ,
        "ymin": ymin-0.5 ,
        "ymax": ymax+0.5 ,
        "zmin": zmin-0.5 ,
        "zmax": zmax+0.5 
    }
  
    


def main(opts):

    # get dataset
    dataset_class, scans = get_dataset(opts.dataset, 
                        opts.dataset_scan_split_file, opts.single_debug_scan_id)
 
    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(opts.output_base_path, opts.name, 
                                        opts.dataset, opts.frame_tuple_type)

    # set up directories for fusion
    if opts.run_fusion:
        mesh_output_folder_name = f"{opts.fusion_resolution}_{opts.fusion_max_depth}_{opts.depth_fuser}"

        if opts.mask_pred_depth:
            mesh_output_folder_name = mesh_output_folder_name + "_masked"
        if opts.fuse_color:
            mesh_output_folder_name = mesh_output_folder_name + "_color"
        if opts.fusion_use_raw_lowest_cost:
            mesh_output_folder_name = mesh_output_folder_name + "_raw_cv"

        mesh_output_dir = os.path.join(results_path, "meshes", 
                                                        mesh_output_folder_name)

        Path(mesh_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Running fusion! Using {opts.depth_fuser} ".center(80, "#"))
        print(f"Output directory:\n{mesh_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    # set up directories for caching depths
    if opts.cache_depths:
        # path where we cache depth maps
        depth_output_dir = os.path.join(results_path, "depths")

        Path(depth_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Caching depths.".center(80, "#"))
        print(f"Output directory:\n{depth_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    # set up directories for quick depth visualizations
    if opts.dump_depth_visualization:
        viz_output_folder_name = "quick_viz"
        viz_output_dir = os.path.join(results_path, "viz", 
                                                        viz_output_folder_name)

        Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"".center(80, "#"))
        print(f" Saving quick viz.".center(80, "#"))
        print(f"Output directory:\n{viz_output_dir} ".center(80, "#"))
        print(f"".center(80, "#"))
        print("")

    # set up directory for saving scores
    scores_output_dir = os.path.join(results_path, "scores")
    Path(scores_output_dir).mkdir(parents=True, exist_ok=True)

    # Set up model. Note that we're not passing in opts as an argument, although
    # we could. We're being pretty stubborn with using the options the model had
    # used when training, saved internally as part of hparams in the checkpoint.
    # You can change this at inference by passing in 'opts=opts,' but there 
    # be dragons if you're not careful.
    model = DepthModel.load_from_checkpoint(
                                opts.load_weights_from_checkpoint,
                                args=None)
    if (opts.fast_cost_volume and  
            isinstance(model.cost_volume, cost_volume.FeatureVolumeManager)):
        model.cost_volume = model.cost_volume.to_fast()

    model = model.cuda().eval()

    # setting up overall result averagers
    all_frame_metrics = None
    all_scene_metrics = None

    all_frame_metrics = ResultsAverager(opts.name, f"frame metrics")
    all_scene_metrics = ResultsAverager(opts.name, f"scene metrics")


    with torch.inference_mode():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # loop over scans
        for scan in tqdm(scans):
            #bounds = generate_bounds_for_scan_from_file(scan,'data_splits/OmniScenes/OmniScenes_camera_center.txt')  # 自定义函数生成扫描特定的 bounds
            bounds = generate_bounds_for_scan_from_file(scan,'/home/yzm/dataset/OmniScenes_O/omniscenes_ver_1_2_release/pcd')
            # initialize fuser if we need to fuse
            if opts.run_fusion:
                fuser = fusers_helper.get_fuser(opts, scan,bounds=bounds)

            # set up dataset with current scan
            dataset = dataset_class(
                        opts.dataset_path,
                        split=opts.split,
                        mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                        limit_to_scan_id=scan,
                        include_full_res_depth=False,
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

            # initialize scene averager
            scene_frame_metrics = ResultsAverager(
                                                opts.name, 
                                                f"scene {scan} metrics"
                                            )

            for batch_ind, batch in enumerate(tqdm(dataloader)):
                # get data, move to GPU
                cur_data, src_data = batch
                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

                #depth_gt = cur_data["full_res_depth_b1hw"]
                #valid_mask_b = cur_data["full_res_mask_b_b1hw"].bool()
                pred_depth = cur_data['pred_depth_b1hw']
                # run to get output, also measure time
                '''
                start_time.record()
                # use unbatched (looping) matching encoder image forward passes 
                # for numerically stable testing. If opts.fast_cost_volume, then 
                # batch.
                outputs = model(
                                "test", cur_data, src_data, 
                                unbatched_matching_encoder_forward=(
                                    not opts.fast_cost_volume
                                ), 
                                return_mask=True,
                            )
                end_time.record()
                torch.cuda.synchronize()

                elapsed_model_time = start_time.elapsed_time(end_time)
                '''
                upsampled_depth_pred_b1hw = F.interpolate(
                                pred_depth, 
                                size=(512, 1024),
                                mode="nearest",
                            )

                # inf max depth matches DVMVS metrics, using minimum of 0.5m
                valid_mask_b =  (upsampled_depth_pred_b1hw > 0.5)
                
                #valid_mask_b[:, :80, :] =False  # 顶端320行
                #valid_mask_b[:, -80:, :] = False  # 底端320行
                
                #valid_mask_b[:,:, :80, :] =False  # 顶端320行
                valid_mask_b[:,:, -80:, :] = False  # 底端320行
                
                #print(depth_gt.min())
                #print(upsampled_depth_pred_b1hw.min())
                # Check if there any valid gt points in this sample
                '''
                if (valid_mask_b).any():
                    # compute metrics
                    metrics_b_dict = compute_depth_metrics_batched(
                        depth_gt.flatten(start_dim=1).float(), 
                        upsampled_depth_pred_b1hw.flatten(start_dim=1).float(), 
                        valid_mask_b.flatten(start_dim=1),
                        mult_a=True,
                    )

                    # go over batch and get metrics frame by frame to update 
                    # the averagers 
                    for element_index in range(depth_gt.shape[0]):
                        if (~valid_mask_b[element_index]).all():
                            # ignore if no valid gt exists
                            continue

                        element_metrics = {}
                        for key in list(metrics_b_dict.keys()):
                            element_metrics[key] = metrics_b_dict[key][element_index]
                        
                        # get per frame time in the batch
                        element_metrics["model_time"] = (elapsed_model_time /
                                                            depth_gt.shape[0])

                        # both this scene and all frame averagers
                        scene_frame_metrics.update_results(element_metrics)
                        all_frame_metrics.update_results(element_metrics)
                '''

                ######################### DEPTH FUSION #########################
                if opts.run_fusion:
                    # mask predicted depths when no vaiid MVS information 
                    # exists, off by default
                    '''
                    if opts.mask_pred_depth:
                        overall_mask_b1hw = outputs[    
                                                    "overall_mask_bhw"
                                            ].cuda().unsqueeze(1).float()

                        overall_mask_b1hw = F.interpolate(
                                overall_mask_b1hw, 
                                size=(512, 1024), 
                                mode="nearest"
                        ).bool()

                        upsampled_depth_pred_b1hw[~overall_mask_b1hw] = -1
                    '''
                    # fuse the raw best guess depths from the cost volume, off 
                    # by default
                    '''
                    if opts.fusion_use_raw_lowest_cost:
                        # upsampled_depth_pred_b1hw becomes the argmax from the 
                        # cost volume
                        upsampled_depth_pred_b1hw = outputs[
                                                        "lowest_cost_bhw"
                                                    ].unsqueeze(1)

                        upsampled_depth_pred_b1hw = F.interpolate(
                                upsampled_depth_pred_b1hw, 
                                size=(512, 1024), 
                                mode="nearest",
                            )

                        overall_mask_b1hw = outputs[
                                                "overall_mask_bhw"
                                            ].cuda().unsqueeze(1).float()

                        overall_mask_b1hw = F.interpolate(
                                overall_mask_b1hw, 
                                size=(512, 1024), 
                                mode="nearest"
                            ).bool()

                        upsampled_depth_pred_b1hw[~overall_mask_b1hw] = -1
                    '''
                    color_frame = (cur_data["high_res_color_b3hw"] 
                                    if  "high_res_color_b3hw" in cur_data 
                                     else cur_data["image_b3hw"])
                    
                    #valid_mask_b = valid_mask_b.unsqueeze(1)
                    fuser.fuse_frames(
                                        upsampled_depth_pred_b1hw, 
                                        cur_data["K_full_depth_b44"], 
                                        cur_data["cam_T_world_b44"], 
                                        color_frame,
                                        valid_mask_b
                                )
                    
                    '''
                    fuser.fuse_frames(
                                        upsampled_depth_pred_b1hw, 
                                        cur_data["invK_full_depth_b44"], 
                                        cur_data["world_T_camera_b44"], 
                                        color_frame
                                )
                    '''
                '''
                ########################### Quick Viz ##########################
                if opts.dump_depth_visualization:
                    # make a dir for this scan
                    output_path = os.path.join(viz_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    quick_viz_export(
                                output_path,
                                outputs, 
                                cur_data,
                                batch_ind, 
                                valid_mask_b,
                                opts.batch_size,
                            )
                ########################## Cache Depths ########################
                if opts.cache_depths:
                    output_path = os.path.join(depth_output_dir, scan)
                    Path(output_path).mkdir(parents=True, exist_ok=True)


                    cache_model_outputs(
                                output_path,
                                outputs, 
                                cur_data,
                                src_data,
                                batch_ind,
                                opts.batch_size,
                            )
                '''     

            # save the fused tsdf into a mesh file
            if opts.run_fusion:
                fuser.export_mesh(
                                    os.path.join(mesh_output_dir, 
                                        f"{scan.replace('/', '_')}.ply"),
                                )

            # compute a clean average
            scene_frame_metrics.compute_final_average()
            
            # one scene counts as a complete unit of metrics 
            all_scene_metrics.update_results(scene_frame_metrics.final_metrics)

            # print running metrics.
            print("\nScene metrics:")
            scene_frame_metrics.print_sheets_friendly(include_metrics_names=True)
            scene_frame_metrics.output_json(
                                os.path.join(scores_output_dir, 
                                    f"{scan.replace('/', '_')}_metrics.json")
                            )
            # print running metrics.
            print("\nRunning frame metrics:")
            all_frame_metrics.print_sheets_friendly(
                                                    include_metrics_names=False,
                                                    print_running_metrics=True,
                                                )

        # compute and print final average
        print("\nFinal metrics:")
        all_scene_metrics.compute_final_average()
        all_scene_metrics.pretty_print_results(print_running_metrics=False)
        all_scene_metrics.print_sheets_friendly(
                                                include_metrics_names=True, 
                                                print_running_metrics=False,
                                            )
        all_scene_metrics.output_json(
                                os.path.join(scores_output_dir, 
                                    f"all_scene_avg_metrics_{opts.split}.json")
                            )

        print("")
        all_frame_metrics.compute_final_average()
        all_frame_metrics.pretty_print_results(print_running_metrics=False)
        all_frame_metrics.print_sheets_friendly(
                                                include_metrics_names=True, 
                                                print_running_metrics=False
                                            )
        all_frame_metrics.output_json(
                                os.path.join(scores_output_dir, 
                                    f"all_frame_avg_metrics_{opts.split}.json")
                            )

if __name__ == '__main__':
    # don't need grad for test.
    torch.set_grad_enabled(False)
    
    sys.argv=['test_OmniScenes_all.py', '--name','PANO_MODEL', '--output_base_path', './output/OmniScenes/PanoGRF_all_bounds', '--config_file', 'configs/models/pano_model.yaml', '--load_weights_from_checkpoint', '/home/yzm/Workspace/simplerecon_v2/logs/PANO_MODEL/version_142/checkpoints/epoch=22-step=39191.ckpt', '--data_config', 'configs/data/OmniScenes_default_test.yaml',
     '--num_workers', '2','--batch_size', '2','--run_fusion','--depth_fuser','ours','--fuse_color','--dump_depth_visualization']
    
    """
    sys.argv=['test.py', '--name','PANO_MODEL', '--output_base_path', './output/S2D3D/v16', '--config_file', 'configs/models/pano_model.yaml', '--load_weights_from_checkpoint', '/home/yzm/Workspace/simplerecon_v2/logs/PANO_MODEL/version_122/checkpoints/epoch=17-step=133344.ckpt', '--data_config', 'configs/data/S2D3D_default_test.yaml',
     '--num_workers', '2','--batch_size', '2','--cache_depth']
    """
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)
