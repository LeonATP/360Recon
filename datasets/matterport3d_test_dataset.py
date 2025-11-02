
#!This code is modified from scannet_dataset.py!
import os
import pickle
import logging
import struct
import numpy as np
import PIL.Image as pil
import torch
from datasets.generic_mvs_dataset import GenericMVSDataset
from torchvision import transforms
from utils.generic_utils import (readlines, read_image_file)
from scipy.spatial.transform import Rotation as R
import cv2

logger = logging.getLogger(__name__)
class Matterport3dTestDataset(GenericMVSDataset):
    """ 
    MVS Matterport3D Dataset class for SimpleRecon.
    
    Inherits from GenericMVSDataset and implements missing methods. See 
    GenericMVSDataset for how tuples work. 

    This dataset expects ScanNetv2 to be in the following format:

    dataset_path
        scans_test (test scans)
            scene0707
                scene0707_00_vh_clean_2.ply (gt mesh)
                sensor_data
                    frame-000261.pose.txt
                    frame-000261.color.jpg 
                    frame-000261.color.512.png (optional, image at 512x384)
                    frame-000261.color.640.png (optional, image at 640x480)
                    frame-000261.depth.png (full res depth, stored scale *1000)
                    frame-000261.depth.256.png (optional, depth at 256x192 also
                                                scaled)
                scene0707.txt (scan metadata and intrinsics)
            ...
        scans (val and train scans)
            scene0000_00
                (see above)
            scene0000_01
            ....

    In this example scene0707.txt should contain the scan's metadata and 
    intrinsics:
        colorHeight = 968
        colorToDepthExtrinsics = 0.999263 -0.010031 0.037048 -0.038549 ........
        colorWidth = 1296
        depthHeight = 480
        depthWidth = 640
        fx_color = 1170.187988
        fx_depth = 570.924255
        fy_color = 1170.187988
        fy_depth = 570.924316
        mx_color = 647.750000
        mx_depth = 319.500000
        my_color = 483.750000
        my_depth = 239.500000
        numColorFrames = 784
        numDepthFrames = 784
        numIMUmeasurements = 1632
    
    frame-000261.pose.txt should contain pose in the form:
        -0.384739 0.271466 -0.882203 4.98152
        0.921157 0.0521417 -0.385682 1.46821
        -0.0587002 -0.961035 -0.270124 1.51837
    
    frame-000261.color.512.png is a precached resized version of the original 
    image to save load and compute time during training and testing. Similarly 
    for frame-000261.color.640.png. frame-000261.depth.256.png is also a 
    precached resized version of the depth map. 

    All resized precached versions of depth and images are nice to have but not 
    required. If they don't exist, the full res versions will be loaded, and 
    downsampled on the fly.

    NOTE: This dataset will place NaNs where gt depth maps are invalid.

    """

    def __init__(
            self,
            dataset_path,
            split,
            mv_tuple_file_suffix,
            include_full_res_depth=False,
            limit_to_scan_id=None,
            num_images_in_tuple=None,
            color_transform=transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            tuple_info_file_location=None,
            image_height=512,
            image_width=1024,
            high_res_image_width=1024,
            high_res_image_height=512,
            image_depth_ratio=2,
            shuffle_tuple=False,
            include_full_depth_K=False,
            include_high_res_color=False,
            pass_frame_id=False,
            skip_frames=None,
            skip_to_frame=None,
            verbose_init=True,
            min_valid_depth=1e-3,
            max_valid_depth=10,
        ):
        super().__init__(
                dataset_path=dataset_path,
                split=split, mv_tuple_file_suffix=mv_tuple_file_suffix, 
                include_full_res_depth=include_full_res_depth, 
                limit_to_scan_id=limit_to_scan_id,
                num_images_in_tuple=num_images_in_tuple, 
                color_transform=color_transform, 
                tuple_info_file_location=tuple_info_file_location, 
                image_height=image_height, image_width=image_width, 
                high_res_image_width=high_res_image_width, 
                high_res_image_height=high_res_image_height, 
                image_depth_ratio=image_depth_ratio, shuffle_tuple=shuffle_tuple, 
                include_full_depth_K=include_full_depth_K, 
                include_high_res_color=include_high_res_color, 
                pass_frame_id=pass_frame_id, skip_frames=skip_frames, 
                skip_to_frame=skip_to_frame, verbose_init=verbose_init,
            )

        """
        Args:
            dataset_path: base path to the dataaset directory.
            split: the dataset split.
            mv_tuple_file_suffix: a suffix for the tuple file's name. The 
                tuple filename searched for wil be 
                {split}{mv_tuple_file_suffix}.
            tuple_info_file_location: location to search for a tuple file, if 
                None provided, will search in the dataset directory under 
                'tuples'.
            limit_to_scan_id: limit loaded tuples to one scan's frames.
            num_images_in_tuple: optional integer to limit tuples to this number
                of images.
            image_height, image_width: size images should be loaded at/resized 
                to. 
            include_high_res_color: should the dataset pass back higher 
                resolution images.
            high_res_image_height, high_res_image_width: resolution images 
                should be resized if we're passing back higher resolution 
                images.
            image_depth_ratio: returned gt depth maps "depth_b1hw" will be of 
                size (image_height, image_width)/image_depth_ratio.
            include_full_res_depth: if true will return depth maps from the 
                dataset at the highest resolution available.
            color_transform: optional color transform that applies when split is
                "train".
            shuffle_tuple: by default source images will be ordered according to 
                overall pose distance to the reference image. When this flag is
                true, source images will be shuffled. Only used for ablation.
            pass_frame_id: if we should return the frame_id as part of the item 
                dict
            skip_frames: if not none, will stride the tuple list by this value.
                Useful for only fusing every 'skip_frames' frame when fusing 
                depth.
            verbose_init: if True will let the init print details on the 
                initialization.
            min_valid_depth, max_valid_depth: values to generate a validity mask
                for depth maps.
        
        """

        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

    @staticmethod
    def get_sub_folder_dir(split):
        """ Where scans are for each split. """
        #!We don't split in folder!
        '''
        if split == "test":
            return "scans_test"
        else:
            return "scans"
        '''
        return None
    
    def get_frame_id_string(self, frame_id):
        """ Returns an id string for this frame_id that's unique to this frame
            within the scan.

            This string is what this dataset uses as a reference to store files 
            on disk.
        """
        return frame_id

    def get_valid_frame_path(self, split, scan):
        """ returns the filepath of a file that contains valid frame ids for a 
            scan. """

        scan_dir = os.path.join(self.dataset_path, 
                            self.get_sub_folder_dir(split), scan)

        return os.path.join(scan_dir, "valid_frames.txt")

    def get_valid_frame_ids(self, split, scan, store_computed=True):
        """ Either loads or computes the ids of valid frames in the dataset for
            a scan.
            
            A valid frame is one that has an existing RGB frame, an existing 
            depth file, and existing pose file where the pose isn't inf, -inf, 
            or nan.

            Args:
                split: the data split (train/val/test)
                scan: the name of the scan
                store_computed: store the valid_frame file where we'd expect to
                see the file in the scan folder. get_valid_frame_path defines
                where this file is expected to be. If the file can't be saved,
                a warning will be printed and the exception reason printed.

            Returns:
                valid_frames: a list of strings with info on valid frames. 
                Each string is a concat of the scan_id and the frame_id.
        """
        scan = scan.rstrip("\n")
        valid_frame_path = self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            # valid frame file exists, read that to find the ids of frames with 
            # valid poses.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:
            # find out which frames have valid poses 

            #get scannet directories
            scan_dir = os.path.join(self.dataset_path, 
                            self.get_sub_folder_dir(split), scan)
            sensor_data_dir = os.path.join(scan_dir, "sensor_data")
            meta_file_path = os.path.join(scan_dir, scan + ".txt")
            
            with open(meta_file_path, 'r') as f:
                meta_info_lines = f.readlines()
                meta_info_lines = [line.split(' = ') for line in 
                                                        meta_info_lines]
                meta_data = {key: val for key, val in meta_info_lines}

            # fetch total number of color files
            color_file_count = int(meta_data["numColorFrames"].strip())

            dist_to_last_valid_frame = 0
            bad_file_count = 0
            valid_frames = []
            for frame_id in range(color_file_count):
                # for a frame to be valid, we need a valid pose and a valid 
                # color frame.

                color_filename = os.path.join(sensor_data_dir, 
                                            f"frame-{frame_id:06d}.color.jpg")
                depth_filename = color_filename.replace(f"color.jpg", 
                                                        f"depth.png")
                pose_path = os.path.join(sensor_data_dir, 
                                            f"frame-{frame_id:06d}.pose.txt")

                # check if an image file exists.
                if not os.path.isfile(color_filename):
                    dist_to_last_valid_frame+=1
                    bad_file_count+=1
                    continue
                
                # check if a depth file exists.
                if not os.path.isfile(depth_filename):
                    dist_to_last_valid_frame+=1
                    bad_file_count+=1
                    continue
                
                world_T_cam_44 = np.genfromtxt(pose_path).astype(np.float32)
                # check if the pose is valid.
                if (np.isnan(np.sum(world_T_cam_44)) or 
                    np.isinf(np.sum(world_T_cam_44)) or 
                    np.isneginf(np.sum(world_T_cam_44))
                ):
                    dist_to_last_valid_frame+=1
                    bad_file_count+=1
                    continue

                valid_frames.append(f"{scan} {frame_id:06d} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(f"Scene {scan} has {bad_file_count} bad frame files out of "
                  f"{color_file_count}.")

            # store computed if we're being asked, but wrapped inside a try 
            # incase this directory is read only.
            if store_computed:
                # store those files to valid_frames.txt
                try:
                    with open(valid_frame_path, 'w') as f:
                        f.write('\n'.join(valid_frames) + '\n')
                except Exception as e:
                    print(f"Couldn't save valid_frames at {valid_frame_path}, "
                        f"cause:\n", e)

        return valid_frames

    @staticmethod
    def get_gt_mesh_path(dataset_path, split, scan_id):
        """ 
        Returns a path to a gt mesh reconstruction file.
        """
        gt_path = os.path.join(
                        dataset_path,
                        Matterport3dTestDataset.get_sub_folder_dir(split),
                        scan_id,
                        f'{scan_id}_vh_clean_2.ply',
                    )
        return gt_path

    def get_color_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's color file at the dataset's 
            configured RGB resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the size 
                required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """
        scene_path = os.path.join(self.scenes_path, scan_id)
        '''
        sensor_data_dir = os.path.join(scene_path, "sensor_data")

        cached_resized_path = os.path.join(sensor_data_dir, 
                            f"frame-{frame_id}.color.{self.image_width}.png")
        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
        
        # instead return the default image
        '''
        #return os.path.join(sensor_data_dir, f"frame-{frame_id}.color.jpg")
        return os.path.join(scene_path,f"frame_{frame_id}.png")

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's higher res color file at the 
            dataset's configured high RGB resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the high res 
                size required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """
        
        scene_path = os.path.join(self.scenes_path, scan_id)
        '''
        sensor_data_dir = os.path.join(scene_path, "sensor_data")
        
        cached_resized_path = os.path.join(sensor_data_dir, 
                f"frame-{frame_id}.color.{self.high_res_image_height}.png")
        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
        
        # instead return the default image
        '''
        return os.path.join(scene_path, f"frame_{frame_id}.png")

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the dataset's 
            configured depth resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Filepath for a precached depth file at the size 
                required.

        """
        scene_path = os.path.join(self.scenes_path, scan_id)
        #sensor_data_dir = os.path.join(scene_path, "sensor_data")

        cached_resized_path = os.path.join(scene_path, 
                f"frame_{frame_id}.dpt")
        
        # instead return the default image
        return cached_resized_path

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the native 
            resolution in the dataset.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached depth file at the size 
                required, or if that doesn't exist, the full size depth frame 
                from the dataset.

        """
        scene_path = os.path.join(self.scenes_path, scan_id)
        #sensor_data_dir = os.path.join(scene_path, "sensor_data")

        return os.path.join(scene_path, 
                        f"frame_{frame_id}.dpt")

    def get_pose_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's pose file.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Filepath for pose information.

        """

        scene_path = os.path.join(self.scenes_path, scan_id)
        #sensor_data_dir = os.path.join(scene_path, "sensor_data")

        return os.path.join(scene_path, f"frame_{frame_id}.txt")

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """ Loads intrinsics, computes scaled intrinsics, and returns a dict 
            with intrinsics matrices for a frame at multiple scales.
            
            ScanNet intrinsics for color and depth are the same up to scale.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame. Not needed for ScanNet as images 
                share intrinsics across a scene.
                flip: flips intrinsics along x for flipped images.

            Returns:
                output_dict: A dict with
                    - K_s{i}_b44 (intrinsics) and invK_s{i}_b44 
                    (backprojection) where i in [0,1,2,3,4]. i=0 provides
                    intrinsics at the scale for depth_b1hw. 
                    - K_full_depth_b44 and invK_full_depth_b44 provides 
                    intrinsics for the maximum available depth resolution.
                    Only provided when include_full_res_depth is true. 
            
        """
        #!For equirectangular images, the intrinsics is very different, and all the same!
        output_dict = {}

        '''
        scene_path = os.path.join(self.scenes_path, scan_id)
        metadata_filename = os.path.join(scene_path, f"{scan_id}.txt")
        
        # load in basic intrinsics for the full size depth map.
        lines = readlines(metadata_filename)
        lines = [line.split(' = ') for line in lines]
        data = {key: val for key, val in lines}

        intrinsics_filepath = os.path.join(scene_path, "intrinsic", "intrinsic_depth.txt")
        
        K = torch.tensor(np.genfromtxt(intrinsics_filepath).astype(np.float32))
        '''
        
        image_width =self.image_width
        image_height = self.image_height

        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(image_width/(2*(np.pi)))
        K[1, 1] = float(-image_height/np.pi)
        K[0, 2] = float(image_width/2)
        K[1, 2] = float(image_height/2)

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:   
            full_K = K.clone()
            
            
            full_K[0] *= (self.native_depth_width/image_width) 
            full_K[1] *= (self.native_depth_height/image_height) 
            
            output_dict[f"K_full_depth_b44"] = full_K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.linalg.inv(full_K)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= (self.depth_width/image_width) 
        K[1] *= (self.depth_height/image_height)


        # Get the intrinsics of all the scales
        for i in range(5):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict

    def load_hinter_depth(self,path,
                          height=None, 
                          width=None,
                          resampling_mode=cv2.INTER_LINEAR,
                          disable_warning=False,
                          target_aspect_ratio=None):
        '''
        Loads depth image from the '.dpt' format used by Hinterstoisser.
        '''
        INT_BYTES = 4

        with open(path, 'rb') as f:
            _ = struct.unpack('i', f.read(INT_BYTES))[0]  # Skipping the first integer
            w = struct.unpack('i', f.read(INT_BYTES))[0]  # Width of the image
            h = struct.unpack('i', f.read(INT_BYTES))[0]  # Height of the image

            # Read the entire depth map in one go
            depth_map = np.fromfile(f, dtype=np.float32, count=w * h).reshape((h, w))
            if height is not None and width is not None:
               if (w, h) != (width, height):
             # warn if it doesn't make sense.
                 if ((width > w or height > h) and not disable_warning):
                    logger.warning(
                        f"WARNING: target size ({width}, {height}) has a "
                        f"dimension larger than input size ({w}, "
                        f"{h}).")
               depth_map = cv2.resize(depth_map, (width, height), interpolation=resampling_mode)

            depth_map_tensor = torch.from_numpy(depth_map)
            
        return depth_map_tensor

    
    
    
    def load_target_size_depth_and_mask(self, scan_id, frame_id):
        """ Loads a depth map at the resolution the dataset is configured for.

            Internally, if the loaded depth map isn't at the target resolution,
            the depth map will be resized on-the-fly to meet that resolution.

            NOTE: This function will place NaNs where depth maps are invalid.

            Args:
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                depth: depth map at the right resolution. Will contain NaNs 
                    where depth values are invalid.
                mask: a float validity mask for the depth maps. (1.0 where depth
                is valid).
                mask_b: like mask but boolean.
        """
        depth_filepath = self.get_cached_depth_filepath(scan_id, frame_id)

        if not os.path.exists(depth_filepath):
            depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)

        # Load depth, resize
        #!We should use the correct function to load the GT depth for matterport3d.
        '''
        depth = read_image_file(
                                depth_filepath,
                                height=self.depth_height,
                                width=self.depth_width,
                                value_scale_factor=1e-3,
                                resampling_mode=pil.NEAREST,
                            )
        '''
        depth = self.load_hinter_depth(depth_filepath,height=self.depth_height,width=self.depth_width,resampling_mode=cv2.INTER_NEAREST)
        depth = depth.unsqueeze(0)
        # Get the float valid mask
        mask_b = ((depth > self.min_valid_depth) 
                                & (depth < self.max_valid_depth))
        mask = mask_b.float()

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)
        
        return depth, mask, mask_b
    
    
    def load_pred_depth(self, scan_id, frame_id):
        #scene_path = os.path.join("/home/yzm/Workspace/PanoFormer/PanoFormer/output/depth/", scan_id)
        scene_path = os.path.join("/home/yzm/Workspace/BiFusev2/output/depth/", scan_id)
        #sensor_data_dir = os.path.join(scene_path, "sensor_data")
        #scene_path = os.path.join("/home/yzm/Workspace/fova-depth/test_logs/matterport360_cube/lightning_logs/version_20/visuals/new", scan_id)
        #scene_path = os.path.join("/home/yzm/Workspace/PanoGRF/logs/mvsdepth/new_new", scan_id)
        cached_resized_path = os.path.join(scene_path, 
                f"frame_{frame_id}_depth.pkl")
                #f"frame_{frame_id}.pkl")
                #frame_id,
                #'original_depth.pkl')
                #f"{frame_id}.npz")
        with open(cached_resized_path, 'rb') as f:
            depth_pred = pickle.load(f)
            #depth_pred = np.load(f)['pred']
            #depth_pred = depth_pred.transpose(2, 0, 1)
        # Ensure depth data is a NumPy array
        if not isinstance(depth_pred, np.ndarray):
            depth_pred = np.array(depth_pred)
        depth_pred = torch.from_numpy(depth_pred)
        depth_pred = depth_pred.unsqueeze(0)

        
        return depth_pred

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """ Loads a depth map at the native resolution the dataset provides.

            NOTE: This function will place NaNs where depth maps are invalid.

            Args:
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
                
            Returns:
                full_res_depth: depth map at the right resolution. Will contain 
                    NaNs where depth values are invalid.
                full_res_mask: a float validity mask for the depth maps. (1.0 
                where depth is valid).
                full_res_mask_b: like mask but boolean.
        """
        full_res_depth_filepath = self.get_full_res_depth_filepath(
                                                    scan_id, frame_id)
        '''
        # Load depth
        full_res_depth = read_image_file(full_res_depth_filepath, 
                                                value_scale_factor=1e-3)
        '''
        full_res_depth = self.load_hinter_depth(full_res_depth_filepath,height=self.native_depth_height,width=self.native_depth_width,resampling_mode=cv2.INTER_NEAREST)
        # Get the float valid mask
        full_res_mask_b = ((full_res_depth > self.min_valid_depth) 
                                & (full_res_depth < self.max_valid_depth))
        full_res_mask = full_res_mask_b.float()

        # set invalids to nan
        full_res_depth[~full_res_mask_b] = torch.tensor(np.nan)

        return full_res_depth, full_res_mask, full_res_mask_b

    def load_pose(self, scan_id, frame_id):
        """ Loads a frame's pose file.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                world_T_cam (numpy array): matrix for transforming from the 
                    camera to the world (pose).
                cam_T_world (numpy array): matrix for transforming from the 
                    world to the camera (extrinsics).

        """
        pose_path = self.get_pose_filepath(scan_id, frame_id)

        #world_T_cam = np.genfromtxt(pose_path).astype(np.float32)
        with open(pose_path, "r") as f:
           pose = list(map(float, f.read().splitlines()[0].split(" ")))

        C = np.array(pose[:3])
        rot = R.from_quat(pose[3:7]).as_matrix()
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        t = C.reshape([3,1])
        world_T_cam = np.concatenate([np.concatenate([rot, t], 1), bottom], 0)
        gl_to_cv = np.array([[1,  0,  0, 0],
                            [0, -1,  0, 0],
                            [0,  0, -1, 0],
                            [0,  0,  0, 1]], dtype=np.float32)
        world_T_cam = world_T_cam@gl_to_cv
        #world_T_cam = world_T_cam.astype(np.float32)
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world