from datasets.matterport3d_dataset import Matterport3dDataset
from datasets.S2D3D_dataset import S2D3DDataset
from datasets.OmniScenes_dataset import OmniScenesDataset
def get_dataset(dataset_name, 
                split_filepath,
                single_debug_scan_id=None, 
                verbose=True):
    """ Helper function for passing back the right dataset class, and helps with
        itentifying the scans in a split file.
    
        dataset_name: a string pointing to the right dataset name, allowed names
            are:
                - scannet
                - arkit: arkit format as obtained and processed by NeuralRecon
                - vdr
                - scanniverse
                - colmap: colmap text format.
                - 7scenes: processed and undistorted seven scenes.
        split_filepath: a path to a text file that contains a list of scans that
            will be passed back as a list called scans.
        single_debug_scan_id: if not None will override the split file and will 
            be passed back in scans as the only item.
        verbose: if True will print the dataset name and number of scans.

        Returns:
            dataset_class: A handle to the right dataset class for use in 
                creating objects of that class.
            scans: a lit of scans in the split file.
    """
 
    
    #!Add new dataset matterport3d here. 
    if dataset_name == "matterport3d":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = Matterport3dDataset

        if verbose:    
            print(f"".center(80, "#"))
            print(f" Matterport3D Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    #!Add new dataset S2D3D here. 
    elif dataset_name == "S2D3D":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = S2D3DDataset

        if verbose:    
            print(f"".center(80, "#"))
            print(f" S2D3D Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")
            
    
    elif dataset_name == "OmniScenes":
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = OmniScenesDataset

        if verbose:    
            print(f"".center(80, "#"))
            print(f" OmniScenes Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")
            
            
            
    else:
        raise ValueError(f"Not a recognized dataset: {dataset_name}")
    

            
        


    return dataset_class, scans
