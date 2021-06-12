import argparse
import os
import requests
import zipfile
import tqdm
import shutil


def download_file(
    orig_url: str,
    dest_file: str,
    chunk_size: int = 2**20,
    verbose: bool = False
) -> None:

    """
    Downloads large file in chunks

    Arguments:
        orig_url: URL to original file

        dest_file: destination

        verbose: set True for status updates
    """

    if verbose:
        print('Start downloading file from: {}'.format(orig_url))

    with requests.get(orig_url, stream=True) as req:
        file_size = int(req.headers['content-Length'])
        num_bars = int(file_size / chunk_size)
        req.raise_for_status()

        with open(dest_file, 'wb') as f:
            for chunk in tqdm.tqdm(
                req.iter_content(chunk_size=chunk_size),
                total=num_bars,
                unit = 'KB',
                desc = 'Downloaded',
                leave = True):
                if chunk: # filters out keep alive new chunks
                    f.write(chunk)
    
    if verbose:
        print('Finsihed downloading file to: {}'.format(dest_file))


def unzip_file(
    file_path:str,
    unzip_dest: str,
    rgb_raw_zip: bool = False,
    keep_zip: bool = False
) -> None:

    """
    Unzips a file into a directory

    Arguments:
        file_path: path to .zip file to unzip

        unzip_dest: path to directory into which to expand
    """

    print('Unzipping: {}'.format(file_path))
    with zipfile.ZipFile(file_path, 'r') as z:
        for f in z.filelist:
            if rgb_raw_zip:
                if f.filename.endswith('.png'):
                    date, drive, img_idx, _, img_name = f.filename.split('/')
                    if img_idx in ['image_02', 'image_03']:
                        f.filename = os.path.basename(f.filename)
                        z.extract(f, '/'.join([unzip_dest, drive, 'image', img_idx]))
            else:
                z.extract(f, unzip_dest)
    
    if not keep_zip:
        os.remove(file_path)


def download_kitti(
    root_dir : str,
    keep_zip: bool = True
) -> None:

    """
    This function downloads the RGB images for the given depth maps.
    
    Arguments:
        root_dir : root directory, the dataset will be downloadeded and unzipped to:
                   <root_dir>/kitti_depth_completion

        keep_zip : Keep the original zip files after unzipping (stored in ./tmp/). Default: True (keep).
    """
 
    # URL's to .zip files
    kitti_raw_lidar_fname = 'data_depth_velodyne.zip'
    kitti_depth_annot_fname = 'data_depth_annotated.zip'
    kitti_test_set_fname = 'data_depth_selection.zip'
    kitti_calib_fnames = ['2011_09_26_calib.zip',
                            '2011_09_28_calib.zip',
                            '2011_09_29_calib.zip', 
                            '2011_09_30_calib.zip', 
                            '2011_10_03_calib.zip']
    
    kitti_base_url = 'http://s3.eu-central-1.amazonaws.com/avg-kitti'
    kitti_raw_data_base_url = '/'.join([kitti_base_url, 'raw_data'])
    kitti_raw_lidar_url = '/'.join([kitti_base_url, kitti_raw_lidar_fname])
    kitti_depth_annot_url = '/'.join([kitti_base_url, kitti_depth_annot_fname])
    kitti_test_set_url = '/'.join([kitti_base_url, kitti_test_set_fname])

    # Local path
    dest_dir = os.path.join(root_dir, 'kitti_depth_completion')

    # Check if output directory exist
    if not(os.path.isdir(dest_dir)):
        os.makedirs(dest_dir, exist_ok=True)
        print('Destination diectory does not exist, created new directory at: {}'.format(dest_dir))

    # Make temporary directory
    tmp_dir = '/'.join([dest_dir, 'tmp'])
    if not(os.path.isdir(tmp_dir)):
        os.makedirs(tmp_dir, exist_ok=True)

    # Directories name
    # lidar_raw_dir = 'velodyne_raw'
    # ground_truth_dir = 'groundtruth'
    rgb_dir = 'image'
    intrinsics_dir = 'intrinsics'
    
    # Raw lidar
    dest_zip_file = '/'.join([tmp_dir, kitti_raw_lidar_fname])

    if not os.path.isfile(dest_zip_file):
        # Check in case the files was donwloaded and previous run didn't complete.
        download_file(kitti_raw_lidar_url, dest_zip_file, verbose=True)
    else:
        print('Destination file already exist, skip downloading')

    unzip_file(dest_zip_file, dest_dir, keep_zip=keep_zip)
    
    # Groundtruh (annotations)
    dest_zip_file = '/'.join([tmp_dir, kitti_depth_annot_fname])

    if not os.path.isfile(dest_zip_file):
        # Check in case the files was donwloaded and previous run didn't complete.
        download_file(kitti_depth_annot_url, dest_zip_file, verbose=True)
    else:
        print('Destination file already exist, skip downloading')

    unzip_file(dest_zip_file, dest_dir, keep_zip=keep_zip)

    # Test/val selection data
    dest_zip_file = '/'.join([tmp_dir, kitti_test_set_fname])
    if not os.path.isfile(dest_zip_file):
        # Check in case the files was donwloaded and previous run didn't complete.
        download_file(kitti_test_set_url, dest_zip_file, verbose=True)
    else:
        print('Destination file already exist, skip downloading')

    unzip_file(dest_zip_file, dest_dir, keep_zip=keep_zip)
        
    # Move test set and validation select sets to place
    depth_sel_path = "/".join([dest_dir, 'depth_selection'])
    shutil.move("/".join([depth_sel_path, "val_selection_cropped"]), "/".join([dest_dir, "val_selection_cropped"]))
    shutil.move("/".join([depth_sel_path, "test_depth_completion_anonymous"]), "/".join([dest_dir, "test_depth_completion_anonymous"]))
    shutil.rmtree(depth_sel_path)

    # Calibration files
    calib_dest_dir = '/'.join([dest_dir, intrinsics_dir])

    for calib_file in kitti_calib_fnames:
        calib_file_url = '/'.join([kitti_raw_data_base_url, calib_file])
        dest_zip_file = '/'.join([tmp_dir, calib_file])
        if not os.path.isfile(dest_zip_file):
            download_file(calib_file_url, dest_zip_file, verbose=True)
        else:
            print('Destination file already exist')

        unzip_file(dest_zip_file, calib_dest_dir, keep_zip=keep_zip)
    
    # RGB images for train and validation 
    # (test and val select are downloaded with RGB iamges)
    print('Downloading RGB images\n')
    for split in ['train', 'val']:
        # List all depth maps
        split_root_path = '/'.join([dest_dir, split])
        depth_map_list = os.listdir(split_root_path)
        print(f'-Split type: {split}, Split size: {len(depth_map_list)} [depth map]\n')

        # Iterate over depth maps in split
        for depth_map_name in depth_map_list[0:1]:
            # Initialize
            zip_file_name = depth_map_name + '.zip'
            orig_file_url = '/'.join([kitti_raw_data_base_url, depth_map_name[:-5], zip_file_name])
            rgb_dest_path = '/'.join([split_root_path, depth_map_name, rgb_dir])
            dest_zip_file = '/'.join([tmp_dir, zip_file_name])
            tmp_depth_map_path = '/'.join([tmp_dir, depth_map_name])
            tmp_depth_map_img_path = '/'.join([tmp_depth_map_path, rgb_dir])

            # Check if destination file or .zip file already exist
            if os.path.isdir(rgb_dest_path):
                print('{} : destination files already exist, skipping\n'.format(depth_map_name))
                continue

            # Check if already exist
            if os.path.exists(dest_zip_file): 
                print('{} : file already exist, skipping\n'.format(depth_map_name))
            else:
                # Download
                download_file(orig_file_url, dest_zip_file, verbose=True)
                
            # Unzip
            unzip_file(dest_zip_file, tmp_dir, rgb_raw_zip=True, keep_zip=keep_zip)

            # Move images to correct directory
            shutil.move(tmp_depth_map_img_path, rgb_dest_path)
            shutil.rmtree(tmp_depth_map_path)
    
    # Remove temporary directory
    if not keep_zip:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KITTI depth completion dataset downloader.')
    parser.add_argument('-r', '--root', action='store',dest='root_dir', help='Root directory', required=True)
    parser.add_argument('-k', '--keep_zip', action='store_true', default=True, dest='keep_zip', help='keep downloaded zip files after extraction')
    args = parser.parse_args()

    download_kitti(root_dir=args.root_dir,
                     keep_zip=args.keep_zip)
