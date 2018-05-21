"""
Author: Jun Zhu, zhujun981661@gmail.com
"""
import os
import zipfile
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    """Show downloading progress"""
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg():
    """Download and extract pre-trained vgg model if it doesn't exist"""
    root_path = '../'
    vgg_folder = os.path.join(root_path, 'vgg')
    vgg_files = [
        os.path.join(vgg_folder, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_folder, 'variables/variables.index'),
        os.path.join(vgg_folder, 'saved_model.pb')
    ]

    vgg_zipfile = os.path.join(root_path, 'vgg.zip')

    missing_vgg_files = [vgg_file for vgg_file in vgg_files
                         if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        try:
            os.makedirs(vgg_folder)
        except FileExistsError:
            print("\033[31m" +
                  "Data path '{}' already exists! \n".format(vgg_folder) +
                  "Delete the folder before downloading data" +
                  "\033[0m")
            raise SystemExit

        # Download vgg
        if not os.path.exists(vgg_zipfile):
            print('Downloading pre-trained vgg model...')
            with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
                urlretrieve(
                    'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                    vgg_zipfile,
                    pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(vgg_zipfile, 'r')
        zip_ref.extractall(root_path)
        zip_ref.close()

        # Remove the zip file
        os.remove(vgg_zipfile)
