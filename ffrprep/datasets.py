import os
import shutil
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm


def download_unzip_exp_data(osf_url, dataset_path=None):
    """
    Download the example dataset from from OSF and unzip it.

    Parameters
    ----------
    osf_url : string
        The URL of the OSF file to download.
    dataset_path : string
        Path where the file will be saved. If None, the file will be saved
        in the current working directory. Default = None.

    Returns
    -------
    file_local : PosixPath or Pandas DataFrame
        PosixPath indicating the path to the downloaded file.

    Examples
    --------
    Download the file without specifying a path.

    >>> download_and_unzip_osf_file(osf_url, dataset_path=None)

    Download the file, specifying a path.

    >>> download_and_unzip_osf_file(osf_url, dataset_path='/home/user/Desktop')
    """

    # check if path where the file should be saved provided,
    # if not save it to the current directory
    if dataset_path is None:
        path = Path(os.curdir) / 'ffrprep_dataset'
    else:
        path = Path(dataset_path) / 'ffrprep_dataset'

    # in either case: check if path exists and if not, create it
    if not path.exists():
        os.makedirs(path)

    # Get the file name from the URL
    file_name = osf_url.split('/')[-1]
    zip_path = os.path.join(path, file_name)

    # if the file does not already exist, download it from osf
    if not Path(zip_path).exists():

        # provide a little update message
        print('Data will be downloaded to %s' % zip_path)

        # download and save the file, updating the user on the
        # download progress
        with requests.get(osf_url, stream=True) as file:

            # get total size of file for download updates
            file_size = int(file.headers.get('Content-Length'))

            # implement progress bar via tqdm
            with tqdm.wrapattr(file.raw, "read", total=file_size,
                               desc="") as raw:

                # save the output to the file specified before
                with open(zip_path, 'wb') as output:
                    shutil.copyfileobj(raw, output)

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)

    # Remove the zip file
    os.remove(zip_path)

    # Move the contents of the subdirectory to the main directory
    sub_dir_zip = Path(path / 'data-bids')

    # Loop over all items in subdirectory and move them to the main directory
    for item in sub_dir_zip.iterdir():
        new_location = path / item.name
        shutil.move(str(item), str(new_location))

    # Remove the empty subdirectory
    os.rmdir(sub_dir_zip)

    # Safely remove __MACOSX folder if it exists
    macosx_path = path / '__MACOSX'
    if macosx_path.exists():
        # Use rmtree to remove non-empty folders
        shutil.rmtree(macosx_path, ignore_errors=True)

    # return either the path to the dataset
    return path
