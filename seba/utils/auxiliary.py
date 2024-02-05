"""
@author: K. Danielewski
"""
import os
from glob import glob

def make_dir_save(save_path, name) -> str:
    """Auxfun. Checks if dir of joined path exists, if not makes dir, else outputs joined path
    """    
    if not os.path.exists(os.path.join(save_path, name)):
        os.mkdir(os.path.join(save_path, name))
        save_here = os.path.join(save_path, name)
    else:
        save_here = os.path.join(save_path, name)
    return save_here

def check_data_folder(data_folder) -> str:
    """Auxfun to check whether data folder was passed correctly.
    """    
    try:
        if isinstance(data_folder, list):
            pass
        elif isinstance(data_folder, str):
            data_folder = glob(data_folder + "\\*")
    except:
        print(f"Passed data folder should be either string or a list, {type(data_folder)} was passed")
    return data_folder
