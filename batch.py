import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import subprocess
import os

GENERATE_TYPE = 'both' # static, dynamic or both, or add _h5, or DoG, or MGF(multiple Gaussian Fitting)
SCRIPT = 'itti4video.py'
def get_filenames(directory):
    with os.scandir(directory) as entries:
        return [f"{entry.name}" for entry in entries if entry.is_file()]
    
def get_folder_names(directory):
    with os.scandir(directory) as entries:
        return [f"{entry.name}" for entry in entries if entry.is_dir()]
    
def run_script(paths):
    path,anotherpath = paths
    tag = path.split("\\")[-1].split("/")[-1].split(".")[0]
    generate_name = anotherpath#\\{tag}.mp4"
    try:
        subprocess.run(['python',SCRIPT, '--video_path',path,'--generate_name',generate_name])
    except Exception as e:
        print(f"\n An Error {e} happened in video {tag}!\n Check it later!\n")

if __name__ == '__main__':
    # 要处理的数据
    path = 'path/to/dataset'
    another_path = 'path/to/generate'
    if os.path.exists(f"{another_path}") == False:
        os.mkdir(f"{another_path}")
    data = get_filenames(path)# get_folder_names(path)# 
    another_data = get_filenames(another_path) # get_folder_names(another_path)
    datas = [[os.path.join(path,original_name), os.path.join(another_path, original_name.split(".")[0])] for original_name in data]
    datas = [paths for paths in datas if not os.path.exists(paths[1])]
    # datas = datas[:2]
    pool_size = 8
    while True:
        print(f'Using {pool_size} processes.')
        print(f'data[:2] = {datas[:2]}')
        print(f"len(data) = {len(datas)}, script = {SCRIPT}")
        a = input("move on?[y/n]\n")
        if a == "n":
            print("cancel.")
            exit()
        elif a == "y":
            print("move on.")
            break
        else:
            print("wrong input!")

    with multiprocessing.Pool(pool_size) as pool:
        results = list(tqdm(pool.imap(run_script, datas), total=len(datas)))
    
    warning = np.random.rand(100,100)
    plt.imshow(warning)
    plt.show()
