#! /usr/bin/env python3.11
import h5py
import config as config
import os
import shutil
import sys
import subprocess

from processing_functions.bin_spikes import *

SCRIPTS = [
    "bin_spikes.py"
    ]
    
def get_dirs(src):
    return [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]

def process_experiment_folder(src, dst):
    dirs = get_dirs(src)

    for exp in dirs:
        process_experiment(f"{src}{exp}/", f"{dst}{exp}/")

def process_experiment(src, dst):
    dirs = get_dirs(src)

    for chip in dirs:
        process_chip(f"{src}{chip}/", f"{dst}{chip}/")

def process_chip(src, dst):
    dirs = get_dirs(src)

    for date in dirs:
        process_date(f"{src}{date}/", f"{dst}{date}/")

def process_date(src, dst):
    dirs = get_dirs(src)
    

    for trial in dirs:
        trial_dst = f"{dst}{trial}/"
        if not os.path.exists(trial_dst):
            print(f"Processing trial {trial_dst}")    
            process_trial(f"{src}{trial}/", trial_dst)

# src is the full destination of the date of the trial
#       e.g. /SRC/parent/exp_name/chip_id/date/trial/
#
# dst is the 
#       e.g. /TARGET/parent/exp_name/chip_id/date/trial/
def process_trial(src, dst):
    wells = get_dirs(src)

    for well in wells:
        process_well(f"{src}{well}/", f"{dst}{well}/")

# src: path for the source of the well number
#       e.g. /SRC/parent/exp_name/chip_id/date/trial_no/well/
#
# dst: mirrored path of src for TARGET instead of SRC 
def process_well(src, dst):
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f.endswith(".raw.h5")]

    os.makedirs(dst, exist_ok=True)

    for file in files:
        process_h5(file, src, dst)

# file is the name of the h5 file to be processed
# src is the path to the folder where the file is stored
#       e.g. /SRC/parent/exp_name/chip_id/date/trial_no
# dst mirrors src, but for TARGET instead of SRC
def process_h5(file, src, dst):
    # file = os.path.basename(file_path)
    file_path = f"{src}{file}"

    name = file.removesuffix(".raw.h5")

    # file path, file name, well no, recording no, data path
    
    with h5py.File(file_path) as h5file:
        # get a list of the wells in h5
        wells = list(h5file['wells'])

        for well in wells:
            #print("making directories!")
            for i in range(len(SCRIPTS)):
                 # src directory, experiment name, well_no, 0, dest directory
                script = [f"./processing_functions/{SCRIPTS[i]}", src, name, str(well[len(well) - 1]), str(0), dst]
                try:
                    print(f"Running ./processing_functions/{SCRIPTS[i]}...")
                    subprocess.run(script)
                    print("great success")
                except Exception as err:
                    print(f"Error occurred while processing {file_path}:")
                    print(err)
            # print(f"{src}, {name}, {well[len(well) - 1]}, 0, {dst}")
            
            #bin_spikes(src, name, well[len(well) - 1], 0, dst)

    

# Processes all h5 files in source directory
if __name__=="__main__":
    # file_path = sys.argv[1]

    src = config.SOURCE
    # files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f.endswith(".raw.h5")]
    dirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and d != "processed"]

    for dir in dirs:
        print(f"Parsing {dir}")
        process_experiment_folder(f"{src}{dir}/", f"{config.TARGET}{dir}/")

    # for file in files:
    #     process_h5(file, config.SOURCE, config.TARGET)
    
    # exps = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]

    
    # print("Experiments to process:")
    # for dir in exps:
    #     print(f"\t{dir}")
    
    # for dir in exps:
    #     process_experiment(dir)

    