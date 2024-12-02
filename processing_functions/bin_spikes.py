#! /usr/bin/env python3.11

import h5py
import numpy as np
import pandas as pd
import sys
import os

top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(top_level_dir)
import analysis_package.maxlab_analysis as mla



def bin_spikes(filepath, filename, well_no, recording_no, data_path):

    bin_duration = 0.01 # this i guess is an argument that can be changed?

    with h5py.File(filepath + filename + ".raw.h5", "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        print(list(h5_object))
        #data = pd.DataFrame(np.array(h5_object["spikes"]))
        samp_rate = np.array(h5_object["settings"]["sampling"])[0]
        mapping = pd.DataFrame(np.array(h5_object["settings"]["mapping"]))

    data = mla.load_spikes_from_file(filepath + filename + ".raw.h5", well_no, recording_no)

    print(data)

    # data["frameno"] = data["frameno"] - data.loc[0, "frameno"]
    # FILTER OUT UNMAPPED CHANNELS
    # may not actually be necessary: Evan's bin spike data code does this for me.
    data = data.loc[data["channel"].isin(mapping["channel"]), :].reset_index(drop = True)


    raster_data_filename = data_path + filename + "_spike_raster.pkl" 
    spike_data_filename = data_path + filename + "_spike_data.pkl"
    times_filename = data_path + filename + "_times.npy"
    
    print(raster_data_filename)
    print(spike_data_filename)
    print(times_filename)

    raster_data, spike_data, times = mla.bin_spike_data(data, mapping, bin_size = bin_duration)
    raster_data.to_pickle(raster_data_filename)
    spike_data.to_pickle(spike_data_filename)
    np.save(times_filename, times)


if __name__ == "__main__":
    filepath = sys.argv[1]
    filename = sys.argv[2]
    well_no = int(sys.argv[3])
    recording_no = int(sys.argv[4])
    data_path = sys.argv[5]

    bin_spikes(filepath, filename, well_no, recording_no, data_path)
    