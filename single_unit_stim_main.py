from analysis_package import *

from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import euclidean
from random import choice

import h5py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec


def load_data(path: str, well_no: int, recording_no: int, voltage_threshold: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    spikes = load_spikes_from_file(path, well_no, recording_no, voltage_threshold)
    with h5py.File(path, "r") as h5_file:
        mapping_h5 = h5_file['wells'][f'well{well_no:0>3}'][f'rec{recording_no:0>4}']['settings']['mapping']
        events_path = f'/wells/well{well_no:0>3}/rec{recording_no:0>4}/events'
        frameno = h5_file[events_path]['frameno'][:]
        event_times = frameno / 10000.0

        return spikes, pd.DataFrame(np.array(mapping_h5)), event_times


def calc_ham_and_phys_dists(df: pd.DataFrame, mapping: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    
    chan_positions = mapping.set_index('channel')[['x', 'y']].to_dict('index')
    num_chans = df.shape[1]
    hamming_distances = []
    physical_distances = []

    total_ops = num_chans * (num_chans - 1) // 2

    with tqdm(total=total_ops, desc="Calculating distances") as pbar:
        for i, col_i in enumerate(df.columns):
            for j, col_j in enumerate(df.columns[i+1:], i+1):
                pos_i = chan_positions.get(col_i)
                pos_j = chan_positions.get(col_j)

                phys_dist = euclidean((pos_i['x'], pos_i['y']), (pos_j['x'], pos_j['y'])) # physical distance
                physical_distances.append(phys_dist)

                spike_mask = (df[col_i] == 1) | (df[col_j] == 1)
                ham_dist = np.sum(df.loc[spike_mask, col_i] != df.loc[spike_mask, col_j]) # hamming distance
                hamming_distances.append(ham_dist)

                pbar.update(1)

    return np.array(hamming_distances), np.array(physical_distances), zero_column_titles


def plot_hamming_distances(distances: np.ndarray) -> None:

    plt.figure(figsize=(10, 8), dpi=100)
    _, bins, patches = plt.hist(distances, bins=128)
    
    cmap = mcolors.LinearSegmentedColormap.from_list("black_to_blue", [(0, 0, 0), (0, 0, 1)])
    
    norm = mcolors.Normalize(vmin=bins.min(), vmax=bins.max())
    
    for bin_start, bin_end, patch in zip(bins[:-1], bins[1:], patches):
        color = cmap(norm((bin_start + bin_end) / 2))
        patch.set_facecolor(color)
    
    plt.title('Hamming Distances')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    
    plt.savefig(f"{plts_path}/ham_dists.png", dpi=1200)
    plt.close()


def ham_and_phys_scat_plot(phys_dists: np.ndarray, ham_dists: np.ndarray, plts_path: str) -> None:
    
    min_length = min(len(phys_dists), len(ham_dists))
    phys_dists = phys_dists[:min_length]
    ham_dists = ham_dists[:min_length]
    
    norm_phys = plt.Normalize(phys_dists.min(), phys_dists.max())
    norm_ham = plt.Normalize(ham_dists.min(), ham_dists.max())

    plt.figure(figsize=(10, 8), dpi=100)
    
    colors = np.zeros((len(phys_dists), 3))
    colors[:, 0] = norm_phys(phys_dists)
    colors[:, 2] = norm_ham(ham_dists)
    
    plt.scatter(phys_dists, ham_dists, c=colors, alpha=0.6, s=8)

    plt.title('Hamming vs. Physical Distance')
    plt.xlabel('Physical Distance (μm)')
    plt.ylabel('Hamming Distance')

    plt.grid(True, linestyle='--', alpha=0.7)

    ax = plt.gca()
    #ax.text(0.05, 0.98, 'Red: High Physical Distance', transform=ax.transAxes, verticalalignment='top', color='red')
    #ax.text(0.35, 0.98, 'Blue: High Hamming Distance', transform=ax.transAxes, verticalalignment='top', color='blue')
    #ax.text(0.65, 0.98, 'Purple/Magenta: High in Both', transform=ax.transAxes, verticalalignment='top', color='purple')

    plt.savefig(os.path.join(plts_path, 'scat_plot.png'), dpi=1200, bbox_inches='tight')
    plt.close()


def cofiring_probability(df: pd.DataFrame, chan_id: int, pre_win: int, post_win: int) -> np.ndarray:
    
    if chan_id not in df.columns:
        raise ValueError(f"Channel {chan_id} not found in DataFrame columns.")
    
    spike_bin = df.to_numpy().T
    num_chan, num_time_bins = spike_bin.shape

    i = df.columns.get_loc(chan_id)

    total_dur = pre_win + post_win + 1
    allall = np.nan * np.ones((num_chan, total_dur))

    wherei = np.where(spike_bin[i, :] == 1)[0]

    if len(wherei) == 0:
        return allall

    for j in range(num_chan):
        wins = np.zeros((len(wherei), total_dur))
        for x, spike_time in enumerate(wherei):
            start_index = max(spike_time - pre_win, 0)
            end_index = min(spike_time + post_win + 1, num_time_bins)
            window_size = end_index - start_index
            wins[x, :window_size] = spike_bin[j, start_index:end_index]
        allall[j, :] = np.sum(wins, axis=0) / len(wherei)

    return allall


def plot_cofiring_prob_one_chan(cofire_matrix: np.ndarray, plts_path: str, chan: int) -> None:

    total_dur = cofire_matrix.shape[1]
    half_dur = (total_dur - 1) // 2
    x_ticks = np.arange(-half_dur, half_dur + 1)
    avg_probability = np.mean(cofire_matrix, axis=0)

    plt.figure(figsize=(8, 20), dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)

    ax0 = plt.subplot(gs[0])
    cax = ax0.imshow(cofire_matrix, aspect='auto', cmap='viridis', extent=[x_ticks[0], x_ticks[-1], cofire_matrix.shape[0], 0], interpolation='none')
    cbaxes = plt.gcf().add_axes([0.15, 0.92, 0.7, 0.02])
    plt.colorbar(cax, cax=cbaxes, orientation='horizontal', label='')
    ax0.set_xticks([])
    ax0.set_ylabel('Channel j')
    ax0.set_title(f'Cofiring Probability for Channel {chan}')
    ax0.invert_yaxis()

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(x_ticks, avg_probability, 'b-')
    ax1.set_ylabel('Avg Probability')
    ax1.set_xlabel('Time from Spike i, ms')
    ax1.set_title(f'Average Cofiring Probability for Channel {chan}')
    slice_step = max(1, len(x_ticks) // 10)
    ax1.set_xticks(x_ticks[::slice_step])
    ax1.set_ylim(0, ax1.get_ylim()[1])
    ax1.grid(True)

    plt.setp(ax0.get_xticklabels(), rotation=45)
    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.savefig(f"{plts_path}/cofiring_with_avg_vertical", dpi=1200)
    plt.close()


def calc_avg_probs(binned_df: pd.DataFrame, zero_column_titles: list, pre_win: int, post_win: int) -> pd.DataFrame:
    
    avg_probs_dict = {}
    chan_titles = binned_df.columns

    with tqdm(total=len(chan_titles) - len(zero_column_titles), desc="Calculating avg probs all chans") as pbar:
        for i in chan_titles:
            if i not in zero_column_titles:
                allall = cofiring_probability(df=binned_df, chan_id=i, pre_win=pre_win, post_win=post_win)
                avg_probability = np.mean(allall, axis=0)
                avg_probs_dict[i] = avg_probability
                pbar.update(1)
        pbar.close()

    avg_probs_df = pd.DataFrame.from_dict(avg_probs_dict, orient='index')
    return avg_probs_df


def plot_avg_probs_heatmap_all_chans(avg_probs_df: pd.DataFrame, plts_path: str) -> None:

    total_dur = avg_probs_df.shape[1]
    half_dur = (total_dur - 1) // 2
    x_ticks = np.arange(-half_dur, half_dur + 1)

    slice_step = max(1, len(x_ticks) // 10)

    plt.figure(figsize=(10, 8), dpi=100)
    heatmap = plt.imshow(avg_probs_df.values, aspect='auto', cmap='viridis', interpolation='none')

    plt.colorbar(heatmap, label='Average Probability')

    plt.xticks(np.arange(0, len(x_ticks), slice_step), x_ticks[::slice_step], rotation=45)

    plt.ylabel('Channel j')
    plt.xlabel('Time Bins from Spike i, ms')
    plt.title('Average Probability Heatmap All Channels')

    plt.tight_layout()
    plt.savefig(f"{plts_path}/avg_prob_heatmap_all_chans", dpi=1200)
    plt.close()


def calc_peak_binary_matrix(avg_probs_df: pd.DataFrame) -> np.ndarray:

    num_chans, num_time_bins = avg_probs_df.shape
    binary_matrix = np.zeros((num_chans, num_time_bins))

    for i in range(num_chans):
        peak_index = np.argmax(avg_probs_df.iloc[i, :])
        binary_matrix[i, peak_index] = 1

    return binary_matrix


def plot_avg_probs_sort_by_peaks(binary_matrix: np.ndarray, avg_probs_df: pd.DataFrame, plts_path: str) -> None:

    peak_indices = np.argmax(binary_matrix, axis=1)
    sort_indices = np.argsort(peak_indices)
    sort_avg_probs_df = avg_probs_df.iloc[sort_indices, :]
    sort_avg_probs_array = sort_avg_probs_df.to_numpy()

    total_dur = sort_avg_probs_array.shape[1]
    half_dur = (total_dur - 1) // 2
    x_ticks = np.arange(-half_dur, half_dur + 1)
    slice_step = max(1, len(x_ticks) // 10)

    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(sort_avg_probs_array, aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(heatmap, label='Average Probability')
    plt.xticks(np.arange(0, len(x_ticks), slice_step), x_ticks[::slice_step], rotation=45)
    plt.yticks([])
    plt.ylabel('Channel j (sort by peak avg probability)')
    plt.xlabel('Time Bins from Spike i, ms')
    plt.title('Average Probability Heatmap All Channels (Sorted)')
    plt.savefig(f"{plts_path}/avg_prob_heatmap_all_chans_sort", dpi=1200)
    plt.close()


def sort_avg_probs_by_phys_dist(binned_df: pd.DataFrame, zero_column_titles: list, mapping: pd.DataFrame, avg_probs_df: pd.DataFrame, ref_chan: int) -> Tuple[pd.DataFrame, list]:
    
    chan_positions = mapping.set_index('channel')[['x', 'y']].to_dict('index')
    ref_coords = (chan_positions[ref_chan]['x'], chan_positions[ref_chan]['y'])
    chan_titles = [chan for chan in binned_df.columns if chan not in zero_column_titles]
    physical_distances = {chan: euclidean((chan_positions[chan]['x'], chan_positions[chan]['y']), ref_coords) for chan in chan_titles}
    
    sort_chans = sorted(physical_distances, key=physical_distances.get)
    sort_distances = [physical_distances[chan] for chan in sort_chans]
    
    sort_indices = [chan_titles.index(chan) for chan in sort_chans]
    
    sort_avg_probs_df = avg_probs_df.iloc[sort_indices, :]
    
    return sort_avg_probs_df, sort_distances


def plot_avg_probs_heatmap_sort_by_phys_dist(sort_avg_probs_df: pd.DataFrame, distances: list, plts_path: str, ref_chan: int) -> None:

    sort_avg_probs_array = sort_avg_probs_df.to_numpy()

    total_dur = sort_avg_probs_array.shape[1]
    half_dur = (total_dur - 1) // 2
    x_ticks = np.arange(-half_dur, half_dur + 1)
    slice_step = max(1, len(x_ticks) // 10)

    plt.figure(figsize=(10, 8), dpi=100)
    heatmap = plt.imshow(sort_avg_probs_array, aspect='auto', cmap='viridis', interpolation='none')

    plt.colorbar(heatmap, label='Average Probability')

    plt.xticks(np.arange(0, len(x_ticks), slice_step), x_ticks[::slice_step], rotation=45)
    
    y_ticks = np.arange(0, len(distances), 50)
    y_labels = [f"{distances[i]:.2f}" for i in y_ticks]
    plt.yticks(y_ticks, y_labels)

    plt.ylabel(f'Distance from chan {ref_chan} (μm)')
    plt.xlabel('Time Bins from Spike i, ms')
    plt.title(f'Average Probability Heatmap Sorted by Distance, Chan {ref_chan} ref')

    plt.savefig(f"{plts_path}/avg_prob_heatmap_sort_by_dist.png", dpi=1200)
    plt.close()


def calc_cofire_integrals(avg_probs_df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:

    total_dur = avg_probs_df.shape[1]
    half_dur = (total_dur - 1) // 2
    time_bins = np.arange(-half_dur, half_dur + 1)

    integrals = avg_probs_df.values @ time_bins

    status = np.where(integrals < 0, "inhibitory", "excitatory")

    results_df = pd.DataFrame({
        'channel': avg_probs_df.index,
        'integral': integrals,
        'status': status
    })

    results_df = results_df.merge(mapping, left_on='channel', right_on='channel')

    return results_df


def plot_chip_visualization(results_df: pd.DataFrame, plts_path: str, chip_div_well: str) -> None:
    
    x_coords = results_df['x']
    y_coords = results_df['y']
    integral_values = results_df['integral']

    norm = plt.Normalize(vmin=-3, vmax=3)
    
    colors = [(1, 0, 0), (1, 0.8, 0.8), (0.8, 0.8, 1), (0, 0, 1)]
    positions = [0, 0.5, 0.5, 1]
    cmap = LinearSegmentedColormap.from_list("custom_red_blue", list(zip(positions, colors)))

    plt.figure(figsize=(20, 8), facecolor='black')
    scatter = plt.scatter(x_coords, y_coords, c=integral_values, cmap=cmap, s=50, alpha=1, edgecolors='none', norm=norm)

    cbar = plt.colorbar(scatter, pad=0.01, aspect=45)
    cbar.set_label('Avg Cofiring Probability Integral Value', color='white')

    cbar.set_ticks(np.linspace(-3, 3, num=5))
    cbar.set_ticklabels([f'{x:.2f}' for x in np.linspace(-3, 3, num=5)])
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.set_tick_params(labelcolor='white')

    plt.xlabel('X Coordinate (µm)', color='white')
    plt.ylabel('Y Coordinate (µm)', color='white')
    plt.title('Chip Visualization', color='white', fontsize=16, fontstyle='italic')

    plt.gca().set_facecolor('black')
    plt.gca().tick_params(colors='white')

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlim(-50, 3875)
    plt.ylim(-50, 2125)

    inhibitory_count = (results_df['status'] == "inhibitory").sum()
    excitatory_count = (results_df['status'] == "excitatory").sum()

    plt.text(-0.05, 1.0322, f'Inhibitory (red): {inhibitory_count}', transform=plt.gca().transAxes, color='white', fontsize=12, verticalalignment='top', horizontalalignment='left')
    plt.text(0.12, 1.0322, f'Excitatory (blue): {excitatory_count}', transform=plt.gca().transAxes, color='white', fontsize=12, verticalalignment='top', horizontalalignment='left')

    plt.text(0.64, 1.0322, chip_div_well, transform=plt.gca().transAxes, color='white', fontsize=12, verticalalignment='top', horizontalalignment='left')

    plt.savefig(f"{plts_path}/chip_vis.png", dpi=1200, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":

    # TODO: change this stuff
    # electrodes
    inhib_targets = [21531, 22855, 25903]
    excite_targets = [7455, 6385, 21415]
    electrode_targets = inhib_targets + excite_targets

    # TODO: labels for the indexes of the binned df pieces ~ pre | stim | post/pre | stim2 | post2 ... only 4 markers, 5 pieces
    # these will be used as folder names***, so make sure not to override! or... use slashes!
    a_list = ['Pre', 'Stim', 'Post~Pre', 'Stim2', 'Post2']

    data_path_list = []

    # TODO: change this stuff
    for root, dirs, files in os.walk('C:/Users/evank/OneDrive/Documents/HALnalysis_summer_2024_data/data/single_unit/240821'):
        for data_path in files:
            if data_path.endswith('.h5'):
                full_path = os.path.join(root, data_path)
                data_path_list.append(full_path)

    print(data_path_list)

    for j, data_path in enumerate(data_path_list):
            
        well_no = 0
        recording_no = 0
        voltage_threshold = None
        # TODO: change this stuff
        chip_div_well_base = f'M07480 | WELL{well_no} | {"Inhib Target Stim" if electrode_targets[j] in inhib_targets else "Excite Stim" if electrode_targets[j] in excite_targets else "The file was set up wrong bro"}'

        print(data_path)
        print(well_no)

        spike_pd_df, mapping_pd_df, event_times = load_data(data_path, well_no, recording_no, voltage_threshold)

        print(spike_pd_df)
        print(mapping_pd_df)
        print('event times' + str(event_times))

        binned_df, spike_binned_data_df, _ = bin_spike_data(spike_df=spike_pd_df, mapping=mapping_pd_df, bin_size=0.01, mode='binary')

        print(binned_df)
        print(binned_df.shape)
        print(spike_binned_data_df)
        print(spike_binned_data_df.shape)

        split_dfs = []
        split_dfs.append(binned_df[binned_df.index < event_times[0]])
        for start, end in zip(event_times[:-1], event_times[1:]):
            split_df = binned_df[(binned_df.index >= start) & (binned_df.index < end)]
            split_dfs.append(split_df)
        split_dfs.append(binned_df[binned_df.index >= event_times[-1]])
        
        print(split_dfs)
        print(len(split_dfs))

        for i, binned_df in enumerate(split_dfs):

            try:
                #TODO: also change this path
                plts_path = f'C:/Users/evank/OneDrive/Documents/HALnalysis_summer_2024_data/plots/M07480_real_deal/240821/electrode{electrode_targets[j]}/{str(a_list[i])}'
                chip_div_well = chip_div_well_base + f' | {str(a_list[i])}'

                print(chip_div_well)

                os.makedirs(plts_path, exist_ok=True)
                print(plts_path)

                zero_column_titles = binned_df.columns[(binned_df == 0).all(axis=0)].tolist()
                print(f"Channels completely zero all the way down: {zero_column_titles}")

                phys_dists, ham_dists, zero_column_titles = calc_ham_and_phys_dists(binned_df, mapping_pd_df)

                print(phys_dists)
                print(ham_dists)

                plot_hamming_distances(ham_dists)

                ham_and_phys_scat_plot(phys_dists=phys_dists, ham_dists=ham_dists, plts_path=plts_path)

                chan = choice(binned_df.columns)
                pre_win = 10
                post_win = 10

                cofire_matrix = cofiring_probability(binned_df, chan, pre_win, post_win)

                plot_cofiring_prob_one_chan(cofire_matrix, plts_path, chan)

                pre_win = 10
                post_win = 10

                avg_probs_df = calc_avg_probs(binned_df=binned_df, zero_column_titles=zero_column_titles, pre_win=pre_win, post_win=post_win)

                print(avg_probs_df)

                plot_avg_probs_heatmap_all_chans(avg_probs_df=avg_probs_df, plts_path=plts_path)

                binary_matrix = calc_peak_binary_matrix(avg_probs_df=avg_probs_df)

                plot_avg_probs_sort_by_peaks(binary_matrix=binary_matrix, avg_probs_df=avg_probs_df, plts_path=plts_path)

                ref_chan = choice(binned_df.columns)

                sort_avg_probs_df, sort_distances = sort_avg_probs_by_phys_dist(binned_df=binned_df, zero_column_titles=zero_column_titles, mapping=mapping_pd_df, avg_probs_df=avg_probs_df, ref_chan=ref_chan)

                print(sort_avg_probs_df)

                plot_avg_probs_heatmap_sort_by_phys_dist(sort_avg_probs_df=sort_avg_probs_df, distances=sort_distances, plts_path=plts_path, ref_chan=ref_chan)

                results_df = calc_cofire_integrals(avg_probs_df, mapping_pd_df)

                sorted_results_df = results_df.sort_values(by='integral')

                inhibitory_count = (sorted_results_df['status'] == "inhibitory").sum()
                excitatory_count = (sorted_results_df['status'] == "excitatory").sum()

                output_data_path = f"{plts_path}/integrals.txt"
                with open(output_data_path, 'w') as file:
                    for _, row in sorted_results_df.iterrows():
                        output_line = f"Channel: {row['channel']}, Electrode: {row['electrode']}, X: {row['x']}, Y: {row['y']}, Status: {row['status']}, Integral: {row['integral']}"
                        print(output_line.strip())
                        file.write(output_line + '\n')
                    output_line_2 = f"Total inhibitory: {inhibitory_count}, Total excitatory: {excitatory_count}"
                    print(output_line_2)
                    file.write(output_line_2)

                plot_chip_visualization(results_df, plts_path, chip_div_well=chip_div_well)
            
            except:

                print(f'Well, something bad happened.... {chip_div_well}')



