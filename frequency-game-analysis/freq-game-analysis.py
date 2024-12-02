#! /usr/bin/env python

# freq-game-analysis.py
#
# Analysis script for frequency game experiment, which analyzes a CSV file 
# produced by the closed-loop experiment. 
# 
# The documentation in this file refers to 2 different CSVs: "pre-press" and 
# "post-press." 
# 
# The CSVs need to have a "stim level" column in order to complete analysis; if 
# it does not exist, it is calculated based on the button presses. However, it 
# can be viewed in two different ways; at any given button press, the 
# "stim level" column can be the stimulation *before* the button press, or the 
# stimulation level *after* the button press. The documentation in this file 
# refers to these as "pre-press" and "post-press," respectively.
#
# Main can be found at the botton of this file.
#
# IMPORTANT NOTE: This script was made for experiments that occur at most once
# per day. It assumes that the data for each day is found in trial 0. Logic 
# needs to be added in order to change that.
#
# Required CSV colums:
#     - "button pressed"
#
# Useful plots:
#   - plot_button_probs()
#   - plot_button_a_probs_change()
#   - plot_average_button_probs()
#
# Written by Trevor M. Sullivan
# October 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings

# ignore annoying warnings
warnings.simplefilter("ignore")
top_level_dir = os.path.abspath(os.path.join(os.path.dirname("binning.ipynb"), '..'))
sys.path.append(top_level_dir)

from analysis_package import maxlab_analysis as mla

# TODO
parent = "/run/user/1001/gvfs/smb-share:server=rstore.it.tufts.edu,share=as_rsch_levinlab_wclawson01$/Experimental Data/Summer 2024/frequency-game"

# TODO
exp_name = "Multiwell_excite_inhib"

# TODO
chip_id = "M07480"

# TODO
dates = [240828, 240829, 240830, 240903, 240904]

# TODO
trial = 0

# TODO
wells = [0, 1, 2, 4]

# TODO
maxtwo = True


def frames_to_ms(frames):
    """
    Calculates milliseconds given frames from the MaxOne/Two (the one that is 
    calculated can be configured with the global variable, ``maxtwo``)

    :param frames: frames to convert 
    :type frames: ``int``
    :return: milliseconds
    :rtype: ``float``
    """
    global maxtwo
    return frames * (0.1 if maxtwo else 0.05)


def get_stim_level_press_time(df, level):
    """
    Returns all press times at a stimulation level

    :param df: pre or post-press frequency game data frame
    :type df: pandas dataframe
    :param level: stimulation level
    :type level: ``int``
    :return: press times
    :rtype: pandas dataframe
    """
    return df.loc[(np.isin(df["stim level"], level)), ["press time"]]


def get_avg_stim_level_press_time(df, level):
    """
    Gets the average press time at a stimulation level

    :param df: pre or post-press frequency game data frame
    :type df: pandas dataframe
    :param level: stimulation level
    :type level: ``int``
    :return: average press time
    :rtype: float
    """
    return get_stim_level_press_time(df, level).apply(np.mean)["press time"]


def plot_stim_level_press_time(df, level):
    """
    Plots the time spent at a particular stimulaton level

    :param df: pre or post-press frequency game data frame
    :type df: pandas dataframe
    :param level: stimulation level
    :type level: ``int``
    """
    df["press time"] = df["frame number"].diff().apply(frames_to_ms).shift(-1)
    plt.subplot(11, 1, level + 1)
    plt.title(f"Average Time Spent: Level {level}")
    plt.xlabel("Time spent at level (ms)")
    plt.hist(get_stim_level_press_time(df, level), bins=50)


stim_level_count = 0
def update_stim_level(b):
    """
    Gets the stimulaion level based on a button press. Keeps track of the 
    stimulation level via a global counter, "stim_level_count." This is 
    meant to be applied on the "button pressed" column of a frequency game 
    dataframe, and used to calculate the *post-press* stimulation level.

    It is recommended to set stim_level_count to 0 before using this function.

    :param b: button pressed (A or B)
    :type b: ``str``
    :return: updated stimulation level
    :rtype: ``int``
    """
    global stim_level_count
    if b == "B" and stim_level_count < 10:
        stim_level_count += 1
    elif b == "A" and stim_level_count > 0:
        stim_level_count -= 1
    return stim_level_count

def get_pre_press_csv(path):
    """
    Creates pre-press data frame from frequency game CSV output 

    :param path: full path of the frequency game CSV
    :type path: str
    :return: pre-press pandas dataframe
    :rtype: pandas dataframe
    """
    global stim_level_count

    # reset global counter before adding stimulation levels
    stim_level_count = 0
    csv_data = pd.read_csv(path)
            
    if " button pressed" in csv_data:
        csv_data = pd.read_csv(path, sep=", ")
    
    # calculate stimulation level 
    csv_data["stim level"] = csv_data["button pressed"].apply(update_stim_level)
    stim_level_count = 0

    pre_press_csv_data = csv_data.loc[:, ["button pressed", "stim level"]]
    pre_press_csv_data["stim level"] = pre_press_csv_data["stim level"].shift(1)
    pre_press_csv_data.loc[0, "stim level"] = 0

    return pre_press_csv_data


def get_button_prob(df, b):
    """
    Gets the probability that a button is pressed 

    :param df: pre-press frequency game data frame
    :type df: pandas dataframe_summary_
    :param b: button to check the probability f ("A" or "B")
    :type b: ``str``
    :return: probability that that the button was pressed
    :rtype: ``float``
    """
    return len(df.loc[(np.isin(df["button pressed"], b))]) / len(df)


def get_level_up_down_prob(df, level):
    """
    Gets the probabilities that, at a given level, each button was pressed

    :param df: pre-press frequency game data frame
    :type df: pandas dataframe
    :param level: The level to check the probabilities of
    :type level: ``int``
    :return: tuple of probabilities, (A, B) or (down, up)
    :rtype: ``Tuple[float, float]``
    """
    lf = df.loc[(np.isin(df["stim level"], level))]
    return (get_button_prob(lf, "A"), get_button_prob(lf, "B"))


def get_zipped_probs(df):
    """
    Get the probabilities of each button being pressed at each stimulation level

    :param df: pre-press frequency game data frame
    :type df: pandas dataframe

    :return: list of tuples, where each index represents the 
            stimulation level, and each tuple contains the probabilities of 
            A and B being pressed
    :rtype: ``List[Tuple[float, float]]``
    """
    zipped_probs = []
    
    for i in range(11):
        if len(df.loc[(np.isin(df["stim level"], i))]) == 0:
            break
        zipped_probs.append(get_level_up_down_prob(df, i))
    
    return zipped_probs


def plot_button_prob(df, day, well_id):
    """
    Plots the probability of each button being pressed at each stimulation level
    for a specific day and well

    :param df: pre-press frequency game data frame
    :type df: pandas dataframe
    :param day: day to plot (format: YYMMDD)
    :type day: ``int``
    :param well_id: well to plot
    :type well_id: ``int``
    """
    zipped_probs = []
    x_labels = []
    for i in range(11):
        if len(df.loc[(np.isin(df["stim level"], i))]) == 0:
            break
        zipped_probs.append(get_level_up_down_prob(df, i))
        x_labels.append(str(i))
    
    # unzip probabilities
    a_probs, b_probs = map(np.array, map(list, zip(*zipped_probs)))
    plt.figure()
    plt.title(f"Probability of Button Presses at each Level ({day}, {well_id})")
    plt.bar(x_labels, a_probs, bottom=b_probs, label = "A (Down) Probability")
    plt.bar(x_labels, b_probs, label = "B (Up) Probability")
    

    plt.legend()
    plt.show()


def plot_probs_individual():
    """
    Plots the button press probabilities for each stimulation level, for each 
    day, for each well, *individually*.

    If you want them to display all at once, commment out the ``plt.show()`` 
    at the end of ``plot_button_prob()``

    NOTE: this makes a lot of graphs.
    """
    global parent, exp_name, chip_id, trial
    days = [240828, 240829, 240830, 240903, 240904]
    wells = [0, 1, 2, 4]
    for well_id in wells:
        
        global stim_level_count
        stim_level_count = 0
        for day in days:
            data_path = f"{parent}/{exp_name}/{chip_id}/{day}/{trial}/well{well_id}/"
            csv_file_name = f"button_data_well_{well_id}.csv"

            pre_press_csv_data = get_pre_press_csv(data_path + csv_file_name)
            
            plot_button_prob(pre_press_csv_data, day, well_id)


def get_prob(well_id, day):
    """
    Gets the probabillities at each stimulation level for a well and day

    :param well_id: well
    :type well_id: ``int``
    :param day: date (YYMMDD)
    :type day: ``int``
    :return: probabilities of each button at each stimulation level (A, B)
    :rtype: ``List[Tuple[float, float]]``
    """
    global parent, exp_name, chip_id, trial, stim_level_count

    data_path = f"{parent}/{exp_name}/{chip_id}/{day}/{trial}/well{well_id}/"
    csv_file_name = f"button_data_well_{well_id}.csv"
    
    pre_press_csv_data = get_pre_press_csv(data_path + csv_file_name)
    day_probs = get_zipped_probs(pre_press_csv_data)
    
    return day_probs


def plot_button_probs():
    """
    Plots the probability of button A being pressed for each level, day and 
    wel, all in one plot
    """
    global dates, wells
    plt.figure(figsize = (10, 10))

    for i in range(len(dates)):
        ax = plt.subplot(len(dates), 1, i + 1)
        plt.title(f"Button A press probabilities on {dates[i]}")
        plt.xlabel("Stimulation level")
        plt.ylabel("Probability")
        for well in wells:
            well_probs = get_prob(well, dates[i])
            a_probs, _ = map(np.array, map(list, zip(*well_probs)))
            ax.plot(a_probs, label = f"Well {well}", marker = "o")
        
    plt.legend()
    plt.tight_layout()


def get_delta_prob_avgs():
    """
    Gets avg change in button A press probabilities over all wells and dates

    :return: average change in probabilities for each stimulation level 
             (interior list) and each well (exterior list)
    :rtype: ``List[List[float]]``
    """
    global dates, wells
    probs = []
    for well in wells:
        probs.append(get_well_delta_prob_avg(well, dates))

    return probs


def print_delta_prob_avgs():
    days = [240828, 240829, 240830, 240903, 240904]
    wells = [0, 1, 2, 4]
    for well in wells:
        print(f"Average change in probabilities for each stimulation level, well {well}")
        print(get_well_delta_prob_avg(well, days))


def get_well_delta_prob_avg(well_id, days):
    """
    Gets the the average change in button A press probabilities for a specific well 
    between some sequence of days

    :param well_id: well
    :type well_id: ``int``
    :param days: dates to get the change in probabilities (YYMMDD)
    :type days: ``List[int]``
    :return: list of change in probabilities for each stimulation level 
             (interior list), over all days (exterior list)
    :rtype: ``List[List[float]]``
    """
    global parent, exp_name, chip_id, trial, stim_level_count
    probs = [] # will contain probs for all days at this well
    min_level = 10 # represents the lowest maximum level achieved over all days
    for day in days:
        data_path = f"{parent}/{exp_name}/{chip_id}/{day}/{trial}/well{well_id}/"
        csv_file_name = f"button_data_well_{well_id}.csv"

        pre_press_csv_data = get_pre_press_csv(data_path + csv_file_name)
        day_probs = get_zipped_probs(pre_press_csv_data)
        probs.append(day_probs)

        if (len(day_probs) - 1 < min_level):
            min_level = len(day_probs)

    delta_probs = []
    num_deltas = len(probs) - 1
    
    # go through all levels that are guaranteed to be in each day
    for lvl in range(min_level):
        a_sum = 0
        b_sum = 0
        for day in range(num_deltas):
            a0, b0 = probs[day][lvl]
            a1, b1 = probs[day + 1][lvl]
            a_sum += (a1 - a0)
            b_sum += (b1 - b0)
            
        delta_probs.append((a_sum / num_deltas, b_sum / num_deltas))

    return delta_probs


def plot_button_a_probs_change():
    """
    Plots the change in the probability of A button presses between each day, at 
    each stimulation level and well
    """
    global dates, wells
    plt.figure(figsize = (10, 10))
    for i in range(len(dates) - 1):
        ax = plt.subplot(len(dates) - 1, 1, i + 1)
        plt.title(f"Change in Button A Press Probabilities between {dates[i]} and {dates[i + 1]}")
        plt.xlabel("Stimulation level")
        plt.ylabel("Change in Probability")
        for well in wells:
            well_probs_0 = get_prob(well, dates[i])
            well_probs_1 = get_prob(well, dates[i + 1])

            a_probs_0, _ = map(np.array, map(list, zip(*well_probs_0)))
            a_probs_1, _ = map(np.array, map(list, zip(*well_probs_1)))

            deltas = list(map(lambda d: d[1] - d[0], list(zip(a_probs_0, a_probs_1))))
            ax.plot(deltas, label = f"Well {well}", marker = "o")
    plt.legend()
    plt.tight_layout()


def plot_average_button_probs():
    """
    Plots the average change in button A presses between days over each well
    and stimulation level
    """
    global wells
    probs = get_delta_prob_avgs()

    plt.figure()

    for i in range(len(wells)):
        well_probs = probs[i]
        a_probs, _ = map(np.array, map(list, zip(*well_probs)))

        plt.plot(a_probs, label = f"Well {wells[i]}", marker = "o")

    plt.legend()
    plt.title("Average change in button A press probability")
    plt.xlabel("Stimulation Level")
    plt.ylabel("A Press Probability")
    

if __name__ == "__main__":

    plot_button_probs()
    plot_button_a_probs_change()
    plot_average_button_probs()

    # display all plots at once
    plt.show()
