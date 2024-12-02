import matplotlib.pyplot as plt
import networkx as nx
import h5py
import numpy as np
import pandas as pd
import re
import os
import sys
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../HALnalysis")
import analysis_package as mla

#Should I save the raw spike data too? From that time window?
#yes
#save the entire mapping also




def build_transition_df(movement_df: pd.DataFrame, world: list):
    num_locs = max(movement_df["position"]) + 1
    transition_df = pd.DataFrame(index = range(0, num_locs), columns = ["leftward", "rightward"], data = np.zeros((num_locs, 2)))
    
    current_pos = movement_df.loc[0, "position"]
    for i in movement_df.index[1:]:
        next_pos = movement_df.loc[i, "position"]
        if (next_pos + num_locs) % num_locs == (current_pos - 1 + num_locs) % num_locs:
            transition_df.loc[current_pos, "leftward"] += 1
        elif (next_pos + num_locs) % num_locs == (current_pos + 1 + num_locs) % num_locs:
            transition_df.loc[current_pos, "rightward"] += 1
        else:
            print("fail - next pos not one away from current_pos")
            print(next_pos)
            print(current_pos)
            sys.exit()

        current_pos = next_pos

    transition_df["total"] = transition_df["leftward"] + transition_df["rightward"]
    transition_df["proportion left"] = transition_df["leftward"]/transition_df["total"]
    transition_df["object"] = world

    return transition_df




def make_probability_bars(train_transition_df, world):
    ax = plt.gca()
    ax.bar(train_transition_df.index, train_transition_df["proportion left"], label = "leftwards")
    ax.bar(train_transition_df.index, 1-train_transition_df["proportion left"], bottom = train_transition_df["proportion left"], label = "rightwards")
    ax.bar_label(ax.containers[0], labels = [f"n={i:.0f}" for i in train_transition_df["total"]], label_type='edge')  # Annotating bars
    plt.ylim(0, 1.1)
    plt.ylabel("Proportion leftwards bursting")
    plt.xlabel("Position in world")
    plt.xticks(range(0, len(world)), [f"{i}\n{x}" if x != "0" else f"{i}" for i, x in enumerate(train_transition_df["object"])])
    plt.legend()


def make_count_bars(train_transition_df, world):
    ax = plt.gca()
    ax.bar(train_transition_df.index, train_transition_df["leftward"], label = "leftwards")
    ax.bar(train_transition_df.index, train_transition_df["rightward"], bottom = train_transition_df["leftward"], label = "rightwards")
    plt.ylabel("num bursts")
    plt.xlabel("Position in world")
    plt.xticks(range(0, len(world)), [f"{i}\n{x}" if x != "0" else f"{i}" for i, x in enumerate(world)])
    ax.legend()



def make_network_graph(train_transition_df, world):
   G = nx.MultiDiGraph()  
   G.add_nodes_from(train_transition_df.index, size = train_transition_df["total"], object = train_transition_df["object"])
   for i in range(len(world)):
      G.add_edge(i, (i+1)%len(world), weight = 1-train_transition_df.loc[i, "proportion left"])
      G.add_edge((i+1)%len(world), i, weight = train_transition_df.loc[(i+1)%len(world), "proportion left"])

   pos = nx.circular_layout(G)


   colors = {
      "0": "grey",
      "1": "y"
   }

   colored_nodes = nx.draw_networkx_nodes(G, pos, train_transition_df.index, node_size = train_transition_df["total"]*1000/min(train_transition_df["total"]), node_color = [colors[i] for i in train_transition_df["object"]])
   nx.draw_networkx_edges(G, pos, alpha = [d['weight'] for u, v, d in G.edges(data=True)], arrowstyle = "simple", width = 1, arrowsize = [d['weight'] * 50 for u, v, d in G.edges(data=True)], connectionstyle="arc3,rad=0.2", node_size = train_transition_df["total"]*800/min(train_transition_df["total"]))

   nx.draw_networkx_labels(G, pos, labels = {i: f"{i}\nn={train_transition_df.loc[i, 'total']:.0f}" for i in train_transition_df.index}, font_size=12, font_family="sans-serif")
   # nx.draw_networkx_edge_labels(
   #     G, pos, edge_labels={(u, v): f"{d['weight']:.3f}" for u, v, d in G.edges(data=True)}, 
   # )
   __my_draw_networkx_edge_labels(
      G, pos, edge_labels={(u, v): f"{d['weight']:.3f}" for u, v, d in G.edges(data=True)}, rad = 0.2
   )

   return G






















def __my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items















if __name__ == '__main__':
    print("hello world")



    homedir=os.path.expanduser("~")

   
    #filepath = "/run/user/1001/gvfs/smb-share:server=rstore.it.tufts.edu,share=as_rsch_levinlab_wclawson01$/Experimental Data/Summer 2024/mouse_1d/7-9 mouse_1d_test/7/well4/"
    #filepath = "/run/user/1001/gvfs/smb-share:server=rstore.it.tufts.edu,share=as_rsch_levinlab_wclawson01$/Experimental Data/Summer 2024/mouse_1d/2h_7-12/M07484/240712/4/well4/"
    #filepath = "/run/user/1001/gvfs/smb-share:server=rstore.it.tufts.edu,share=as_rsch_levinlab_wclawson01$/Experimental Data/Summer 2024/mouse_1d/DIV13/M07475/240716/0/well1/"
    filepath = "/run/user/1001/gvfs/smb-share:server=rstore.it.tufts.edu,share=as_rsch_levinlab_wclawson01$/Experimental Data/Summer 2024/mouse_1d/DIV13_no_media_change/M07472/240716/1/well0/"


    #filename = "DIV13_well_1"
    filename = "DIV13_no_media_change_well_0"
    #filename = "2h_well_4"
    filetag = ".raw.h5"
    datapath = "mouse1d-analysis/processed_data/"
    well_no = 0
    recording_no = 0


    burst_dict = find_bursts(filepath, filename, well_no, recording_no, burst_thresh = 2, stim_thresh = 0.95)

    print(burst_dict)

    for i in range(0, len(burst_dict)):
        burst_dict[i].plot()

