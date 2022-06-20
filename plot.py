from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

def bigplot(dataframe_list : List[pd.DataFrame], excluded_labels, x_axis, y_axis, figsize=(25,8), offset_width=0.15, title=None, legends=None, colors=None, show_median=True, show_mean=False):
    
    if not colors:
        colors = ("red", "blue", "green")

    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=20)
        
    label_dict = {v: i for i, v in enumerate(dataframe_list[0].columns) if v not in excluded_labels}
    
    ax.set_xticks(list(label_dict.values()))
    ax.set_xticklabels(list(label_dict.keys()), fontsize=18, rotation=25, ha="right")
    for i, df in enumerate(dataframe_list):
        offset = offset_width * i
        color = colors[i]

        for i in ax.get_xticklabels():
            i = i.get_text()
            
            y = [l for l in df[str(i)].dropna()]
            x = [label_dict[i]+offset for l in y]
            ax.scatter(x=x, y=y, edgecolor=color, marker="o", facecolor="none")
            
            if show_median:
                median = df[str(i)].median(skipna=True)
                ax.scatter(x=label_dict[i]+offset, y=median, c=color, marker="_", s=1001)
            
            
            if show_mean:
                median = df[str(i)].mean(skipna=True)
                ax.scatter(x=label_dict[i]+offset, y=median, c=color, marker="^", s=100)
                
            
    ax.set_xlabel(x_axis,  fontsize=18)
    ax.set_ylabel(y_axis, fontsize=18)
    ax.tick_params(labelsize=18)
    fig.set_facecolor("white")
    return fig, ax

def add_legend(ax, color, label, marker):
    
    mlines.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="none", markeredgecolor="red", label="DL")
