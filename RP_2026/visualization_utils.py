"""RP_2026 visualization utilities."""
import os
import re
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

console = Console()

# Set global settings
plt.rcParams.update(
    {
        "savefig.dpi": 450,  # Set resolution to 450 dpi
        "font.family": "DejaVu Sans",  # Set font to Arial  # TODO: replace with Arial for Nature
        "pdf.fonttype": 42,  # Embed fonts as TrueType (keeps text editable)
        "ps.fonttype": 42,  # Same for PostScript files
        "savefig.format": "pdf",  # Default save format as PNG
        "savefig.bbox": "tight",  # Adjust bounding box to fit tightly
        "figure.facecolor": "white",  # Set figure background to white (common for RGB)
        "savefig.transparent": False,  # Disable transparency
    }
)

color_map_10 = plt.get_cmap("tab10").colors  # Default color map with 10 colors
color_map_20 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]  # plotly category 20
DPI = 450
