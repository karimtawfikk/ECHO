"""
Proper Waterfall Chart - Stacked Components (Like LangSmith Trace)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path

# ============================================================================
# Component Latency Data
# ============================================================================

components = [
    {"name": "Rewriter", "time": 1.310, "color": "#3b82f6"},    # Blue
    {"name": "Retriever", "time": 1.567, "color": "#f59e0b"},   # Orange
    {"name": "Reranker", "time": 0.857, "color": "#10b981"},    # Green
    {"name": "Generator", "time": 1.377, "color": "#ef4444"},   # Red
]

total_time = sum(c["time"] for c in components)

# ============================================================================
# Create Waterfall Chart (Stacked Vertically)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#0f172a')  # Dark background like LangSmith
ax.set_facecolor('#0f172a')

# Starting positions
start_time = 0
bar_height = 0.5
y_spacing = 1.0

# Draw "LangGraph" parent bar at top
parent_y = len(components) * y_spacing
parent_bar = FancyBboxPatch(
    (0, parent_y - bar_height/2),
    total_time,
    bar_height,
    boxstyle="round,pad=0.03",
    linewidth=2,
    edgecolor='#94a3b8',
    facecolor='#1e3a8a',
    alpha=0.7,
    zorder=5
)
ax.add_patch(parent_bar)

ax.text(
    0.1, parent_y,
    f"LangGraph  {total_time:.2f}s",
    ha='left', va='center',
    fontsize=12,
    fontweight='bold',
    color='white',
    zorder=10
)

# Draw each component stacked below
cumulative_time = 0

for i, comp in enumerate(components):
    y_pos = (len(components) - 1 - i) * y_spacing
    percentage = (comp["time"] / total_time) * 100
    
    # Draw bar starting from cumulative time
    bar = FancyBboxPatch(
        (cumulative_time, y_pos - bar_height/2),
        comp["time"],
        bar_height,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor='white',
        facecolor=comp["color"],
        alpha=0.9,
        zorder=10
    )
    ax.add_patch(bar)
    
    # Add component name and time
    ax.text(
        cumulative_time + 0.1, y_pos,
        f"{comp['name']}  {comp['time']:.2f}s",
        ha='left', va='center',
        fontsize=11,
        fontweight='bold',
        color='white',
        zorder=20
    )
    
    # Add percentage on the right side of bar
    ax.text(
        cumulative_time + comp["time"] + 0.15, y_pos,
        f"({percentage:.1f}%)",
        ha='left', va='center',
        fontsize=9,
        color='#94a3b8',
        fontweight='bold',
        zorder=20
    )
    
    # Draw connector line from parent to child (except for first component)
    if i == 0:
        # First component connects to parent start
        connector = ConnectionPatch(
            (0, parent_y - bar_height/2),
            (cumulative_time, y_pos + bar_height/2),
            "data", "data",
            arrowstyle="-",
            shrinkA=0, shrinkB=0,
            color='#475569',
            linewidth=1.5,
            linestyle='--',
            alpha=0.6,
            zorder=3
        )
        ax.add_artist(connector)
    else:
        # Connect from previous component end to current start
        prev_y = (len(components) - i) * y_spacing
        connector = ConnectionPatch(
            (cumulative_time, prev_y - bar_height/2),
            (cumulative_time, y_pos + bar_height/2),
            "data", "data",
            arrowstyle="-",
            shrinkA=0, shrinkB=0,
            color='#475569',
            linewidth=1.5,
            linestyle='--',
            alpha=0.6,
            zorder=3
        )
        ax.add_artist(connector)
    
    cumulative_time += comp["time"]

# ============================================================================
# Add Timeline (Top X-axis)
# ============================================================================

# Timeline ticks every 0.5s
timeline_ticks = np.arange(0, total_time + 0.5, 0.5)
for tick in timeline_ticks:
    ax.axvline(
        x=tick,
        ymin=0, ymax=1,
        color='#334155',
        linewidth=0.8,
        linestyle=':',
        alpha=0.5,
        zorder=1
    )
    
    # Add time labels at top
    ax.text(
        tick, parent_y + 0.7,
        f"{tick:.1f}s",
        ha='center', va='bottom',
        fontsize=9,
        color='#94a3b8'
    )

# ============================================================================
# Styling
# ============================================================================

# Set limits
ax.set_xlim(-0.3, total_time + 0.8)
ax.set_ylim(-0.5, parent_y + 1.2)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Title
ax.text(
    total_time / 2, parent_y + 1.5,
    'RAG Pipeline - Waterfall Trace',
    ha='center', va='bottom',
    fontsize=16,
    fontweight='bold',
    color='#f1f5f9'
)

# Add "Total Time" annotation
ax.annotate(
    f'Total: {total_time:.2f}s',
    xy=(total_time, parent_y),
    xytext=(total_time + 0.3, parent_y + 0.5),
    fontsize=11,
    fontweight='bold',
    color='#22d3ee',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='#0f172a',
        edgecolor='#22d3ee',
        linewidth=2
    ),
    arrowprops=dict(
        arrowstyle='->',
        color='#22d3ee',
        linewidth=2,
        connectionstyle='arc3,rad=0.2'
    )
)

plt.tight_layout()

# Save
output_dir = Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\efficiency_evaluation_results")
output_dir.mkdir(exist_ok=True)
plt.savefig(
    output_dir / 'waterfall_stacked.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='#0f172a'
)
print(f"📊 Saved stacked waterfall chart to: {output_dir / 'waterfall_stacked.png'}")

plt.show()