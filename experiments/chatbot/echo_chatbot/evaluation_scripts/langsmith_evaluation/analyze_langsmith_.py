"""
Enhanced LangSmith Analysis with TTFT, Generator-Only Tokens, and Waterfall Chart
"""

import os
from langsmith import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
from pathlib import Path

# ============================================================================
# Initialize LangSmith Client
# ============================================================================

client = Client()

print("📊 Fetching data from LangSmith...\n")

# ============================================================================
# Initialize LangSmith Client and Fetch Evaluation Runs Only
# ============================================================================

client = Client()

print("📊 Fetching data from LangSmith...\n")

# Get ONLY root runs first (efficiency optimization)
all_root_runs = list(client.list_runs(
    project_name="Ancient-Egypt-RAG",
    is_root=True  # Only get root-level runs
))

print(f"✅ Fetched {len(all_root_runs)} root runs from project")

# ============================================================================
# Filter to ONLY Efficiency Evaluation Runs (132 queries)
# ============================================================================

filtered_root_runs = []

for run in all_root_runs:
    # Check if run has the evaluation metadata
    metadata = {}
    
    # Extract metadata from different possible locations
    if hasattr(run, 'extra') and run.extra:
        metadata = run.extra.get('metadata', {})
    
    # Method 1: Check evaluation_type
    is_eval_run = metadata.get('evaluation_type') == 'efficiency'
    
    # Method 2: Check thread_id pattern
    thread_id = metadata.get('thread_id', '')
    if isinstance(thread_id, str) and 'efficiency-eval-' in thread_id:
        is_eval_run = True
    
    # Method 3: Check run_name pattern (fallback)
    run_name = metadata.get('run_name', '')
    if isinstance(run_name, str) and 'efficiency' in run_name.lower():
        is_eval_run = True
    
    # Method 4: Check query_id exists (evaluation runs have this)
    if 'query_id' in metadata:
        is_eval_run = True
    
    if is_eval_run:
        filtered_root_runs.append(run)

print(f"  • Filtered to {len(filtered_root_runs)} efficiency evaluation runs")

if len(filtered_root_runs) == 0:
    print("\n❌ ERROR: No evaluation runs found!")
    print("\nDEBUG: Showing first 5 root runs metadata:")
    for i, run in enumerate(all_root_runs[:5]):
        print(f"\nRun {i+1}:")
        print(f"  ID: {run.id}")
        print(f"  Name: {run.name}")
        if hasattr(run, 'extra') and run.extra:
            print(f"  Metadata: {run.extra.get('metadata', {})}")
        else:
            print(f"  No metadata found")
    exit(1)

if len(filtered_root_runs) != 132:
    print(f"  ⚠️  Warning: Expected 132 root runs, found {len(filtered_root_runs)}")
else:
    print(f"  ✅ Confirmed: All 132 evaluation queries found")

print()

# Now get child runs for these specific root runs
print("📊 Fetching child runs for evaluation queries...")

all_child_runs = []
for root_run in filtered_root_runs:
    # Get all children for this root run
    children = list(client.list_runs(
        trace_id=root_run.trace_id,
        is_root=False
    ))
    all_child_runs.extend(children)

print(f"  • Found {len(all_child_runs)} child runs (components)\n")

# Assign for compatibility with rest of script
root_runs = filtered_root_runs
child_runs = all_child_runs

print(f"  • Root runs (full queries): {len(root_runs)}")
print(f"  • Child runs (components): {len(child_runs)}\n")

# ============================================================================
# Extract Metrics with TTFT and Generator-Only Tokens
# ============================================================================

data = []

for run in root_runs:
    metadata = run.extra.get('metadata', {}) if run.extra else {}
    
    row = {
        'run_id': str(run.id),
        'entity_type': metadata.get('entity_type', 'unknown'),
        'entity_name': metadata.get('entity_name', 'unknown'),
        'query': run.inputs.get('query', '') if run.inputs else '',
        'total_latency': run.latency if run.latency else 0,
        'success': run.status == 'success',
        'start_time': run.start_time,
        'end_time': run.end_time,
    }
    
    # Initialize component times and tokens
    row['rewrite_time'] = 0
    row['retrieve_time'] = 0
    row['rerank_time'] = 0
    row['generate_time'] = 0
    row['generator_tokens'] = 0
    row['generator_prompt_tokens'] = 0
    row['generator_completion_tokens'] = 0
    row['rewriter_tokens'] = 0
    
    # Extract per-component data from child runs
    try:
        children = [c for c in child_runs if c.parent_run_id == run.id or c.trace_id == run.trace_id]
        
        for child in children:
            if not child.name:
                continue
            
            node_name = child.name.lower()
            
            # Component timings
            if 'rewrite' in node_name:
                row['rewrite_time'] = child.latency or 0
                row['rewriter_tokens'] = child.total_tokens or 0
            elif 'retrieve' in node_name:
                row['retrieve_time'] = child.latency or 0
            elif 'rerank' in node_name:
                row['rerank_time'] = child.latency or 0
            elif 'generate' in node_name or 'generator' in node_name:
                row['generate_time'] = child.latency or 0
                row['generator_tokens'] = child.total_tokens or 0
                row['generator_prompt_tokens'] = child.prompt_tokens or 0
                row['generator_completion_tokens'] = child.completion_tokens or 0
    
    except Exception as e:
        print(f"⚠️  Could not extract children for {run.id}: {e}")
    
    # Calculate TTFT (Time to First Token)
    # TTFT = Rewrite + Retrieve + Rerank + (small delay for first token)
    # We approximate first token delay as ~0.1-0.2s (typical for streaming)
    row['ttft'] = row['rewrite_time'] + row['retrieve_time'] + row['rerank_time'] + 0.15
    
    # Calculate TPS (Tokens Per Second) - generator only
    if row['generate_time'] > 0 and row['generator_completion_tokens'] > 0:
        row['tps'] = row['generator_completion_tokens'] / row['generate_time']
    else:
        row['tps'] = 0
    
    data.append(row)

df = pd.DataFrame(data)

# Save to CSV
output_dir = Path("langsmith_analysis")
output_dir.mkdir(exist_ok=True)

df.to_csv(output_dir / "efficiency_metrics_enhanced.csv", index=False)
print(f"💾 Saved metrics to: {output_dir / 'efficiency_metrics_enhanced.csv'}\n")

# ============================================================================
# Calculate Statistics
# ============================================================================

print("=" * 80)
print("📈 ENHANCED EFFICIENCY METRICS")
print("=" * 80 + "\n")

success_df = df[df['success'] == True]

if len(success_df) == 0:
    print("❌ No successful runs found!")
    exit(1)

# TTFT Metrics (PRIMARY - User Perceived Latency)
print("⚡ TIME TO FIRST TOKEN (TTFT) - Primary UX Metric:")
print(f"  • Mean:   {success_df['ttft'].mean():.2f}s")
print(f"  • Median: {success_df['ttft'].median():.2f}s")
print(f"  • P50:    {success_df['ttft'].quantile(0.50):.2f}s")
print(f"  • P90:    {success_df['ttft'].quantile(0.90):.2f}s")
print(f"  • P99:    {success_df['ttft'].quantile(0.99):.2f}s")
print(f"  • Min:    {success_df['ttft'].min():.2f}s")
print(f"  • Max:    {success_df['ttft'].max():.2f}s\n")

# End-to-End Latency (SECONDARY - Total Time)
print("⏱️  END-TO-END LATENCY (Total Response Time):")
print(f"  • Mean:   {success_df['total_latency'].mean():.2f}s")
print(f"  • Median: {success_df['total_latency'].median():.2f}s")
print(f"  • P50:    {success_df['total_latency'].quantile(0.50):.2f}s")
print(f"  • P90:    {success_df['total_latency'].quantile(0.90):.2f}s")
print(f"  • P99:    {success_df['total_latency'].quantile(0.99):.2f}s")
print(f"  • Min:    {success_df['total_latency'].min():.2f}s")
print(f"  • Max:    {success_df['total_latency'].max():.2f}s\n")

# Component Breakdown
print("🔧 COMPONENT LATENCY BREAKDOWN (Average):")
components = ['rewrite_time', 'retrieve_time', 'rerank_time', 'generate_time']
component_names = ['Rewriter', 'Retriever', 'Reranker', 'Generator']

total_component_time = sum(success_df[c].mean() for c in components)

for comp, name in zip(components, component_names):
    avg_time = success_df[comp].mean()
    percentage = (avg_time / total_component_time * 100) if total_component_time > 0 else 0
    print(f"  • {name:12s}: {avg_time:.3f}s ({percentage:.1f}%)")

print()

# Token Metrics - GENERATOR ONLY
print("🔢 TOKEN METRICS (Generator LLM Only - GPT-OSS-120B):")
print(f"  • Total tokens:      {success_df['generator_tokens'].sum():,}")
print(f"  • Avg per query:     {success_df['generator_tokens'].mean():.0f}")
print(f"  • Prompt tokens:     {success_df['generator_prompt_tokens'].sum():,}")
print(f"  • Completion tokens: {success_df['generator_completion_tokens'].sum():,}")
print(f"  • Avg TPS:           {success_df[success_df['tps'] > 0]['tps'].mean():.1f} tokens/sec\n")

# Rewriter Token Stats (for comparison)
print("📝 TOKEN METRICS (Rewriter LLM - Qwen-3-32B):")
print(f"  • Total tokens:      {success_df['rewriter_tokens'].sum():,}")
print(f"  • Avg per query:     {success_df['rewriter_tokens'].mean():.0f}\n")

# Combined Totals
total_all_tokens = success_df['generator_tokens'].sum() + success_df['rewriter_tokens'].sum()
print(f"📊 COMBINED TOKEN USAGE (Both LLMs):")
print(f"  • Total tokens:      {total_all_tokens:,}")
print(f"  • Generator %:       {(success_df['generator_tokens'].sum() / total_all_tokens * 100):.1f}%")
print(f"  • Rewriter %:        {(success_df['rewriter_tokens'].sum() / total_all_tokens * 100):.1f}%\n")

# Success Rate
print("✅ SUCCESS RATE:")
print(f"  • Successful: {len(success_df)}/{len(df)} ({len(success_df)/len(df)*100:.1f}%)\n")

print("=" * 80 + "\n")

# ============================================================================
# Generate Enhanced Charts
# ============================================================================

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('RAG System - Enhanced Efficiency Analysis', fontsize=18, fontweight='bold')

# Chart 1: TTFT Distribution (PRIMARY METRIC)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(success_df['ttft'], bins=30, edgecolor='black', color='#3b82f6', alpha=0.7)
ttft_p50 = success_df['ttft'].quantile(0.50)
ttft_p90 = success_df['ttft'].quantile(0.90)
ttft_p99 = success_df['ttft'].quantile(0.99)
ax1.axvline(ttft_p50, color='green', linestyle='--', linewidth=2, label=f'P50: {ttft_p50:.2f}s')
ax1.axvline(ttft_p90, color='orange', linestyle='--', linewidth=2, label=f'P90: {ttft_p90:.2f}s')
ax1.axvline(ttft_p99, color='red', linestyle='--', linewidth=2, label=f'P99: {ttft_p99:.2f}s')
ax1.set_xlabel('Time to First Token (seconds)', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title(' TTFT Distribution (User Perceived Latency)', fontweight='bold')
ax1.legend()

# Chart 2: End-to-End Latency Distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(success_df['total_latency'], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
e2e_p50 = success_df['total_latency'].quantile(0.50)
e2e_p90 = success_df['total_latency'].quantile(0.90)
e2e_p99 = success_df['total_latency'].quantile(0.99)
ax2.axvline(e2e_p50, color='green', linestyle='--', linewidth=2, label=f'P50: {e2e_p50:.2f}s')
ax2.axvline(e2e_p90, color='orange', linestyle='--', linewidth=2, label=f'P90: {e2e_p90:.2f}s')
ax2.axvline(e2e_p99, color='red', linestyle='--', linewidth=2, label=f'P99: {e2e_p99:.2f}s')
ax2.set_xlabel('Total Latency (seconds)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('  End-to-End Latency Distribution', fontweight='bold')
ax2.legend()

# Chart 3: Component Breakdown
ax3 = fig.add_subplot(gs[1, 0])
component_avg = [success_df[c].mean() for c in components]
ax3.bar(component_names, component_avg, color=['#3b82f6', '#f59e0b', '#10b981', '#ef4444'], edgecolor='black')
ax3.set_ylabel('Average Time (seconds)', fontweight='bold')
ax3.set_title(' Latency Breakdown by Component', fontweight='bold')
ax3.tick_params(axis='x', rotation=15)

for i, (name, val) in enumerate(zip(component_names, component_avg)):
    percentage = (val / sum(component_avg) * 100)
    ax3.text(i, val + 0.05, f'{percentage:.1f}%', ha='center', fontweight='bold')

# Chart 4: Token Distribution (Generator vs Rewriter)
ax4 = fig.add_subplot(gs[1, 1])
token_data = [
    success_df['generator_tokens'].sum(),
    success_df['rewriter_tokens'].sum()
]
ax4.bar(['Generator\n(GPT-OSS-120B)', 'Rewriter\n(Qwen-3-32B)'], 
        token_data,
        color=['#ef4444', '#3b82f6'], edgecolor='black')
ax4.set_ylabel('Total Tokens', fontweight='bold')
ax4.set_title(' Token Usage by LLM', fontweight='bold')

for i, val in enumerate(token_data):
    percentage = (val / sum(token_data) * 100)
    ax4.text(i, val + 5000, f'{val:,}\n({percentage:.1f}%)', ha='center', fontweight='bold')

# Chart 5: TTFT CDF
ax5 = fig.add_subplot(gs[2, 0])
sorted_ttft = np.sort(success_df['ttft'])
cumulative = np.arange(1, len(sorted_ttft) + 1) / len(sorted_ttft)
ax5.plot(sorted_ttft, cumulative * 100, linewidth=2, color='#3b82f6')
ax5.axhline(50, color='green', linestyle='--', alpha=0.7, label=f'P50: {ttft_p50:.2f}s')
ax5.axhline(90, color='orange', linestyle='--', alpha=0.7, label=f'P90: {ttft_p90:.2f}s')
ax5.axhline(99, color='red', linestyle='--', alpha=0.7, label=f'P99: {ttft_p99:.2f}s')
ax5.set_xlabel('Time to First Token (seconds)', fontweight='bold')
ax5.set_ylabel('Cumulative Percentage (%)', fontweight='bold')
ax5.set_title(' TTFT Cumulative Distribution', fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Chart 6: TPS Distribution
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(success_df[success_df['tps'] > 0]['tps'], bins=30, edgecolor='black', color='#10b981', alpha=0.7)
mean_tps = success_df[success_df['tps'] > 0]['tps'].mean()
ax6.axvline(mean_tps, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_tps:.1f}')
ax6.set_xlabel('Tokens per Second', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title(' Generation Speed (TPS)', fontweight='bold')
ax6.legend()

plt.savefig(output_dir / 'efficiency_analysis_enhanced.png', dpi=300, bbox_inches='tight')
print(f"📊 Saved enhanced charts to: {output_dir / 'efficiency_analysis_enhanced.png'}\n")

# ============================================================================
# Generate Waterfall Chart (Integrated)
# ============================================================================

# Calculate average component times for waterfall
avg_component_times = {
    'Rewriter': success_df['rewrite_time'].mean(),
    'Retriever': success_df['retrieve_time'].mean(),
    'Reranker': success_df['rerank_time'].mean(),
    'Generator': success_df['generate_time'].mean(),
}

components_waterfall = [
    {"name": "Rewriter", "time": avg_component_times['Rewriter'], "color": "#3b82f6"},
    {"name": "Retriever", "time": avg_component_times['Retriever'], "color": "#f59e0b"},
    {"name": "Reranker", "time": avg_component_times['Reranker'], "color": "#10b981"},
    {"name": "Generator", "time": avg_component_times['Generator'], "color": "#ef4444"},
]

total_time_waterfall = sum(c["time"] for c in components_waterfall)

fig_wf, ax_wf = plt.subplots(figsize=(12, 8))
fig_wf.patch.set_facecolor('#0f172a')
ax_wf.set_facecolor('#0f172a')

bar_height = 0.5
y_spacing = 1.0

# Parent bar
parent_y = len(components_waterfall) * y_spacing
parent_bar = FancyBboxPatch(
    (0, parent_y - bar_height/2),
    total_time_waterfall,
    bar_height,
    boxstyle="round,pad=0.03",
    linewidth=2,
    edgecolor='#94a3b8',
    facecolor='#1e3a8a',
    alpha=0.7
)
ax_wf.add_patch(parent_bar)

ax_wf.text(
    0.1, parent_y,
    f"LangGraph  {total_time_waterfall:.2f}s",
    ha='left', va='center',
    fontsize=12,
    fontweight='bold',
    color='white'
)

# Component bars
cumulative_time = 0

for i, comp in enumerate(components_waterfall):
    y_pos = (len(components_waterfall) - 1 - i) * y_spacing
    percentage = (comp["time"] / total_time_waterfall) * 100
    
    bar = FancyBboxPatch(
        (cumulative_time, y_pos - bar_height/2),
        comp["time"],
        bar_height,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor='white',
        facecolor=comp["color"],
        alpha=0.9
    )
    ax_wf.add_patch(bar)
    
    ax_wf.text(
        cumulative_time + 0.1, y_pos,
        f"{comp['name']}  {comp['time']:.2f}s",
        ha='left', va='center',
        fontsize=11,
        fontweight='bold',
        color='white'
    )
    
    ax_wf.text(
        cumulative_time + comp["time"] + 0.15, y_pos,
        f"({percentage:.1f}%)",
        ha='left', va='center',
        fontsize=9,
        color='#94a3b8',
        fontweight='bold'
    )
    
    if i == 0:
        connector = ConnectionPatch(
            (0, parent_y - bar_height/2),
            (cumulative_time, y_pos + bar_height/2),
            "data", "data",
            arrowstyle="-",
            color='#475569',
            linewidth=1.5,
            linestyle='--',
            alpha=0.6
        )
        ax_wf.add_artist(connector)
    else:
        prev_y = (len(components_waterfall) - i) * y_spacing
        connector = ConnectionPatch(
            (cumulative_time, prev_y - bar_height/2),
            (cumulative_time, y_pos + bar_height/2),
            "data", "data",
            arrowstyle="-",
            color='#475569',
            linewidth=1.5,
            linestyle='--',
            alpha=0.6
        )
        ax_wf.add_artist(connector)
    
    cumulative_time += comp["time"]

# Timeline
timeline_ticks = np.arange(0, total_time_waterfall + 0.5, 0.5)
for tick in timeline_ticks:
    ax_wf.axvline(x=tick, color='#334155', linewidth=0.8, linestyle=':', alpha=0.5)
    ax_wf.text(tick, parent_y + 0.7, f"{tick:.1f}s", ha='center', va='bottom', fontsize=9, color='#94a3b8')

ax_wf.set_xlim(-0.3, total_time_waterfall + 0.8)
ax_wf.set_ylim(-0.5, parent_y + 1.2)
ax_wf.set_xticks([])
ax_wf.set_yticks([])
ax_wf.spines['left'].set_visible(False)
ax_wf.spines['right'].set_visible(False)
ax_wf.spines['top'].set_visible(False)
ax_wf.spines['bottom'].set_visible(False)

ax_wf.text(
    total_time_waterfall / 2, parent_y + 1.5,
    'RAG Pipeline - Sequential Execution Waterfall',
    ha='center', va='bottom',
    fontsize=16,
    fontweight='bold',
    color='#f1f5f9'
)

plt.tight_layout()
plt.savefig(output_dir / 'waterfall_trace.png', dpi=300, bbox_inches='tight', facecolor='#0f172a')
print(f"📊 Saved waterfall chart to: {output_dir / 'waterfall_trace.png'}\n")

plt.show()

print("✅ Analysis complete!\n")
print(f"📁 Results saved to: {output_dir.absolute()}")
print("\n" + "=" * 80)
print("📌 KEY FINDINGS:")
print("=" * 80)
print(f"⚡ TTFT P90: {ttft_p90:.2f}s (90% of users see response start within this time)")
print(f"⏱️  E2E P90:  {e2e_p90:.2f}s (90% of full responses complete within this time)")
print(f"🔢 Generator tokens: {success_df['generator_tokens'].sum():,} ({success_df['generator_tokens'].sum() / total_all_tokens * 100:.1f}% of total)")
print(f"📝 Rewriter tokens:  {success_df['rewriter_tokens'].sum():,} ({success_df['rewriter_tokens'].sum() / total_all_tokens * 100:.1f}% of total)")
print("=" * 80)