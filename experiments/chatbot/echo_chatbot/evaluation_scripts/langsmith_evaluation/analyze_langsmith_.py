"""
Extract and analyze efficiency metrics from LangSmith
Generates custom charts for thesis
"""

import os
from langsmith import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Initialize LangSmith client
client = Client()

print("📊 Fetching data from LangSmith...\n")

# Get all runs from your project
runs = list(client.list_runs(
    project_name="Ancient-Egypt-RAG",
    is_root=True  # Only get root runs (not sub-traces)
))

print(f"✅ Fetched {len(runs)} runs\n")

# ============================================================================
# Extract Metrics
# ============================================================================

data = []
for run in runs:
    # Get metadata
    metadata = run.extra.get('metadata', {}) if run.extra else {}
    
    # Basic metrics
    row = {
        'run_id': str(run.id),
        'entity_type': metadata.get('entity_type', 'unknown'),
        'entity_name': metadata.get('entity_name', 'unknown'),
        'query': run.inputs.get('query', '') if run.inputs else '',
        'total_latency': run.latency if run.latency else 0,  # in seconds
        'total_tokens': run.total_tokens or 0,
        'prompt_tokens': run.prompt_tokens or 0,
        'completion_tokens': run.completion_tokens or 0,
        'success': run.status == 'success',
        'start_time': run.start_time,
        'end_time': run.end_time,
    }
    
    # Try to extract per-node timings from child runs
    try:
        child_runs = list(client.list_runs(trace_id=run.trace_id))
        
        for child in child_runs:
            if not child.name:
                continue
                
            node_name = child.name.lower()
            
            if 'rewrite' in node_name:
                row['rewrite_time'] = child.latency or 0
            elif 'retrieve' in node_name or 'retrieval' in node_name:
                row['retrieve_time'] = child.latency or 0
            elif 'rerank' in node_name:
                row['rerank_time'] = child.latency or 0
            elif 'generate' in node_name or 'generator' in node_name:
                row['generate_time'] = child.latency or 0
    except Exception as e:
        print(f"⚠️  Could not extract child runs for {run.id}: {e}")
    
    data.append(row)

df = pd.DataFrame(data)

# Fill missing node times with 0
for col in ['rewrite_time', 'retrieve_time', 'rerank_time', 'generate_time']:
    if col not in df.columns:
        df[col] = 0
    else:
        df[col] = df[col].fillna(0)

# Calculate TPS (Tokens Per Second) - only if generate_time > 0
df['tps'] = df.apply(
    lambda row: row['completion_tokens'] / row['generate_time'] if row['generate_time'] > 0 else 0,
    axis=1
)

# Save to CSV
output_dir = Path("langsmith_analysis")
output_dir.mkdir(exist_ok=True)

df.to_csv(output_dir / "efficiency_metrics.csv", index=False)
print(f"💾 Saved metrics to: {output_dir / 'efficiency_metrics.csv'}\n")

# ============================================================================
# Calculate Key Statistics
# ============================================================================

print("=" * 80)
print("📈 EFFICIENCY METRICS SUMMARY")
print("=" * 80 + "\n")

# Filter successful runs
success_df = df[df['success'] == True]

if len(success_df) == 0:
    print("❌ No successful runs found!")
    exit(1)

# Overall Latency
print("⏱️  LATENCY METRICS:")
print(f"  • Mean:   {success_df['total_latency'].mean():.2f}s")
print(f"  • Median: {success_df['total_latency'].median():.2f}s")
print(f"  • P50:    {success_df['total_latency'].quantile(0.50):.2f}s")
print(f"  • P90:    {success_df['total_latency'].quantile(0.90):.2f}s")
print(f"  • P99:    {success_df['total_latency'].quantile(0.99):.2f}s")
print(f"  • Min:    {success_df['total_latency'].min():.2f}s")
print(f"  • Max:    {success_df['total_latency'].max():.2f}s\n")

# Component Breakdown (Average)
print("🔧 COMPONENT LATENCY BREAKDOWN (Average):")
components = ['rewrite_time', 'retrieve_time', 'rerank_time', 'generate_time']
component_names = ['Rewriter', 'Retriever', 'Reranker', 'Generator']

# Check if we have component data
has_component_data = any(success_df[c].sum() > 0 for c in components)

if has_component_data:
    total_component_time = sum(success_df[c].mean() for c in components)
    
    for comp, name in zip(components, component_names):
        avg_time = success_df[comp].mean()
        percentage = (avg_time / total_component_time * 100) if total_component_time > 0 else 0
        print(f"  • {name:12s}: {avg_time:.3f}s ({percentage:.1f}%)")
else:
    print("  ⚠️  Per-node timing data not available in LangSmith traces")
    print("  → Showing total latency only")

print()

# Token Metrics
print("🔢 TOKEN METRICS:")
print(f"  • Total tokens:      {success_df['total_tokens'].sum():,}")
print(f"  • Avg per query:     {success_df['total_tokens'].mean():.0f}")
print(f"  • Prompt tokens:     {success_df['prompt_tokens'].sum():,}")
print(f"  • Completion tokens: {success_df['completion_tokens'].sum():,}")

if success_df['tps'].sum() > 0:
    print(f"  • Avg TPS:           {success_df[success_df['tps'] > 0]['tps'].mean():.1f} tokens/sec")
else:
    # Calculate TPS from total tokens / total latency
    total_completion = success_df['completion_tokens'].sum()
    total_latency = success_df['total_latency'].sum()
    tps_estimate = total_completion / total_latency if total_latency > 0 else 0
    print(f"  • Avg TPS (est):     {tps_estimate:.1f} tokens/sec")

print()

# Success Rate
print("✅ SUCCESS RATE:")
print(f"  • Successful: {len(success_df)}/{len(df)} ({len(success_df)/len(df)*100:.1f}%)\n")

print("=" * 80 + "\n")

# ============================================================================
# Generate Charts for Thesis
# ============================================================================

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Determine number of subplots based on available data
if has_component_data:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle('RAG System Efficiency Metrics', fontsize=16, fontweight='bold')

# Chart 1: Latency Distribution with Percentiles
ax = axes[0, 0]
ax.hist(success_df['total_latency'], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
p50 = success_df['total_latency'].quantile(0.50)
p90 = success_df['total_latency'].quantile(0.90)
p99 = success_df['total_latency'].quantile(0.99)
ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.2f}s')
ax.axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'P90: {p90:.2f}s')
ax.axvline(p99, color='red', linestyle='--', linewidth=2, label=f'P99: {p99:.2f}s')
ax.set_xlabel('Latency (seconds)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('End-to-End Latency Distribution', fontweight='bold')
ax.legend()

# Chart 2: Component Breakdown OR Token Distribution
ax = axes[0, 1]
if has_component_data:
    component_avg = [success_df[c].mean() for c in components]
    ax.bar(component_names, component_avg, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black')
    ax.set_ylabel('Average Time (seconds)', fontweight='bold')
    ax.set_title('Latency Breakdown by Component', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    
    # Add percentage labels on bars
    for i, (name, val) in enumerate(zip(component_names, component_avg)):
        percentage = (val / sum(component_avg) * 100) if sum(component_avg) > 0 else 0
        ax.text(i, val + 0.05, f'{percentage:.1f}%', ha='center', fontweight='bold')
else:
    # Show token distribution instead
    ax.bar(['Prompt', 'Completion'], 
           [success_df['prompt_tokens'].sum(), success_df['completion_tokens'].sum()],
           color=['skyblue', 'lightcoral'], edgecolor='black')
    ax.set_ylabel('Total Tokens', fontweight='bold')
    ax.set_title('Token Usage Distribution', fontweight='bold')

# Chart 3: Latency CDF (Cumulative Distribution)
ax = axes[1, 0]
sorted_latencies = np.sort(success_df['total_latency'])
cumulative = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
ax.plot(sorted_latencies, cumulative * 100, linewidth=2, color='steelblue')
ax.axhline(50, color='green', linestyle='--', alpha=0.7, label=f'P50: {p50:.2f}s')
ax.axhline(90, color='orange', linestyle='--', alpha=0.7, label=f'P90: {p90:.2f}s')
ax.axhline(99, color='red', linestyle='--', alpha=0.7, label=f'P99: {p99:.2f}s')
ax.set_xlabel('Latency (seconds)', fontweight='bold')
ax.set_ylabel('Cumulative Percentage (%)', fontweight='bold')
ax.set_title('Latency Cumulative Distribution (CDF)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Chart 4: Latency by Entity Type
ax = axes[1, 1]
entity_latencies = success_df.groupby('entity_type')['total_latency'].agg(['mean', 'count'])
entity_latencies = entity_latencies[entity_latencies['count'] > 0].sort_values('mean')

if len(entity_latencies) > 0:
    ax.barh(entity_latencies.index, entity_latencies['mean'], 
            color=['coral', 'skyblue'], edgecolor='black')
    ax.set_xlabel('Average Latency (seconds)', fontweight='bold')
    ax.set_title('Latency by Entity Type', fontweight='bold')
    
    # Add count labels
    for i, (idx, row) in enumerate(entity_latencies.iterrows()):
        ax.text(row['mean'] + 0.1, i, f"n={int(row['count'])}", va='center')
else:
    ax.text(0.5, 0.5, 'No entity type data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Latency by Entity Type', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
print(f"📊 Saved chart to: {output_dir / 'efficiency_analysis.png'}\n")

plt.show()

print("✅ Analysis complete!\n")
print(f"📁 Results saved to: {output_dir.absolute()}")