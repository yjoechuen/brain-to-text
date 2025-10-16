"""
Electrode-Temporal-Phoneme EDA Analysis for Brain-to-Text Dataset
================i===================================================

FOCUSED ANALYSES (Non-Redundant with Existing Work):
1. Phoneme Class Balance & Distribution (excluding padding tokens)
2. Temporal Autocorrelation Analysis (spatial vs temporal encoding)
3. Normalization Strategy Comparison (preprocessing selection)

Author: Neural Navigators Team
Dataset: Brain-to-Text '25 Kaggle Competition
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))
from src.data.hdf5_loader import HDF5DataLoader

# ============================================================================
# PHONEME ID TO NAME MAPPING
# ============================================================================

# ARPAbet phoneme mapping (from vocabulary)
PHONEME_NAMES = {
    0: 'AA', 1: 'AE', 2: 'AH', 3: 'AO', 4: 'AW', 5: 'AY', 6: 'B', 7: 'CH',
    8: 'D', 9: 'DH', 10: 'EH', 11: 'ER', 12: 'EY', 13: 'F', 14: 'G', 15: 'HH',
    16: 'IH', 17: 'IY', 18: 'JH', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'NG',
    24: 'OW', 25: 'OY', 26: 'P', 27: 'R', 28: 'S', 29: 'SH', 30: 'SIL', 31: 'T',
    32: 'TH', 33: 'UH', 34: 'UNK', 35: 'UW', 36: 'V', 37: 'W', 38: 'Y', 39: 'Z', 40: 'ZH'
}

# ============================================================================
# CLEAN VISUALIZATION CONFIG
# ============================================================================

DATA_DIR = Path(r"data/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final")
SESSION_PATH = DATA_DIR / "t15.2023.08.13" / "data_train.hdf5"
OUTPUT_DIR = Path("eda_outputs") / "electrode_temporal_phoneme"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal, clean design
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 2,
    'grid.color': '#DDDDDD',
    'grid.linestyle': ':',
    'grid.linewidth': 1,
    'grid.alpha': 0.5,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'legend.fontsize': 12,
    'font.sans-serif': ['Arial', 'Helvetica']
})

COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'purple': '#785EF0',
    'red': '#DC143C',
    'gray': '#7F7F7F'
}

REGION_COLORS = {
    'Ventral 6v': COLORS['blue'],
    'Area 4': COLORS['orange'],
    '55b': COLORS['green'],
    'Dorsal 6v': COLORS['purple']
}

DPI = 300

print("="*80)
print("ELECTRODE-TEMPORAL-PHONEME EDA ANALYSIS")
print("="*80)
print(f"Session: {SESSION_PATH.name}\n")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_session_data():
    """Load neural data and phoneme sequences."""
    print("[1/4] Loading data...")
    
    loader = HDF5DataLoader(verbose=False)
    session_data = loader.load_session_data(str(SESSION_PATH))
    
    all_features = []
    all_phonemes = []
    
    for trial in session_data.trials:
        all_features.append(trial.input_features)
        if trial.seq_class_ids is not None:
            all_phonemes.extend(trial.seq_class_ids)
    
    features_concat = np.vstack(all_features)
    print(f"  Loaded: {len(session_data.trials)} trials, {len(all_phonemes):,} phonemes\n")
    
    return features_concat, all_phonemes

# ============================================================================
# ANALYSIS 1: PHONEME DISTRIBUTION (CLEAN)
# ============================================================================

def analyze_phoneme_distribution(all_phonemes):
    """Clean phoneme distribution visualization."""
    print("[2/4] Phoneme distribution...")
    
    # Exclude padding (ID 0)
    real_phonemes = [p for p in all_phonemes if p != 0]
    phoneme_counts = Counter(real_phonemes)
    sorted_ph = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)
    
    total = len(real_phonemes)
    imbalance = sorted_ph[0][1] / sorted_ph[-1][1]
    
    print(f"  Real phonemes: {total:,} (excluded {len(all_phonemes)-total:,} padding)")
    print(f"  Imbalance: {imbalance:.0f}:1\n")
    
    # Clean visualization
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='white')
    
    ids = [p for p, _ in sorted_ph]
    percentages = [c/total*100 for _, c in sorted_ph]
    
    # 1. Main bar chart with phoneme IDs
    bars = axes[0].bar(range(len(ids)), percentages, color=COLORS['blue'],
                      edgecolor='white', linewidth=1, alpha=0.9)
    axes[0].axhline(1, color=COLORS['red'], linestyle='--', linewidth=2,
                   label='1% Threshold')
    
    # Show top 15 phoneme IDs on x-axis
    x_positions = range(min(15, len(ids)))
    x_labels = [f'ID{ids[i]}' for i in x_positions]
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    # Add direct labels on top 5 bars
    for i in range(min(5, len(ids))):
        axes[0].text(i, percentages[i] + max(percentages)*0.02, 
                    f'ID {ids[i]}\n{percentages[i]:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    axes[0].set_title('Phoneme Frequency Distribution', fontweight='bold', fontsize=17)
    axes[0].set_xlabel('Phoneme ID (Top 15 Shown)', fontweight='bold')
    axes[0].set_ylabel('Frequency (%)', fontweight='bold')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(True, axis='y')
    
    # 2. Cumulative curve
    cumulative = np.cumsum(percentages)
    axes[1].plot(range(len(ids)), cumulative, linewidth=4, color=COLORS['green'])
    axes[1].fill_between(range(len(ids)), cumulative, alpha=0.2, color=COLORS['green'])
    axes[1].axhline(80, color=COLORS['orange'], linestyle='--', linewidth=2)
    axes[1].set_title('Cumulative Coverage', fontweight='bold', fontsize=17)
    axes[1].set_xlabel('Number of Phonemes', fontweight='bold')
    axes[1].set_ylabel('Cumulative %', fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(True)
    axes[1].set_ylim([0, 105])
    
    # 3. Top 10 horizontal bars with phoneme names
    top_10 = ids[:10]
    top_pcts = percentages[:10]
    top_names = [PHONEME_NAMES.get(id, f'ID{id}') for id in top_10]
    
    y_pos = range(len(top_10))
    axes[2].barh(y_pos, top_pcts, color=COLORS['purple'], 
                edgecolor='white', linewidth=1, alpha=0.9)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([f'{name} (ID {id})' for name, id in zip(top_names, top_10)],
                            fontsize=11, fontweight='bold')
    axes[2].invert_yaxis()
    axes[2].set_title('Top 10 Most Common Phonemes', fontweight='bold', fontsize=17)
    axes[2].set_xlabel('Frequency (%)', fontweight='bold')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].grid(True, axis='x')
    
    # Add percentage labels on bars
    for i, pct in enumerate(top_pcts):
        axes[2].text(pct + 0.3, i, f'{pct:.1f}%', 
                    va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phoneme_distribution_analysis.png', 
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved\n")
    
    return {'imbalance_ratio': imbalance, 'total_real': total}

# ============================================================================
# ANALYSIS 2: TEMPORAL AUTOCORRELATION (CLEAN)
# ============================================================================

def analyze_temporal_autocorrelation(features_concat):
    """Clean temporal autocorrelation visualization."""
    print("[3/4] Temporal autocorrelation...")
    
    sample_electrodes = {'Ventral 6v': 0, 'Area 4': 64, '55b': 128, 'Dorsal 6v': 192}
    
    tc_autocorrs = {}
    sbp_autocorrs = {}
    
    for region, idx in sample_electrodes.items():
        # TC
        tc_sig = features_concat[:, idx]
        if np.std(tc_sig) > 0:
            tc_norm = (tc_sig - np.mean(tc_sig)) / np.std(tc_sig)
            tc_ac = np.correlate(tc_norm, tc_norm, mode='full')
            tc_ac = tc_ac[len(tc_ac)//2:] / tc_ac[len(tc_ac)//2]
            tc_autocorrs[region] = tc_ac[:50]
        
        # SBP
        sbp_sig = features_concat[:, idx + 256]
        if np.std(sbp_sig) > 0:
            sbp_norm = (sbp_sig - np.mean(sbp_sig)) / np.std(sbp_sig)
            sbp_ac = np.correlate(sbp_norm, sbp_norm, mode='full')
            sbp_ac = sbp_ac[len(sbp_ac)//2:] / sbp_ac[len(sbp_ac)//2]
            sbp_autocorrs[region] = sbp_ac[:50]
    
    tc_mean = np.mean(list(tc_autocorrs.values()), axis=0)
    sbp_mean = np.mean(list(sbp_autocorrs.values()), axis=0)
    
    tc_mem = np.where(tc_mean < 0.3)[0]
    tc_mem_len = tc_mem[0] if len(tc_mem) > 0 else 50
    
    sbp_mem = np.where(sbp_mean < 0.3)[0]
    sbp_mem_len = sbp_mem[0] if len(sbp_mem) > 0 else 50
    
    print(f"  TC memory: {tc_mem_len} steps ({tc_mem_len*20}ms)")
    print(f"  SBP memory: {sbp_mem_len} steps ({sbp_mem_len*20}ms)\n")
    
    # Clean visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor='white')
    
    lags = np.arange(50)
    
    # TC plot
    for region, autocorr in tc_autocorrs.items():
        axes[0].plot(lags, autocorr, linewidth=3, label=region, 
                    color=REGION_COLORS[region], alpha=0.9)
    
    axes[0].axhline(0.3, color=COLORS['red'], linestyle='--', 
                   linewidth=2, label='Threshold (0.3)')
    axes[0].set_title('Threshold Crossings', fontweight='bold', fontsize=18)
    axes[0].set_xlabel('Lag (timesteps, 20ms each)', fontweight='bold')
    axes[0].set_ylabel('Autocorrelation', fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # SBP plot
    for region, autocorr in sbp_autocorrs.items():
        axes[1].plot(lags, autocorr, linewidth=3, label=region, 
                    color=REGION_COLORS[region], alpha=0.9)
    
    axes[1].axhline(0.3, color=COLORS['red'], linestyle='--', 
                   linewidth=2, label='Threshold (0.3)')
    axes[1].set_title('Spike Band Power', fontweight='bold', fontsize=18)
    axes[1].set_xlabel('Lag (timesteps, 20ms each)', fontweight='bold')
    axes[1].set_ylabel('Autocorrelation', fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_autocorrelation_analysis.png', 
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved\n")
    
    return {'tc_memory_len': tc_mem_len, 'sbp_memory_len': sbp_mem_len}

# ============================================================================
# ANALYSIS 3: NORMALIZATION (CLEAN)
# ============================================================================

def analyze_normalization_strategies(features_concat):
    """Clean heatmap comparison."""
    print("[4/4] Normalization comparison...")
    
    # Extract and normalize
    tc_features = features_concat[:, :256]
    sbp_features = features_concat[:, 256:]
    
    tc_norm = np.zeros_like(tc_features)
    sbp_norm = np.zeros_like(sbp_features)
    for i in range(256):
        tc_norm[:, i] = (tc_features[:, i] - np.mean(tc_features[:, i])) / (np.std(tc_features[:, i]) + 1e-8)
        sbp_norm[:, i] = (sbp_features[:, i] - np.mean(sbp_features[:, i])) / (np.std(sbp_features[:, i]) + 1e-8)
    
    # Compute full correlation matrices
    corr_tc_tc = np.corrcoef(tc_norm.T)
    
    # Extract 64×64 blocks for each region pair (10 unique combinations)
    regions = ['Ventral 6v', 'Area 4', '55b', 'Dorsal 6v']
    region_ranges = [(0, 64), (64, 128), (128, 192), (192, 256)]
    
    # Create 10 64×64 heatmaps (upper triangle of 4×4 region grid)
    fig, axes = plt.subplots(4, 3, figsize=(18, 22), facecolor='white')
    
    # Flatten for easier indexing (we only need 10 of 12 subplots)
    axes_flat = axes.flatten()
    
    plot_idx = 0
    for i in range(4):
        for j in range(i, 4):  # Upper triangle only (includes diagonal)
            region_i, region_j = regions[i], regions[j]
            start_i, end_i = region_ranges[i]
            start_j, end_j = region_ranges[j]
            
            # Extract 64×64 block
            block = corr_tc_tc[start_i:end_i, start_j:end_j]
            
            # Mask diagonal if same region
            if i == j:
                np.fill_diagonal(block, np.nan)
            
            ax = axes_flat[plot_idx]
            
            # Plot with ultra-high saturation
            im = ax.imshow(block, cmap='RdYlBu_r', aspect='auto',
                          vmin=-0.05, vmax=0.05, interpolation='nearest')
            im.cmap.set_bad(color='white')
            
            # Title
            if i == j:
                title = f'{region_i}\n(Within-Region)'
                color = COLORS['blue']
            else:
                title = f'{region_i} ↔ {region_j}\n(Cross-Region)'
                color = COLORS['orange']
            
            ax.set_title(title, fontweight='bold', fontsize=13, color=color, pad=10)
            ax.set_xlabel(region_j, fontsize=10, fontweight='bold')
            ax.set_ylabel(region_i, fontsize=10, fontweight='bold')
            
            # Minimal ticks
            ax.set_xticks([0, 31, 63])
            ax.set_xticklabels(['0', '32', '63'], fontsize=8)
            ax.set_yticks([0, 31, 63])
            ax.set_yticklabels(['0', '32', '63'], fontsize=8)
            
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, 12):
        axes_flat[idx].axis('off')
    
    # Single colorbar for all
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, 
                label='Correlation', shrink=0.8)
    
    fig.suptitle('TC Feature Correlations: 10 Region-Pair Combinations (64×64 each)', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.98, 0.99])
    plt.savefig(OUTPUT_DIR / 'normalization_strategies_comparison.png', 
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Single overlay Q-Q plot (clean)
    fig2, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
    
    strategies = ['Raw', 'Global Z', 'Per-Feature Z', 'Robust']
    colors = [COLORS['gray'], COLORS['blue'], COLORS['green'], COLORS['orange']]
    widths = [2, 2, 5, 2]
    
    test_feat = features_concat[:, 330]
    test_data = [
        test_feat,
        (test_feat - np.mean(features_concat)) / np.std(features_concat),
        (test_feat - np.mean(test_feat)) / (np.std(test_feat) + 1e-8),
        (test_feat - np.median(test_feat)) / (np.percentile(test_feat, 75) - 
                                               np.percentile(test_feat, 25) + 1e-8)
    ]
    
    # Reference line
    ax.plot([-4, 4], [-4, 4], color=COLORS['red'], linestyle='--', 
           linewidth=3, alpha=0.7, label='Perfect Gaussian')
    
    # Overlay all strategies
    for data, strat, color, width in zip(test_data, strategies, colors, widths):
        sorted_data = np.sort(data)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
        ax.plot(theoretical, sorted_data, color=color, linewidth=width,
               alpha=0.8, label=strat)
    
    ax.set_title('Normality Assessment (All Strategies Overlaid)', 
                fontweight='bold', fontsize=18, pad=15)
    ax.set_xlabel('Theoretical Quantiles', fontweight='bold')
    ax.set_ylabel('Sample Quantiles', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'normalization_qq_plots.png', 
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved\n")
    
    return {'recommended': 'Per-Feature Z-Score'}

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute analyses."""
    
    features_concat, all_phonemes = load_session_data()
    
    results = {}
    results['phoneme'] = analyze_phoneme_distribution(all_phonemes)
    results['temporal'] = analyze_temporal_autocorrelation(features_concat)
    results['normalization'] = analyze_normalization_strategies(features_concat)
    
    # Summary
    summary_path = OUTPUT_DIR / 'analysis_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ELECTRODE-TEMPORAL-PHONEME EDA SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("KEY FINDINGS:\n\n")
        f.write(f"1. PHONEME DISTRIBUTION\n")
        f.write(f"   - Imbalance: {results['phoneme']['imbalance_ratio']:.0f}:1\n")
        f.write(f"   - Real phonemes: {results['phoneme']['total_real']:,}\n\n")
        
        f.write(f"2. TEMPORAL AUTOCORRELATION\n")
        f.write(f"   - TC memory: {results['temporal']['tc_memory_len']} steps\n")
        f.write(f"   - SBP memory: {results['temporal']['sbp_memory_len']} steps\n\n")
        
        f.write(f"3. NORMALIZATION\n")
        f.write(f"   - Recommended: {results['normalization']['recommended']}\n")
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
