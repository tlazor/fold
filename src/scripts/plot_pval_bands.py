import numpy as np
import matplotlib.pyplot as plt
import re

from print_helper import save_fig

def parse_band_data(filename):
    bands = []
    metrics = ['fsi', 'lex_sim', 'mut_int', 'pho_sim']
    p_pvals = {metric: [] for metric in metrics}
    p_coefs = {metric: [] for metric in metrics}
    s_pvals = {metric: [] for metric in metrics}
    s_coefs = {metric: [] for metric in metrics}
    
    with open(filename, 'r') as f:
        content = f.read()
        
    # Split content into band sections
    band_sections = content.split('band=')
    
    for section in band_sections[1:]:  # Skip first empty split
        # Extract band range
        band_match = re.search(r'\(np\.float64\(([\d.]+)\), np\.float64\(([\d.]+)\)\)', section)
        if band_match:
            band_start = float(band_match.group(1))
            band_end = float(band_match.group(2))
            bands.append((band_start + band_end) / 2)  # Use midpoint of band
            
            # Extract p-values and coefficients
            for metric in metrics:
                # Parse p-values and coefficients
                p_pattern = f'\\| {metric}\\s+\\|\\s+(?P<p_coef>[\\d.-]+)\\s+\\|\\s+(?P<p_pval>[\\d.-]+)\\s+\\|\\s+(?P<s_coef>[\\d.-]+)\\s+\\|\\s+(?P<s_pval>[\\d.-]+)\\s+\\|'
                p_match = re.search(p_pattern, section)
                if p_match:
                    p_coefs[metric].append(float(p_match.group('p_coef')))
                    p_pvals[metric].append(float(p_match.group('p_pval')))

                    s_coefs[metric].append(float(p_match.group('s_coef')))
                    s_pvals[metric].append(float(p_match.group('s_pval')))
    
    print(p_pvals)
    print(p_coefs)
    print(s_pvals)
    print(s_coefs)
    return np.array(bands), p_pvals, p_coefs, s_pvals, s_coefs

def plot_pvalues(bands, p_pvals, p_coefs, s_pvals, s_coefs, plot_type):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot p-values on top left subplot
    for metric, values in p_pvals.items():
        ax1.plot(bands, values, marker='o', alpha=0.5, label=metric)
    
    ax1.set_ylabel('P-Value')
    ax1.set_title(f'Pearson Correlation Across Frequency Bands ({plot_type})')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim([0,1])
    ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
    
    # Plot p-coefficients on top right subplot
    for metric, values in p_coefs.items():
        ax2.plot(bands, values, marker='o', alpha=0.5, label=metric)
    
    ax2.set_ylabel('Coefficient')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim([-1,1])
    
    # Plot s-values on bottom left subplot
    for metric, values in s_pvals.items():
        ax3.plot(bands, values, marker='o', alpha=0.5, label=metric)
    
    ax3.set_xlabel('Frequency Band Midpoint')
    ax3.set_ylabel('P-Value')
    ax3.set_title(f'Spearman Correlation Across Frequency Bands ({plot_type})')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim([0,1])
    ax3.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
    
    # Plot s-coefficients on bottom right subplot
    for metric, values in s_coefs.items():
        ax4.plot(bands, values, marker='o', alpha=0.5, label=metric)
    
    ax4.set_xlabel('Frequency Band Midpoint')
    ax4.set_ylabel('Coefficient')
    ax4.legend()
    ax4.grid(True)
    ax4.set_ylim([-1,1])
    
    plt.tight_layout()
    save_fig(f"pval_coef_bands_{plot_type}")
    plt.close()

def main():
    filename = 'freq_.04_embedding_output.txt'
    # Extract type from filename
    plot_type = 'embedding' if 'embedding' in filename else 'likelihood'
    bands, p_pvals, p_coefs, s_pvals, s_coefs = parse_band_data(filename)
    plot_pvalues(bands, p_pvals, p_coefs, s_pvals, s_coefs, plot_type)

if __name__ == "__main__":
    main() 