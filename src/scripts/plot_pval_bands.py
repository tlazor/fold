import numpy as np
import matplotlib.pyplot as plt
import re

from print_helper import save_fig

def parse_band_data(filename):
    bands = []
    metrics = ['fsi', 'lex_sim', 'mut_int', 'pho_sim']
    p_pvals = {metric: [] for metric in metrics}
    p_coefs = {metric: [] for metric in metrics}
    
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
            
            # Extract p-values and coefficients for each metric
            for metric in metrics:
                pattern = f'\\| {metric}\\s+\\|\\s+([\\d.-]+)\\s+\\|\\s+([\\d.-]+)\\s+\\|'
                match = re.search(pattern, section)
                if match:
                    p_coefs[metric].append(float(match.group(1)))
                    p_pvals[metric].append(float(match.group(2)))
    
    return np.array(bands), p_pvals, p_coefs

def plot_pvalues(bands, p_pvals, p_coefs, plot_type):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Plot p-values on top subplot
    for metric, values in p_pvals.items():
        ax1.plot(bands, values, marker='o', alpha=0.5, label=metric)
    
    ax1.set_ylabel('p-value')
    ax1.set_title(f'P-values and Coefficients Across Frequency Bands ({plot_type})')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim([0,1])
    ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
    
    # Plot coefficients on bottom subplot
    for metric, values in p_coefs.items():
        ax2.plot(bands, values, marker='o', alpha=0.5, label=metric)
    
    ax2.set_xlabel('Frequency Band Midpoint')
    ax2.set_ylabel('Coefficient')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_fig(f"pval_coef_bands_{plot_type}")
    plt.close()

def main():
    filename = 'freq_.04_embedding_output.txt'
    # Extract type from filename
    plot_type = 'embedding' if 'embedding' in filename else 'likelihood'
    bands, p_pvals, p_coefs = parse_band_data(filename)
    plot_pvalues(bands, p_pvals, p_coefs, plot_type)

if __name__ == "__main__":
    main() 