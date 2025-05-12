import numpy as np
import matplotlib.pyplot as plt
import re

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

def plot_metrics(bands, p_pvals, p_coefs):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot p-values on the left y-axis
    ax1.set_xlabel('Frequency Band Midpoint')
    ax1.set_ylabel('p-value', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')
    
    # Plot p-values
    for metric, values in p_pvals.items():
        ax1.plot(bands, values, marker='o', linestyle='--', label=f'{metric} (p-value)',
                color=f'tab:{["blue", "orange", "green", "red"][list(p_pvals.keys()).index(metric)]}')
    
    # Add horizontal line at p=0.05 for reference
    ax1.axhline(y=0.05, color='tab:blue', linestyle=':', alpha=0.5, label='p=0.05')
    
    # Create second y-axis for coefficients
    ax2 = ax1.twinx()
    ax2.set_ylabel('p-coefficient', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Plot coefficients
    for metric, values in p_coefs.items():
        ax2.plot(bands, values, marker='s', linestyle='-', label=f'{metric} (coef)',
                color=f'tab:{["blue", "orange", "green", "red"][list(p_coefs.keys()).index(metric)]}')
    
    # Add horizontal line at y=0 for reference
    ax2.axhline(y=0, color='tab:red', linestyle=':', alpha=0.5, label='coef=0')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))
    
    plt.title('P-values and Coefficients Across Frequency Bands')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pval_coef_bands.png', bbox_inches='tight')
    plt.close()

def main():
    filename = 'freq_.1_embedding_output.txt'
    bands, p_pvals, p_coefs = parse_band_data(filename)
    plot_metrics(bands, p_pvals, p_coefs)

if __name__ == "__main__":
    main() 