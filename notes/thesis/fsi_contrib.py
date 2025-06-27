from datetime import datetime
import matplotlib.pyplot as plt

# Data provided by the user
data = {
    "ar":   -0.004629,
    "bg":   -0.000100,
    "cs":   -0.000068,
    "da":    0.031394,
    "de":    0.021365,
    "el":    0.000136,
    "es":    0.028867,
    "fi":   -0.000124,
    "fr":    0.033372,
    "he":    0.000177,
    "hi":    0.000458,
    "hu":    0.000115,
    "id":   -0.017577,
    "it":    0.014551,
    "ja":    0.107195,
    "ko":    0.079914,
    "lt":   -0.000011,
    "ne":    0.000622,
    "nl":   -0.030934,
    "no":    0.054716,
    "pl":    0.000006,
    "pt":    0.067782,
    "ro":   -0.006884,
    "ru":   -0.000230,
    "sk":   -0.000106,
    "sl":   -0.000238,
    "sq":   -0.000049,
    "sr":   -0.000329,
    "sv":    0.026288,
    "th":    0.000421,
    "tr":    0.000535,
    "vi":    0.000592,
    "zh":    0.060918,
}

# Extracting keys and values, preserving the order provided
languages = list(data.keys())
values = list(data.values())

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(languages, values)
plt.xticks(rotation=90)
plt.xlabel('Language Code')
plt.ylabel('Point-wise Pearson Contribution')
plt.title('Point-wise Pearson Contribution by Language Code')
plt.tight_layout()

# Display the plot
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
plt.savefig(f"fsi_pearson_contrib_{current_time}.png")
plt.close()
