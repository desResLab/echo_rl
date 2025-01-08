import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gamma

fs = 20
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)

# Parameters
a, b = 20, 150
k, theta = 10, 4.5

# Generate gamma-distributed samples
raw_samples = gamma.rvs(k, scale=theta, size=5000)

# Filter samples to be within the interval [20, 150]
samples = raw_samples[(raw_samples >= a) & (raw_samples <= b)]

# If we have less than 1000 samples after filtering, generate more
while len(samples) < 1000:
    more_samples = gamma.rvs(k, scale=theta, size=1000)
    samples = np.concatenate((samples, more_samples[(more_samples >= a) & (more_samples <= b)]))

samples = samples[:100000]  # Ensure exactly 1000 samples



# Plotting the gamma PDF for comparison
x = np.linspace(a, b, 1000)
pdf_values = gamma.pdf(x, k, scale=theta)
normalization_factor = np.trapz(pdf_values, x)

plt.xlabel('Time (minutes)', fontsize=fs)
plt.ylabel('Density', fontsize=fs)
plt.hist(samples, bins=50, color='b', edgecolor='black', alpha=0.7, density=True)
plt.title('Echo test duration', fontsize=fs)
plt.tick_params(axis='both', which='both', labelsize=fs-2)
plt.grid(True)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', 'echo duration.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()