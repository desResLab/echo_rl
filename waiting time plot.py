import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
from scipy.optimize import root_scalar

def exponential_pdf(x, theta):
    """Probability Density Function (PDF) of the exponential distribution."""
    return np.exp(-x / theta) / theta

def cumulative_distribution_function(x, a, b, theta):
    """Calculate the cumulative distribution function."""
    integral = quad(exponential_pdf, a, x, args=(theta,))[0] / quad(exponential_pdf, a, b, args=(theta,))[0]
    return integral

def generate_samples(a, b, N, theta, mixture_ratio):
    """Generate samples from a mixture of exponential and inverse cutted exponential distribution."""
    samples = []
    for _ in range(N):
        # Generate a uniform random number to determine which distribution to sample from
        choice = np.random.choice([0, 1], p=[mixture_ratio, 1 - mixture_ratio])
        if choice == 0:
            # Sample from exponential distribution
            samples.append(np.random.exponential(20))
        else:
            # Sample from inverse cutted exponential distribution with a minus sign
            U = np.random.rand()
            early_step = -root_scalar(lambda x, u=U: cumulative_distribution_function(x, 0, 60, 5) - u,
                                      bracket=[0, 60]).root
            samples.append(early_step)
    return samples
fs=20
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
# Generate samples
samples = generate_samples(0, 60, 1000, 5, 0.8)

# Plot histogram with density
plt.hist(samples, bins=50, color='b', edgecolor='black', alpha=0.7, density=True)
plt.title('Difference between Arrival Time and Scheduled Time', fontsize=fs)
plt.xlabel('Time (minutes)', fontsize=fs)
plt.ylabel('Density', fontsize=fs)
plt.tick_params(axis='both', which='both', labelsize=fs-2)
plt.grid(True)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', 'waiting time.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()

