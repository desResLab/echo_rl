import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
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
            u = np.random.rand()
            early_step = -root_scalar(lambda x, u=u: cumulative_distribution_function(x, 0, 60, 5) - u,
                                      bracket=[0, 60]).root
            samples.append(early_step)
    return samples

# Generate samples
samples = generate_samples(0, 60, 1000, 5, 0.8)

# Plot histogram with density
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=60, color='skyblue', edgecolor='black', alpha=0.7, density=True)
plt.title('Distribution of the difference between Arrival Time and Scheduled Times', fontsize=18)
plt.xlabel('Time (minutes)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True)
plt.show()
