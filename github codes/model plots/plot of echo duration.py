import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gamma
from scipy.integrate import quad
from scipy.optimize import root_scalar

fs = 20
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)

samples = []

# Parameters
a, b = 20, 150
k, theta = 10, 4.5

def gamma_pdf(x, k, theta):
    return gamma.pdf(x, k, scale=theta)

def cumulative_distribution_function(x, a, b, k, theta):
    integral, _ = quad(gamma_pdf, a, x, args=(k, theta))
    normalization_factor, _ = quad(gamma_pdf, a, b, args=(k, theta))
    return integral / normalization_factor

# Sampling via root-finding inverse transform
for _ in range(10000):
    U = np.random.rand()
    spell_time = root_scalar(lambda x, u=U: cumulative_distribution_function(x, a, b, k, theta) - u,
                             bracket=[a, b]).root
    samples.append(spell_time)



# Plotting the gamma PDF for comparison
x = np.linspace(a, b, 1000)
pdf_values = gamma.pdf(x, k, scale=theta)
normalization_factor = np.trapz(pdf_values, x)

plt.xlabel('Time (minutes)', fontsize=fs)
plt.ylabel('Density', fontsize=fs)
plt.hist(samples, bins=50, color='b', edgecolor='black', alpha=0.7, density=True)
# plt.title('Echo test duration', fontsize=fs)
plt.tick_params(axis='both', which='both', labelsize=fs-2)
plt.grid(True)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', 'echo duration.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()