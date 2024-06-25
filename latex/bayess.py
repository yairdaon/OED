import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({'font.size': 16})

# Set up the x-axis
x = np.linspace(30, 100, 1000)

# Prior distribution (blue)
prior_mean, prior_std = 60, 10
prior = norm.pdf(x, prior_mean, prior_std)

# Single observation (green dot)
observation = 75

# Likelihood (green)
likelihood_mean = observation
likelihood_std = 5  # Adjust this value to change the width of the likelihood curve
likelihood = norm.pdf(x, likelihood_mean, likelihood_std)

# Posterior distribution (purple)
posterior_mean = (prior_mean/prior_std**2 + observation/likelihood_std**2) / (1/prior_std**2 + 1/likelihood_std**2)
posterior_std = np.sqrt(1 / (1/prior_std**2 + 1/likelihood_std**2))
posterior = norm.pdf(x, posterior_mean, posterior_std)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, prior, 'b-', label='Prior (m)')
plt.plot(x, likelihood, 'g-', label='Likelihood (d|m)')
plt.plot(x, posterior, 'red', label='Posterior (m|d)')

# Plot single observation as green dot on x-axis
plt.plot(observation, 0, 'go', markersize=10, label='Observation (d)')

# Customize the plot
# plt.xlabel('μ')
# plt.ylabel('Probability Density')
# plt.title('Bayesian Inference with Single Observation')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Set axis limits
plt.ylim(-0.005, 0.12)
plt.xlim(30, 100)


plt.xticks([])
plt.yticks([])

# Remove x and y axis labels
plt.xlabel('')
plt.ylabel('')

plt.box(False)
    
# Show the plot
plt.tight_layout()
plt.savefig('bayes.png')
#plt.show()
