import numpy as np
import matplotlib.pyplot as plt

# Parameters
r_data = np.array([0.001, 0.1, 1, 10, 100, 1000])
r_beta = 0.025
r_beta2 = 0.0025
r_beta3 = 0.00025

# Calculate r_p, r_p2, and r_p3
r_p = np.power(r_data, r_beta)  
r_p2 = np.power(r_data, r_beta2)
r_p3 = np.power(r_data, r_beta3)

# Print the results
print('r_p:', r_p)
print('r_p2:', r_p2)
print('r_p3:', r_p3)

# Plot the results
plt.plot(r_data, r_p, marker='o', linestyle='-', color='b', label=f'r_p = exp(-{r_beta} * r_data)')
plt.plot(r_data, r_p2, marker='x', linestyle='--', color='g', label=f'r_p2 = exp(-{r_beta2} * r_data)')
plt.plot(r_data, r_p3, marker='s', linestyle='-.', color='r', label=f'r_p3 = exp(-{r_beta3} * r_data)')

# Set axis scales and labels
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('r_data')
plt.ylabel('r_p')
plt.title('Exponential Decay with Different r_beta Values')

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()



