import numpy as np
import matplotlib.pyplot as plt

#arr = np.array([0.001, 0.1, 1, 10, 100, 1000])

# make array starting from 1 to 1000 with step of 5
arr = np.arange(1, 1000, 5)

log_data = np.log(arr)
exp_data = np.exp(arr)

# Plot the results
#plt.plot(log_data, arr, marker='o', linestyle='-', color='b', label=f'log(r_data)')
plt.plot(exp_data, arr, marker='x', linestyle='--', color='g', label=f'exponential(r_data)')
#plt.plot(r_data, r_p3, marker='s', linestyle='-.', color='r', label=f'r_p3 = exp(-{r_beta3} * r_data)')

# Set axis scales and labels
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('r_data')
plt.ylabel('r_p')
plt.title('log')

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

