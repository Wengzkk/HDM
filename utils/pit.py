import matplotlib.pyplot as plt
import numpy as np

# Generate histogram data for plotting
green_data = np.random.beta(2, 5, 1000)  # Skewed distribution for green
red_data = np.random.normal(1, 0.2, 1000)  # Normal distribution for red

# Scale red to half the height of green and clip values
red_data = red_data[red_data >= 0]  # Remove negative values
red_data = red_data[red_data <= 1.5]  # Limit red to 0-1.5 range
red_data *= 0.5  # Scale red down

# Plot the histograms
plt.hist(green_data, bins=50, density=True, alpha=0.7, color='green', label='Green Distribution')
plt.hist(red_data, bins=50, density=True, alpha=0.7, color='red', label='Red Distribution')

# Adjust aesthetics (no axis numbers, no labels)
plt.axis('off')

# Show the plot
plt.show()
