import numpy as np
from sklearn.decomposition import PCA

# Define simple trackable numbers
n_average_epochs = 2
n_channels = 4
n_times = 5

# Create an array with sequential numbers to easily track where each element goes
average_phase_epochs = np.arange(n_average_epochs * n_channels * n_times).reshape(n_average_epochs, n_channels, n_times)

print("Original average_phase_epochs:")
print(average_phase_epochs)
print("Original Shape:", average_phase_epochs.shape)

stacked_phase = np.concatenate(
    [average_phase_epochs[i] for i in range(n_average_epochs)],
    axis=1
)

print("\nStacked Phase (axis=1):")
print(stacked_phase)
print("Stacked Shape:", stacked_phase.shape)

input_pca = np.arange(n_average_epochs * 3 * n_times).reshape(n_average_epochs, 3, n_times)
print("PCA input: ")
print(input_pca)
print(input_pca.shape)

pca_stacked = np.concatenate(
    [input_pca[i] for i in range(2)],
    axis=1
).T

print("PCA output: ")
print(pca_stacked)
print(pca_stacked.shape)

pca_average_epochs = pca_stacked.reshape(
    n_average_epochs, n_times, 3
).transpose(0, 2, 1)

print("Original again: ")
print(pca_average_epochs)
print(pca_average_epochs.shape)

print("Are they equal?")
print(np.array_equal(pca_average_epochs, input_pca))