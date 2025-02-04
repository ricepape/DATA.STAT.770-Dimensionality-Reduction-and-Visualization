import numpy as np

def compute_proportions(dimensions, num_points=10_000_000):
    results_hypersphere = {}
    results_shell = {}

    for d in dimensions:
        # Generate points uniformly in [-1, 1]^d
        points = np.random.uniform(-1, 1, size=(num_points, d))

        # Compute squared distances from the origin
        squared_distances = np.sqrt(np.sum(points**2, axis=1))

        # a) Proportion inside the hypersphere
        inside_hypersphere = squared_distances <= 1
        proportion_hypersphere = np.mean(inside_hypersphere)
        results_hypersphere[d] = proportion_hypersphere

        # b) Proportion inside the spherical shell
        inside_shell = (squared_distances >= 0.95) & (squared_distances <= 1)
        proportion_shell = np.sum(inside_shell) / np.sum(inside_hypersphere)
        results_shell[d] = proportion_shell

    return results_hypersphere, results_shell

# Dimensions to test
dimensions = [1, 2, 3, 4, 7, 11, 16]

# Run the computation
proportions_hypersphere, proportions_shell = compute_proportions(dimensions)

# Display results
# a) Proportion of points inside the hypersphere
print("Proportion of points inside the hypersphere:")
for d, proportion in proportions_hypersphere.items():
    print(f"Dimension {d}: {proportion:.6f}")

# b) Proportion of points inside the spherical shell
print("\nProportion of points inside the spherical shell:")
for d, proportion in proportions_shell.items():
    print(f"Dimension {d}: {proportion:.6f}")
