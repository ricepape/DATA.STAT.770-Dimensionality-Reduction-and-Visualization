import numpy as np
from scipy.stats import pearsonr

# Load the data
data = np.loadtxt('/Users/vudinhthi2304/Desktop/DATASTAT770/DATA.STAT.770-Dimensionality-Reduction-and-Visualization/Exercise2/week02/noisy_sculpt_faces.txt')

# 2.1. Pearson Correlation
# Extract features and angles
features = data[:, :-3]
angles = data[:, -3:]

# Compute Pearson correlation for each feature
correlations = []
for i in range(features.shape[1]):
    feature = features[:, i]
    corr = [abs(pearsonr(feature, angles[:, j])[0]) for j in range(angles.shape[1])]
    avg_corr = np.mean(corr)
    correlations.append(avg_corr)

# Rank features by average Pearson correlation
ranked_features = np.argsort(correlations)[::-1]
print(f'Ranked features by Pearson correlation: {ranked_features}')

# 2.2. Leave-One-Out Error
# Define the nearest neighbor predictor and leave-one-out error functions
def nearest_neighbor_predictor(features, angles):
    n = features.shape[0]
    predictions = np.zeros_like(angles)
    
    for i in range(n):
        distances = np.sum((features - features[i]) ** 2, axis=1)
        distances[i] = np.inf  # Exclude the current face
        nearest_neighbor = np.argmin(distances)
        predictions[i] = angles[nearest_neighbor]
    
    return predictions

def leave_one_out_error(features, angles):
    predictions = nearest_neighbor_predictor(features, angles)
    errors = np.sum((predictions - angles) ** 2, axis=1)
    total_error = np.sum(errors)
    return total_error

# Compute leave-one-out error for each feature
errors = []
for i in range(features.shape[1]):
    feature = features[:, i].reshape(-1, 1)
    error = leave_one_out_error(feature, angles)
    errors.append(error)

# Rank features by leave-one-out error
ranked_features = np.argsort(errors)
print(f'Ranked features by leave-one-out error: {ranked_features}')

# Assuming the forward_selection_variant function is defined in forward_selection.py
from forward_selection import forward_selection_variant

# Run the variant of forward selection
selected_features, _ = forward_selection_variant(features, angles)
print(f'Order of features added in forward selection variant: {selected_features}')

