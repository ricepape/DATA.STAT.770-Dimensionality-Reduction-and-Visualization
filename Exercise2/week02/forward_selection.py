import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/vudinhthi2304/Desktop/DATASTAT770/DATA.STAT.770-Dimensionality-Reduction-and-Visualization/Exercise2/week02/noisy_sculpt_faces.txt')

# Extract features and angles
features = data[:, :-3]
angles = data[:, -3:]

# Part a)
def nearest_neighbor_predictor(features, angles):
    n = features.shape[0]
    predictions = np.zeros_like(angles)
    
    for i in range(n):
        # Compute the sum of squared differences
        distances = np.sum((features - features[i]) ** 2, axis=1)
        distances[i] = np.inf  # Exclude the current face
        
        # Find the nearest neighbor
        nearest_neighbor = np.argmin(distances)
        predictions[i] = angles[nearest_neighbor]
    
    return predictions

# Part b)
def leave_one_out_error(features, angles):
    predictions = nearest_neighbor_predictor(features, angles)
    errors = np.sum((predictions - angles) ** 2, axis=1)
    total_error = np.sum(errors)
    return total_error

def forward_selection(features, angles):
    n_features = features.shape[1]
    selected_features = []
    min_error = float('inf')
    best_features = None
    errors = []
    
    while True:
        best_feature = None
        for feature in range(n_features):
            if feature in selected_features:
                continue
            current_features = selected_features + [feature]
            current_error = leave_one_out_error(features[:, current_features], angles)
            if current_error < min_error:
                min_error = current_error
                best_feature = feature
                best_features = current_features
        
        if best_feature is None:
            break
        
        selected_features.append(best_feature)
        errors.append(min_error)
        print(f'Selected feature: {best_feature}, Current error: {min_error}')
    
    return best_features, min_error, errors

def forward_selection_variant(features, angles):
    n_features = features.shape[1]
    selected_features = []
    errors = []
    
    for _ in range(n_features):
        best_feature = None
        min_error = float('inf')
        
        for feature in range(n_features):
            if feature in selected_features:
                continue
            current_features = selected_features + [feature]
            current_error = leave_one_out_error(features[:, current_features], angles)
            if current_error < min_error:
                min_error = current_error
                best_feature = feature
        
        if best_feature is not None:
            selected_features.append(best_feature)
            errors.append(min_error)
            print(f'Selected feature: {best_feature}, Current error: {min_error}')
    
    return selected_features, errors

# Part c)
def forward_selection_variant(features, angles):
    n_features = features.shape[1]
    selected_features = []
    errors = []
    
    for _ in range(n_features):
        best_feature = None
        min_error = float('inf')
        
        for feature in range(n_features):
            if feature in selected_features:
                continue
            current_features = selected_features + [feature]
            current_error = leave_one_out_error(features[:, current_features], angles)
            if current_error < min_error:
                min_error = current_error
                best_feature = feature
        
        if best_feature is not None:
            selected_features.append(best_feature)
            errors.append(min_error)
            print(f'Selected feature: {best_feature}, Current error: {min_error}')
    
    return selected_features, errors

# Compute the leave-one-out error using all features
baseline_error = leave_one_out_error(features, angles)
print(f'Baseline leave-one-out error: {baseline_error}')

# Compute the leave-one-out error using all features
best_features, best_error, errors_list = forward_selection(features, angles)
print(f'Best leave-one-out error: {best_error}')
print(f'Number of features selected: {len(best_features)}')
print(f'Selected features: {best_features}')

# Run the variant of forward selection
selected_features, errors = forward_selection_variant(features, angles)
print(f'Order of features added: {selected_features}')
print(f'Errors with each number of features: {errors}')

# Plot the performance measure with respect to the number of features
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(errors_list) + 1), errors_list, label='Forward Selection')
plt.plot(range(1, len(errors) + 1), errors, label='Forward Selection Variant')
plt.xlabel('Number of Features')
plt.ylabel('Leave-One-Out Error')
plt.title('Performance Measure with Respect to Number of Features')
plt.legend()
plt.show()