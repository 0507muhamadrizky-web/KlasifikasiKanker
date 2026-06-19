import sys
import numpy as np
sys.path.insert(0, r'D:\python_packages')
import h5py

class NumpyANN:
    def __init__(self, h5_path):
        self.weights = {}
        with h5py.File(h5_path, 'r') as f:
            w = f['model_weights']
            
            # Dense 1 (128)
            self.weights['d1_k'] = w['dense']['sequential']['dense']['kernel'][:]
            self.weights['d1_b'] = w['dense']['sequential']['dense']['bias'][:]
            self.weights['bn1_gamma'] = w['batch_normalization']['sequential']['batch_normalization']['gamma'][:]
            self.weights['bn1_beta'] = w['batch_normalization']['sequential']['batch_normalization']['beta'][:]
            self.weights['bn1_mm'] = w['batch_normalization']['sequential']['batch_normalization']['moving_mean'][:]
            self.weights['bn1_mv'] = w['batch_normalization']['sequential']['batch_normalization']['moving_variance'][:]
            
            # Dense 2 (64)
            self.weights['d2_k'] = w['dense_1']['sequential']['dense_1']['kernel'][:]
            self.weights['d2_b'] = w['dense_1']['sequential']['dense_1']['bias'][:]
            self.weights['bn2_gamma'] = w['batch_normalization_1']['sequential']['batch_normalization_1']['gamma'][:]
            self.weights['bn2_beta'] = w['batch_normalization_1']['sequential']['batch_normalization_1']['beta'][:]
            self.weights['bn2_mm'] = w['batch_normalization_1']['sequential']['batch_normalization_1']['moving_mean'][:]
            self.weights['bn2_mv'] = w['batch_normalization_1']['sequential']['batch_normalization_1']['moving_variance'][:]
            
            # Dense 3 (32)
            self.weights['d3_k'] = w['dense_2']['sequential']['dense_2']['kernel'][:]
            self.weights['d3_b'] = w['dense_2']['sequential']['dense_2']['bias'][:]
            self.weights['bn3_gamma'] = w['batch_normalization_2']['sequential']['batch_normalization_2']['gamma'][:]
            self.weights['bn3_beta'] = w['batch_normalization_2']['sequential']['batch_normalization_2']['beta'][:]
            self.weights['bn3_mm'] = w['batch_normalization_2']['sequential']['batch_normalization_2']['moving_mean'][:]
            self.weights['bn3_mv'] = w['batch_normalization_2']['sequential']['batch_normalization_2']['moving_variance'][:]
            
            # Dense 4 (1)
            self.weights['d4_k'] = w['dense_3']['sequential']['dense_3']['kernel'][:]
            self.weights['d4_b'] = w['dense_3']['sequential']['dense_3']['bias'][:]

    def predict(self, x, verbose=0):
        # Layer 1
        x = np.dot(x, self.weights['d1_k']) + self.weights['d1_b']
        x = np.maximum(0, x)  # ReLU
        x = self.weights['bn1_gamma'] * (x - self.weights['bn1_mm']) / np.sqrt(self.weights['bn1_mv'] + 1e-3) + self.weights['bn1_beta']
        
        # Layer 2
        x = np.dot(x, self.weights['d2_k']) + self.weights['d2_b']
        x = np.maximum(0, x)  # ReLU
        x = self.weights['bn2_gamma'] * (x - self.weights['bn2_mm']) / np.sqrt(self.weights['bn2_mv'] + 1e-3) + self.weights['bn2_beta']
        
        # Layer 3
        x = np.dot(x, self.weights['d3_k']) + self.weights['d3_b']
        x = np.maximum(0, x)  # ReLU
        x = self.weights['bn3_gamma'] * (x - self.weights['bn3_mm']) / np.sqrt(self.weights['bn3_mv'] + 1e-3) + self.weights['bn3_beta']
        
        # Layer 4
        x = np.dot(x, self.weights['d4_k']) + self.weights['d4_b']
        x = 1 / (1 + np.exp(-x))  # Sigmoid
        
        return np.array([x])

# Test
model = NumpyANN('models/ann_model.h5')
# dummy data (30 features)
dummy = np.random.rand(1, 30)
print("Predict:", model.predict(dummy))
