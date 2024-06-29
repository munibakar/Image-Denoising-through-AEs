import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import itertools
import json
import argparse

parser = argparse.ArgumentParser(description="Specify whether to perform hyperparameter optimization.")
parser.add_argument("--hyperparameters", type=str, default="off", choices=["on", "off"],
                    help="Specify whether to perform hyperparameter optimization. Options: 'on' or 'off'. Default: 'off'.")
args = parser.parse_args()

digits = load_digits()

X = digits.data
y = digits.target

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

def add_noise(img, noise_level):
    return img + noise_level * np.random.randn(*img.shape)

X_train_noisy_low = add_noise(X_train_scaled, 0.1)
X_train_noisy_medium = add_noise(X_train_scaled, 0.5)
X_train_noisy_high = add_noise(X_train_scaled, 1.0)

X_val_noisy_low = add_noise(X_val_scaled, 0.1)
X_val_noisy_medium = add_noise(X_val_scaled, 0.5)
X_val_noisy_high = add_noise(X_val_scaled, 1.0)

X_test_noisy_low = add_noise(X_test_scaled, 0.1)
X_test_noisy_medium = add_noise(X_test_scaled, 0.5)
X_test_noisy_high = add_noise(X_test_scaled, 1.0)

if args.hyperparameters == "on":
    hyperparameters_grid = {
        'hidden_layer_sizes': [(128, 64), (256, 128), (64, 32)],
        'activation': ['relu', 'logistic'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.0001, 0.001, 0.01],
        'solver': ['adam', 'sgd']
    }

    best_score = float('inf')
    best_hyperparameters = None

    for hidden_layer_sizes, activation, learning_rate_init, alpha, solver in itertools.product(*hyperparameters_grid.values()):
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate_init=learning_rate_init, alpha=alpha, solver=solver, max_iter=50, random_state=42)
        model.fit(X_train_noisy_low, X_train_scaled)

        val_loss = np.mean((model.predict(X_val_noisy_low) - X_val_scaled) ** 2)

        if val_loss < best_score:
            best_score = val_loss
            best_hyperparameters = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'learning_rate': learning_rate_init,
                'alpha': alpha,
                'solver': solver
            }

    print("Best Hyperparameters:", best_hyperparameters)

    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_hyperparameters, f)

else:
    with open('best_hyperparameters.json', 'r') as f:
        best_hyperparameters = json.load(f)

best_model = MLPRegressor(hidden_layer_sizes=best_hyperparameters['hidden_layer_sizes'],
                          activation=best_hyperparameters['activation'],
                          learning_rate_init=best_hyperparameters['learning_rate'],
                          alpha=best_hyperparameters['alpha'],
                          solver=best_hyperparameters['solver'],
                          max_iter=50, random_state=42)

train_losses = []
val_losses = []
for epoch in range(50):
    best_model.partial_fit(X_train_noisy_low, X_train_scaled)

    train_loss = np.mean((best_model.predict(X_train_noisy_low) - X_train_scaled) ** 2)
    val_loss = np.mean((best_model.predict(X_val_noisy_low) - X_val_scaled) ** 2)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

plt.plot(range(50), train_losses, label='Train Loss')
plt.plot(range(50), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

noisy_images = add_noise(X_test_scaled[:4], 0.1)
denoised_images = best_model.predict(noisy_images)

plt.figure(figsize=(12, 8))
for i in range(4):
    # Original
    plt.subplot(4, 3, 3*i + 1)
    plt.imshow(X_test_scaled[i].reshape(8, 8), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Noised
    plt.subplot(4, 3, 3*i + 2)
    plt.imshow(noisy_images[i].reshape(8, 8), cmap='gray')
    plt.title("Noised")
    plt.axis('off')
    
    # Denoised
    plt.subplot(4, 3, 3*i + 3)
    plt.imshow(denoised_images[i].reshape(8, 8), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

plt.tight_layout()
plt.show()
