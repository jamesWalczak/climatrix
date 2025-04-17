import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ============================
# Simulation Parameters
# ============================
n_trials = 1000  # Number of training trials
T = 100  # Trial duration in ms
dt = 1.0  # Time step in ms
time_steps = int(T / dt)

# Encoding layer parameters
n_enc = 10  # Number of encoding neurons
rate_max = 50  # Maximum firing rate (Hz)
enc_centers = np.linspace(0, 1, n_enc)  # Preferred x-values
sigma = 0.1  # Tuning width

# Output layer parameters: 3 spiking neurons (Izhikevich dynamics)
n_out = 3
a = 0.02
b = 0.2
c = -65
d = 8

# Bias current (tuned lower to prevent saturation)
I_bias = 5.0

# ============================
# Weight Initialization
# ============================
# Initialize with a larger range, but not too high
W = np.random.uniform(-1.0, 1.0, (n_enc, n_out))

# Decoding coefficients for reading out the spike counts
decoding_coefs = np.linspace(0.5, 1.5, n_out)

# Learning rate for reward-modulated STDP
eta = 0.005  # slightly increased for more robust updates

# Decay factor for eligibility trace (to favor recent spike coincidences)
elig_decay = 0.99

# Containers for training data
loss_history = []
weights_history = []

# ============================
# Training Loop
# ============================
for trial in range(n_trials):
    x = np.random.uniform(0, 1)
    y_true = x**2

    # Population encoding: Compute firing rates based on Gaussian tuning
    rates = rate_max * np.exp(-((x - enc_centers) ** 2) / (2 * sigma**2))
    spikes_enc = (
        np.random.rand(n_enc, time_steps) < (rates[:, None] * dt / 1000)
    ).astype(float)

    # Initialize output layer neuron states
    v = np.full(n_out, c, dtype=float)
    u = b * v
    spikes_out = np.zeros((n_out, time_steps))

    # Eligibility trace initialization
    elig = np.zeros((n_enc, n_out))

    for t in range(time_steps):
        # Calculate the total synaptic input from encoding spikes
        I_syn = np.sum(W * spikes_enc[:, t][:, None], axis=0)
        I = I_syn + I_bias
        # Izhikevich neuron update
        v = v + dt * (0.04 * v**2 + 5 * v + 140 - u + I)
        u = u + dt * (a * (b * v - u))

        # Identify which neurons fire
        fired = v >= 30
        spikes_out[fired, t] = 1

        # Reset for neurons that fired
        for j in range(n_out):
            if fired[j]:
                v[j] = c
                u[j] += d

        # Update eligibility traces, then decay them slightly
        elig = elig_decay * elig + np.outer(
            spikes_enc[:, t], fired.astype(float)
        )

    # Decode output: weighted sum of spike counts (normalized)
    spike_counts = spikes_out.sum(axis=1)
    y_pred = np.dot(decoding_coefs, spike_counts) / time_steps

    loss = (y_pred - y_true) ** 2
    loss_history.append(loss)
    weights_history.append(W.copy())

    # Reward-modulated STDP update using the reward signal (negative loss)
    reward = -loss
    W += eta * reward * elig

# ============================
# Testing the Learned Network
# ============================
test_x = np.linspace(0, 1, 100)
test_pred = []

for x in test_x:
    rates = rate_max * np.exp(-((x - enc_centers) ** 2) / (2 * sigma**2))
    spikes_enc = (
        np.random.rand(n_enc, time_steps) < (rates[:, None] * dt / 1000)
    ).astype(float)

    v = np.full(n_out, c, dtype=float)
    u = b * v
    spikes_out = np.zeros((n_out, time_steps))

    for t in range(time_steps):
        I_syn = np.sum(W * spikes_enc[:, t][:, None], axis=0)
        I = I_syn + I_bias
        v = v + dt * (0.04 * v**2 + 5 * v + 140 - u + I)
        u = u + dt * (a * (b * v - u))
        fired = v >= 30
        spikes_out[fired, t] = 1
        for j in range(n_out):
            if fired[j]:
                v[j] = c
                u[j] += d

    spike_counts = spikes_out.sum(axis=1)
    pred = np.dot(decoding_coefs, spike_counts) / time_steps
    test_pred.append(pred)

test_pred = np.array(test_pred)

# ============================
# Visualization
# ============================
plt.figure(figsize=(8, 5))
sns.lineplot(x=np.arange(n_trials), y=loss_history, color="navy")
plt.xlabel("Training Trial")
plt.ylabel("Squared Error Loss")
plt.title("Loss over Training Trials")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(test_x, test_x**2, label="True $x^2$", color="green", alpha=0.7)
plt.plot(test_x, test_pred, label="Predicted", color="red", linewidth=2)
plt.xlabel("Input $x$")
plt.ylabel("Output")
plt.title("Regression Result: $x^2$ Function")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

weights_history = np.array(weights_history)
plt.figure(figsize=(8, 5))
for i in range(n_enc):
    plt.plot(weights_history[:, i, 0], label=f"Enc neuron {i} -> Out neuron 0")
plt.xlabel("Training Trial")
plt.ylabel("Weight Value")
plt.title("Synaptic Weight Evolution (Encoding -> Output Neuron 0)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()
