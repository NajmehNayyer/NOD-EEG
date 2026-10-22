# -*- coding: utf-8 -*-
"""Single-trial EEG phase-space trajectory animation."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import random


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import mne

n_needed_trials = 125

"""Access data: download the public OpenNeuro files if they are not already present."""

from pathlib import Path
import subprocess
import shutil

project_dir = Path(__file__).resolve().parent
raw_dir = project_dir / "ds005811" / "derivatives" / "preprocessed" / "raw"
events_dir = project_dir / "ds005811" / "derivatives" / "detailed_events"
raw_dir.mkdir(parents=True, exist_ok=True)
events_dir.mkdir(parents=True, exist_ok=True)

s3_raw_file_path = "s3://openneuro.org/ds005811/derivatives/preprocessed/raw/sub-01_ses-ImageNet01_task-ImageNet_run-01_eeg_clean.fif"
local_raw_file_path = raw_dir / "sub-01_ses-ImageNet01_task-ImageNet_run-01_eeg_clean.fif"

s3_events_file_path = "s3://openneuro.org/ds005811/derivatives/detailed_events/sub-01_events.csv"
local_events_file_path = events_dir / "sub-01_events.csv"

if shutil.which("aws") is None:
    raise RuntimeError(
        "AWS CLI was not found. Install AWS CLI v2, restart the VS Code terminal, "
        "and run this script again."
    )

if not local_raw_file_path.exists():
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", s3_raw_file_path, str(local_raw_file_path)],
        check=True,
    )

if not local_events_file_path.exists():
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", s3_events_file_path, str(local_events_file_path)],
        check=True,
    )

# 1. Load the raw EEG data
eeg_file_path = local_raw_file_path
raw = mne.io.read_raw_fif(eeg_file_path, preload=True, verbose=False)
print("--- EEG Raw Data Info ---")
print(raw)

# 2. Load the detailed events CSV
events_file_path = local_events_file_path
events_df = pd.read_csv(events_file_path)

events_df = events_df[:n_needed_trials]

# print("="*15, "Class Value Counts", "="*15)
# print(events_df["class"].value_counts(), "\n")

print("="*15, "Super Class Value Counts", "="*15)
print(events_df["super_class"].value_counts(), "\n")

print("="*15, "number of trials: ", len(events_df), "="*15, "\n")

print(events_df.head())

# Check if the raw data is an instance of mne.io.Raw (continuous) or mne.Epochs
is_epoched = isinstance(raw, mne.BaseEpochs)
print(f"Is the raw object epoched? {is_epoched}")

# Get the time window of this raw file
t_min = raw.times[0]
t_max = raw.times[-1]
duration_seconds = raw.times[-1] - raw.times[0]
print(f"\nTime window: {t_min:.3f}s to {t_max:.3f}s (Total duration: {duration_seconds:.2f} seconds)")

"""## Analysis

## Phase
"""

# Filter raw data between 0.5 and 100 Hz
raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=100.0, fir_design='firwin')

# Apply Hilbert transform and extract the phase
raw_phase = raw_filtered.copy().apply_hilbert(envelope=False)
phase_data = np.angle(raw_phase.get_data())

print("Continues: ", raw.get_data().shape)
print("Filtered: ", raw_filtered.get_data().shape)
print("Phase: ", phase_data.shape)

"""Dimension reduction"""

# Transpose continuous phase data
phase_data_transposed = phase_data.T  # Shape: (times, channels)

# 2. Fit and transform using PCA
pca = PCA(n_components=3)
pca_transformed = pca.fit_transform(phase_data.T)

pca.explained_variance_

"""Epoching"""

# Find events from raw annotations
events, event_dict = mne.events_from_annotations(raw)
print("Original event dictionary found from annotations:", event_dict)

# Keep only the target number of events (first 125 triggers)
if len(events) > n_needed_trials:
    events = events[:n_needed_trials]

# Dynamically filter event_dict to keep only event types present in the sliced events
unique_event_ids_in_subset = set(events[:, 2])
filtered_event_dict = {key: val for key, val in event_dict.items() if val in unique_event_ids_in_subset}
print("Filtered event dictionary for subset:", filtered_event_dict)

sfreq = raw.info['sfreq']
ch_names = ['PC1', 'PC2', 'PC3']
ch_types = ['misc'] * 3

pca_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
pca_raw = mne.io.RawArray(pca_transformed.T, pca_info)

epochs = mne.Epochs(
    pca_raw,
    events=events,
    event_id=filtered_event_dict,
    tmin=-0.1,
    tmax=0.8,
    baseline=None,
    reject_by_annotation=True,
    metadata = events_df,
    on_missing='warning',
    preload=True,
    verbose=True)

events_df = epochs.metadata
print(f"\nNumber of epochs: {len(events_df)}")
print(f"Epochs data shape: {epochs.get_data().shape}")

# np.save('pca_epochs.npy', epochs.get_data())
# events_df.to_csv('events_df.csv', index=False)


sample_name = ['device', 'food', 'invertebrate', 'bird']

# sample_index = []

# for cat in sample_name:

#     match = events_df[events_df['super_class'] == cat]
#     trial_idx = match.index[random.randint(0, len(match)-1)]
#     sample_index.append(trial_idx)

#     print(f"{cat}: {trial_idx}")

sample_index = [53, 91, 105, 63]

# # Retrieve epoch data
epoch_data = epochs.get_data()
onset_idx = np.argmin(np.abs(epochs.times - 0))

fig = plt.figure(figsize=(16, 12))

for i, (name, trial_idx) in enumerate(zip(sample_name, sample_index)):
    # Extract data for current trial
    trial_data = epoch_data[trial_idx, :, :]

    x = trial_data[0, :]
    y = trial_data[1, :]
    z = trial_data[2, :]
    n_times = len(x)

    # Prepare segments for Line3DCollection
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color values representing time progression
    colors = np.arange(n_times)

    # Setup subplot
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')

    # Add 3D Line collection using Reds colormap
    lc = Line3DCollection(segments, cmap='Reds', linewidths=3.0, alpha=0.8)
    lc.set_array(colors)
    ax.add_collection3d(lc)

    # Set axes limits with a small padding
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_zlim(z.min() - 0.5, z.max() + 0.5)

    ax.set_title(f"{name.capitalize()} (Trial {trial_idx})", fontsize=12)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    # Colorbar showing time progression
    cbar = fig.colorbar(lc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Time Progression')

    # Add a start circle
    ax.scatter(x[0], y[0], z[0], color='grey', s=100, alpha=1.0, label='Start', depthshade=False)

    # Add a stimulus onset circle
    ax.scatter(x[onset_idx], y[onset_idx], z[onset_idx],
        color="green", s=100, marker="o", label="Onset", depthshade=False)

    # Add a end circle
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, alpha=1.0, label='End', depthshade=False)

    ax.legend(loc='upper left')

plt.suptitle("3D Trajectories of Epoched PCA Components", fontsize=16, y=0.98)
plt.tight_layout()
plt.show()

# Download the figure
output_filename = 'single_trial_phase_trajectories.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Animation settings
output_filename = "single_trial_phase_trajectories.gif"
fps = 15

n_frames = epoch_data.shape[2]

if any(index < 0 or index >= len(epoch_data) for index in sample_index):
    raise IndexError(
        f"At least one sample_index is outside the available epoch range "
        f"(0 to {len(epoch_data) - 1}). Check how many epochs were retained "
        f"and update sample_index."
    )

fig = plt.figure(figsize=(16, 12))
axes = []

for i, (name, trial_idx) in enumerate(zip(sample_name, sample_index)):
    trial_data = epoch_data[trial_idx]
    x, y, z = trial_data[0], trial_data[1], trial_data[2]
    n_times = len(x)

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")

    # Fix axis limits throughout the animation
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_zlim(z.min() - 0.5, z.max() + 0.5)

    ax.set_title(f"{name.capitalize()} (Trial {trial_idx})", fontsize=12)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    # Full trajectory segments, colored by temporal progression
    points = np.column_stack((x, y, z)).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.arange(n_times - 1)

    # Initially show only the first segment
    lc = Line3DCollection(
        segments[:1],
        cmap="Reds",
        norm=plt.Normalize(0, n_times - 1),
        linewidths=3.0,
        alpha=0.8
    )
    lc.set_array(colors[:1])
    ax.add_collection3d(lc)

    # Start and end points
    start_dot = ax.scatter(
        x[0], y[0], z[0],
        color="grey", s=100, alpha=1.0,
        label="Start", depthshade=False
    )
    onset_dot = ax.scatter(
        x[onset_idx], y[onset_idx], z[onset_idx],
        color="green", s=100, alpha=1.0,
        label="Onset", depthshade=False
    )
    end_dot = ax.scatter(
        x[0], y[0], z[0],
        color="red", s=100, alpha=1.0,
        label="End", depthshade=False
    )

    # Colorbar showing temporal progression
    cbar = fig.colorbar(lc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("Time Progression")

    ax.legend(loc="upper left")

    axes.append({
        "line": lc,
        "start": start_dot,
        "onset": onset_dot,
        "end": end_dot,
        "segments": segments,
        "colors": colors,
        "x": x,
        "y": y,
        "z": z,
        "n_times": n_times
    })

fig.suptitle("3D Trajectories of Epoched PCA Components", fontsize=16, y=0.98)
plt.tight_layout()

def update(frame):
    for item in axes:
        # Reveal the trajectory progressively
        n_segments = min(frame + 1, item["n_times"] - 1)

        item["line"].set_segments(item["segments"][:n_segments])
        item["line"].set_array(item["colors"][:n_segments])

        # Move the red endpoint along the trajectory
        idx = min(frame, item["n_times"] - 1)
        item["end"]._offsets3d = (
            [item["x"][idx]],
            [item["y"][idx]],
            [item["z"][idx]]
        )

    return [item["line"] for item in axes]

animation = FuncAnimation(
    fig,
    update,
    frames=n_frames - 1,
    interval=1000 / fps,
    blit=False
)

animation.save(
    output_filename,
    writer=PillowWriter(fps=fps)
)

plt.close(fig)
print(f"GIF saved to: {output_filename}")
