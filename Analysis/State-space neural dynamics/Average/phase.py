# %%
# Converted from average_trial_phase_space_notebook.ipynb
# Run cells in order in VS Code (Python: Run Cell).

# %% [markdown]
# # Installation

# %%
# Cell 2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import circmean
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import mne

# Number of trials to load/process from the events table
n_needed_trials = 125

#
# Output folder for the figure and animation
output_dir = r"C:\Users\user\OneDrive\Documents\PARA\1. Projects\Clarke Lab Internship\4. assignment-2. Trail Averaging\average"
os.makedirs(output_dir, exist_ok=True)

# Local data folder
data_root = r"C:\Users\user\OneDrive\Documents\PARA\1. Projects\Clarke Lab Internship\4. assignment-2. Trail Averaging\ds005811"

raw_dir = os.path.join(data_root, "derivatives", "preprocessed", "raw")
events_dir = os.path.join(data_root, "derivatives", "detailed_events")

raw_file_path = os.path.join(
    raw_dir, "sub-01_ses-ImageNet01_task-ImageNet_run-01_eeg_clean.fif"
)
events_file_path = os.path.join(events_dir, "sub-01_events.csv")

# %% [markdown]
# ## Access data

# %%
# Cell 4

raw = mne.io.read_raw_fif(raw_file_path, preload=True, verbose=False)
events_df = pd.read_csv(events_file_path).iloc[:n_needed_trials].reset_index(drop=True)

# %% [markdown]
# ## Phase extraction

# %%
# Cell 6
# Filter the continuous EEG, then extract instantaneous phase using the Hilbert transform.
raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=100.0, fir_design="firwin")
raw_phase = raw_filtered.copy().apply_hilbert(envelope=False)
phase_data = np.angle(raw_phase.get_data())

print("Continuous EEG shape:", raw.get_data().shape)
print("Filtered EEG shape:", raw_filtered.get_data().shape)
print("Continuous phase shape:", phase_data.shape)

# %% [markdown]
# ## Epoch the phase data

# %%
# Cell 8
# Extract annotation events
events, event_dict = mne.events_from_annotations(raw)

# Keep only stim_on events
stim_on_id = event_dict["stim_on"]
events = events[events[:, 2] == stim_on_id]

# Epoch the phase signal
phase_info = mne.create_info(
    ch_names=raw.ch_names,
    sfreq=raw.info["sfreq"],
    ch_types=["misc"] * len(raw.ch_names)
)
phase_raw = mne.io.RawArray(phase_data, phase_info, verbose=False)
phase_epochs = mne.Epochs(
    phase_raw,
    events=events,
    event_id=stim_on_id,
    tmin=-0.1,
    tmax=0.8,
    baseline=None,
    reject_by_annotation=False,
    metadata=events_df,
    on_missing="warning",
    preload=True,
    verbose=False
)

# Keep metadata aligned with any epochs omitted by MNE.
epoch_data = phase_epochs.get_data()
epoch_metadata = phase_epochs.metadata.reset_index(drop=True)
onset_idx = int(np.argmin(np.abs(phase_epochs.times - 0)))

# %% [markdown]
# ## Average phase across the first five trials per super class

# %%
# Cell 10
sample_name = ["device", "food", "invertebrate", "bird"]
n_trials_to_average = 5

# Select the first five epochs for each requested super class and compute
# their circular mean across trials. Result: (n_channels, n_times) per class.
average_phase_epochs = np.empty(
    (len(sample_name), epoch_data.shape[1], epoch_data.shape[2]),
    dtype=float)
selected_trial_indices = {}

for i, name in enumerate(sample_name):
    matching_indices = np.flatnonzero(epoch_metadata["super_class"].to_numpy() == name)
    selected_indices = matching_indices[:n_trials_to_average]

    selected_trial_indices[name] = selected_indices
    selected_trials = epoch_data[selected_indices, :, :]
    average_phase_epochs[i, :, :] = circmean(
        selected_trials, low=-np.pi, high=np.pi, axis=0
    )

    print(f"{name}: selected epoch indices {selected_indices.tolist()}")

print("Average phase epochs shape:", average_phase_epochs.shape)
# Shape: (n_sample_classes, n_channels, n_times)

# %% [markdown]
# ## PCA on the continuous stack of trial averages

# %%
# Cell 12
# Concatenate the averaged epochs along time:
# [average 1 times, average 2 times, ...] -> (n_channels, n_classes * n_times).
n_average_epochs, n_channels, n_times = average_phase_epochs.shape
stacked_phase = np.concatenate(
    [average_phase_epochs[i] for i in range(n_average_epochs)],
    axis=1
)
print("Stacked phase shape:", stacked_phase.shape)

# PCA expects observations x features, so transpose to (stacked_times, channels).
pca = PCA(n_components=3)
pca_stacked = pca.fit_transform(stacked_phase.T)  # (n_classes * n_times, 3)

# Reverse the stacking to recover (n_epochs, n_components, n_times).
pca_average_epochs = pca_stacked.reshape(
    n_average_epochs, n_times, 3
).transpose(0, 2, 1)


print("PCA output shape:", pca_stacked.shape)
print("PCA average epochs shape:", pca_average_epochs.shape)

# %% [markdown]
# ## Plot average-trial trajectories

# %%
# Cell 14
fig = plt.figure(figsize=(16, 12))

for i, name in enumerate(sample_name):
    trial_data = pca_average_epochs[i]
    x, y, z = trial_data[0], trial_data[1], trial_data[2]
    n_points = len(x)

    points = np.column_stack((x, y, z)).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.arange(len(segments))

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    lc = Line3DCollection(segments, cmap="Reds", linewidths=3.0, alpha=0.8)
    lc.set_array(colors)
    ax.add_collection3d(lc)

    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_zlim(z.min() - 0.5, z.max() + 0.5)
    ax.set_title(name.capitalize(), fontsize=12)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    cbar = fig.colorbar(lc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("Time Progression")
    ax.scatter(x[0], y[0], z[0], color="grey", s=80, label="Start", depthshade=False)
    ax.scatter(x[onset_idx], y[onset_idx], z[onset_idx],
               color="green", s=80, label="Onset", depthshade=False)
    ax.scatter(x[-1], y[-1], z[-1], color="red", s=80, label="End", depthshade=False)
    ax.legend(loc="upper left")

# No figure subtitle/suptitle.
plt.tight_layout()
plot_path = os.path.join(output_dir, "average_trial_phase_trajectories.png")
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved plot to:", plot_path)

# %% [markdown]
# ## Animate average-trial trajectories

# %%
# Cell 16
fps = 15
n_frames = n_times

fig = plt.figure(figsize=(16, 12))
animation_axes = []

for i, name in enumerate(sample_name):
    trial_data = pca_average_epochs[i]
    x, y, z = trial_data[0], trial_data[1], trial_data[2]
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")

    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_zlim(z.min() - 0.5, z.max() + 0.5)
    ax.set_title(name.capitalize(), fontsize=12)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    points = np.column_stack((x, y, z)).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.arange(len(segments))

    lc = Line3DCollection(
        segments[:1], cmap="Reds",
        norm=plt.Normalize(0, max(1, len(segments) - 1)),
        linewidths=3.0, alpha=0.8
    )
    lc.set_array(colors[:1])
    ax.add_collection3d(lc)

    start_dot = ax.scatter(x[0], y[0], z[0], color="grey", s=80,
                           label="Start", depthshade=False)
    onset_dot = ax.scatter(x[onset_idx], y[onset_idx], z[onset_idx],
                           color="green", s=80, label="Onset", depthshade=False)
    end_dot = ax.scatter(x[0], y[0], z[0], color="red", s=80,
                         label="End", depthshade=False)

    cbar = fig.colorbar(lc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("Time Progression")
    ax.legend(loc="upper left")

    animation_axes.append({
        "line": lc, "end": end_dot, "segments": segments,
        "colors": colors, "x": x, "y": y, "z": z
    })

# No figure subtitle/suptitle.
plt.tight_layout()

def update(frame):
    artists = []
    for item in animation_axes:
        n_segments = min(frame + 1, len(item["segments"]))
        item["line"].set_segments(item["segments"][:n_segments])
        item["line"].set_array(item["colors"][:n_segments])

        idx = min(frame + 1, len(item["x"]) - 1)
        item["end"]._offsets3d = (
            [item["x"][idx]], [item["y"][idx]], [item["z"][idx]]
        )
        artists.extend([item["line"], item["end"]])
    return artists

animation = FuncAnimation(
    fig, update, frames=n_frames - 1,
    interval=1000 / fps, blit=False
)

gif_path = os.path.join(output_dir, "average_trial_phase_trajectories.gif")
animation.save(gif_path, writer=PillowWriter(fps=fps))
plt.close(fig)
print("Saved animation to:", gif_path)