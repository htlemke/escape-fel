%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import threading
import time
from ipywidgets import interact, IntSlider

# Turn on interactive mode to handle manual display updating
plt.ion()

# -------------------------------------------------------------
# 1. Create Mock Data: A list of heavy 3D Dask Arrays
# -------------------------------------------------------------
# Let's say we have 5 steps in our list. Each step is a heavy 3D stack (500 frames, 100x100)
num_steps = 5
preview_frames = 50  # Fast preview chunk size

list_of_3d_arrays = [
    da.random.random((500, 100, 100), chunks=(50, 100, 100)) + (i * 0.1)
    for i in range(num_steps)
]

# -------------------------------------------------------------
# 2. Setup Matplotlib Figure
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
dummy_data = np.zeros((100, 100))
im = ax.imshow(dummy_data, cmap='viridis', origin='lower', vmin=0, vmax=1.5)
fig.colorbar(im, ax=ax)
plt.close(fig)

# Global variables to track background execution state
current_timer = None
bg_thread = None
cancel_event = threading.Event()

# -------------------------------------------------------------
# 3. Background Processing Logic
# -------------------------------------------------------------
def compute_full_average(step_idx):
    """Computes the full heavy stack average unless interrupted."""
    global cancel_event
    
    # Check if the user already moved the slider before we started
    if cancel_event.is_set():
        return
        
    # 1. Compute the full average across axis 0 (the stack dimension)
    full_lazy_avg = list_of_3d_arrays[step_idx].mean(axis=0)
    full_computed = full_lazy_avg.compute()
    
    # 2. Final safety check: if user moved slider during computation, discard results
    if not cancel_event.is_set():
        im.set_data(full_computed)
        ax.set_title(f"Step {step_idx} - Full Average (Complete)")
        fig.canvas.draw_idle()
        display(fig)

def trigger_refinement(step_idx):
    """Spins up the thread to calculate the high-res background average."""
    global bg_thread, cancel_event
    cancel_event.clear()
    
    bg_thread = threading.Thread(target=compute_full_average, args=(step_idx,))
    bg_thread.start()

# -------------------------------------------------------------
# 4. Primary Slider Callback
# -------------------------------------------------------------
def on_slider_change(step_idx):
    global current_timer, cancel_event
    
    # STEP A: Stop any full-scale background computations currently running
    cancel_event.set()
    if current_timer is not None:
        current_timer.cancel()
        
    # STEP B: Instantly calculate and show the low-cost preview
    # We grab just the first 'preview_frames' and average them along axis 0
    preview_lazy_avg = list_of_3d_arrays[step_idx][:preview_frames, :, :].mean(axis=0)
    preview_computed = preview_lazy_avg.compute()
    
    # Update UI with the quick preview
    im.set_data(preview_computed)
    ax.set_title(f"Step {step_idx} - Fast Preview ({preview_frames} frames)...")
    fig.canvas.draw_idle()
    display(fig)
    
    # STEP C: Set a timer. If the user stays on this step for 0.5 seconds,
    # kick off the deep background calculation for the full average.
    current_timer = threading.Timer(0.5, trigger_refinement, args=(step_idx,))
    current_timer.start()

# Render the interactive UI
interact(on_slider_change, step_idx=IntSlider(min=0, max=num_steps-1, step=1, value=0));