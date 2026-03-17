import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import collections

# --- CONFIGURATION ---
PORT = 'COM7'
BAUD = 115200
MAX_POINTS = 100  # Number of data points to keep in the rolling window

# Filter parameters
HAMPEL_WINDOW = 5
HAMPEL_SIGMA = 2.0
KALMAN_Q = 1e-4  # Process noise
KALMAN_R = 0.05   # Measurement noise

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
    print(f"Connected to {PORT}")
except Exception as e:
    print(f"Error opening {PORT}. Close ESP-IDF monitor first!")
    print(e)
    exit(1)

# --- DATA BUFFERS ---
timestamps = collections.deque(maxlen=MAX_POINTS)
distances_raw = collections.deque(maxlen=MAX_POINTS)
distances_smoothed = collections.deque(maxlen=MAX_POINTS)
raw_frames = collections.deque(maxlen=16)
heartbeat_count = [0]

# --- Kalman State ---
k_est = 0.0
k_err = 1.0

def kalman_update(measurement):
    global k_est, k_err
    if k_est == 0.0:
        k_est = measurement
        return measurement
    
    # Prediction
    k_est_pred = k_est
    k_err_pred = k_err + KALMAN_Q
    
    # Update
    gain = k_err_pred / (k_err_pred + KALMAN_R)
    k_est = k_est_pred + gain * (measurement - k_est_pred)
    k_err = (1 - gain) * k_err_pred
    
    return k_est

# --- MATPLOTLIB DARK STYLING ---
plt.style.use('dark_background')
fig, (ax_main, ax_bar) = plt.subplots(1, 2, figsize=(14, 6), 
                                       facecolor='#0D1117',
                                       gridspec_kw={'width_ratios': [3, 1]})
fig.suptitle("Wi-Fi FTM Smoothed Range Monitor", color='#58A6FF', fontsize=18, fontweight='bold')

# --- LEFT: Rolling Distance Line Chart ---
ax_main.set_facecolor('#0D1117')
ax_main.set_xlabel('Time (s)', color='#8B949E', fontsize=11)
ax_main.set_ylabel('Distance (m)', color='#8B949E', fontsize=11)
ax_main.set_title('Raw vs Smoothed Range Over Time', color='#C9D1D9', fontsize=13)
ax_main.grid(color='#21262D', linestyle='--', linewidth=0.8)
ax_main.tick_params(colors='#8B949E')
ax_main.set_ylim(-0.2, 5.0)
ax_main.spines['bottom'].set_color('#30363D')
ax_main.spines['left'].set_color('#30363D')
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)

line_raw, = ax_main.plot([], [], color='#8B949E', linewidth=1, alpha=0.5, linestyle='--', label='Raw (Multi-path)')
line_smooth, = ax_main.plot([], [], color='#58A6FF', linewidth=3, alpha=0.9, label='Kalman + Hampel')
ax_main.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#C9D1D9')

scatter = ax_main.scatter([], [], c='#39D353', s=40, zorder=3, alpha=0.9)

# Current distance text
dist_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes,
                         fontsize=22, color='#39D353', fontweight='bold',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#161B22', 
                                   edgecolor='#30363D', alpha=0.9))

hb_text = ax_main.text(0.98, 0.95, '', transform=ax_main.transAxes,
                       fontsize=11, color='#8B949E', fontweight='normal',
                       verticalalignment='top', horizontalalignment='right')

# --- RIGHT: Per-Frame Bar Chart ---
ax_bar.set_facecolor('#0D1117')
ax_bar.set_title('Last Burst Frame Ranges', color='#C9D1D9', fontsize=12)
ax_bar.set_xlabel('Frame #', color='#8B949E', fontsize=10)
ax_bar.set_ylabel('Range (m)', color='#8B949E', fontsize=10)
ax_bar.grid(color='#21262D', linestyle='--', linewidth=0.5, axis='y')
ax_bar.tick_params(colors='#8B949E')
ax_bar.set_ylim(0, 5.0)
ax_bar.spines['bottom'].set_color('#30363D')
ax_bar.spines['left'].set_color('#30363D')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

def apply_hampel(new_val):
    if len(distances_smoothed) < HAMPEL_WINDOW:
        return new_val
    window = list(distances_raw)[-HAMPEL_WINDOW:]
    window.append(new_val)
    med = np.median(window)
    mad = np.median(np.abs(window - med))
    
    # If outlier, replace with median
    if mad > 0 and np.abs(new_val - med) > HAMPEL_SIGMA * mad:
        return med
    return new_val

def update(frame):
    global k_est
    while ser.in_waiting > 0:
        try:
            line_data = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line_data.startswith('WIFI_RAW,'):
                parts = line_data.split(',')
                raw_frames.clear()
                for p in parts[1:-1]:
                    try:
                        raw_frames.append(float(p))
                    except ValueError:
                        pass

            elif line_data.startswith('WIFI,'):
                parts = line_data.split(',')
                if len(parts) == 3:
                    try:
                        r = float(parts[1])
                        ts = float(parts[2]) / 1000.0  
                        
                        distances_raw.append(r)
                        timestamps.append(ts)
                        
                        # 1. Hampel Outlier Rejection
                        r_hampel = apply_hampel(r)
                        
                        # 2. Kalman 1D Filter
                        r_smooth = kalman_update(r_hampel)
                        distances_smoothed.append(r_smooth)
                        
                    except ValueError:
                        pass

            elif line_data.startswith('TAG_ALIVE,'):
                heartbeat_count[0] += 1
        except:
            pass

    # --- UPDATE LINE CHART ---
    if len(timestamps) > 1:
        ts_arr = np.array(timestamps)
        d_raw_arr = np.array(distances_raw)
        d_smooth_arr = np.array(distances_smoothed)
        
        ts_rel = ts_arr - ts_arr[0]

        line_raw.set_data(ts_rel, d_raw_arr)
        line_smooth.set_data(ts_rel, d_smooth_arr)
        
        # Scatter is at the leading edge of the smooth line
        scatter.set_offsets(np.c_[ts_rel[-1:], d_smooth_arr[-1:]])

        ax_main.set_xlim(max(0, ts_rel[-1] - 30), ts_rel[-1] + 1)
        
        if len(d_smooth_arr) > 0:
            y_max = max(d_smooth_arr.max() * 1.5, 1.0)
            ax_main.set_ylim(-0.1, y_max)

        current = d_smooth_arr[-1]
        color = '#39D353' if current < 2.0 else ('#D29922' if current < 4.0 else '#F85149')
        dist_text.set_text(f'{current:.3f} m (Smooth)')
        dist_text.set_color(color)
        scatter.set_color(color)

    hb_text.set_text(f'♥ HB: {heartbeat_count[0]}')

    # --- UPDATE BAR CHART ---
    ax_bar.cla()
    ax_bar.set_facecolor('#0D1117')
    ax_bar.set_title('Last Burst Frame Ranges', color='#C9D1D9', fontsize=12)
    ax_bar.set_xlabel('Frame #', color='#8B949E', fontsize=10)
    ax_bar.grid(color='#21262D', linestyle='--', linewidth=0.5, axis='y')
    ax_bar.spines['bottom'].set_color('#30363D')
    ax_bar.spines['left'].set_color('#30363D')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.tick_params(colors='#8B949E')

    if len(raw_frames) > 0:
        frames_arr = np.array(raw_frames)
        x_pos = np.arange(len(frames_arr))
        colors = ['#F85149' if v == frames_arr.min() and v > 0 else '#58A6FF' for v in frames_arr]
        ax_bar.bar(x_pos, frames_arr, color=colors, alpha=0.85, edgecolor='#30363D', linewidth=0.5)
        if frames_arr.max() > 0:
            ax_bar.set_ylim(0, max(frames_arr.max() * 1.3, 0.5))

    return line_raw, line_smooth, scatter

ani = animation.FuncAnimation(fig, update, interval=50, blit=False, save_count=50)

print("Starting FTM Smoothed visualizer... Close the window to exit.")
try:
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
