import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import collections

# --- CONFIGURATION ---
PORT = 'COM7'
BAUD = 115200
MAX_POINTS = 15  # Number of points to keep in the trail

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
    print(f"Successfully opened {PORT}!")
except Exception as e:
    print(f"Error opening {PORT}. Please ensure you closed the ESP-IDF monitor!")
    print(e)
    exit(1)

# --- MATPLOTLIB STYLING (DARK MODE) ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8), facecolor='#0D1117')
ax = fig.add_subplot(111, polar=True, facecolor='#0D1117')

# Configure the polar coordinate system
ax.set_theta_zero_location('N')  # 0 degrees is straight ahead (up/North)
ax.set_theta_direction(-1)       # Positive angles go clockwise to the right
ax.set_thetamin(-60)             # Clamp left
ax.set_thetamax(60)              # Clamp right
ax.set_ylim(0, 3)                # Maximum radius (3 meters). Change if needed.

# Grid lines and spines
ax.grid(color='#30363D', linestyle='--', linewidth=1)
ax.spines['polar'].set_color('#30363D')
ax.tick_params(colors='#8B949E')
ax.set_rticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # Metre markers
ax.set_rlabel_position(0)  # Move radius labels out of the way

plt.title("HLK-LD2450 Real-Time Polar Tracking", color='#58A6FF', size=18, weight='bold', pad=30)

# Create the scatter plot for the target and a fixed marker for the radar origin
scatter_plot = ax.scatter([], [], c='#58A6FF', s=150, alpha=0.9, edgecolors='white', zorder=2)
ax.scatter([0], [0], c='#238636', s=100, zorder=3, marker='P') # The Radar Sensor

# Create buffers for trailing effect
r_data = collections.deque(maxlen=MAX_POINTS)
theta_data = collections.deque(maxlen=MAX_POINTS)
colors_data = np.zeros((MAX_POINTS, 4))

# Precompute decreasing alpha colors for the trail (#58A6FF)
for i in range(MAX_POINTS):
    alphas = (i + 1) / MAX_POINTS  # Scales from nearly transparent to 1.0 solid
    colors_data[i] = [0.345, 0.651, 1.0, alphas]

def update(frame):
    # Read all pending serial lines from the buffer
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith('RADAR'):
                parts = line.split(',')
                if len(parts) == 4:
                    # Valid Target Detected
                    if parts[1] != 'NAN' and parts[2] != 'NAN':
                        r = float(parts[1])
                        theta_deg = float(parts[2])
                        
                        r_data.append(r)
                        theta_data.append(np.radians(theta_deg))
                    # Empty Target Detected
                    else:
                        r_data.clear()
                        theta_data.clear()
        except:
            pass

    if len(r_data) > 0:
        # Dynamic resizing of arrays for the trail effect
        curr_len = len(r_data)
        scatter_plot.set_offsets(np.c_[list(theta_data), list(r_data)])
        
        # Scale sizes (older trailing points are smaller)
        sizes = np.linspace(20, 200, curr_len)
        scatter_plot.set_sizes(sizes)
        
        # Colors fade out gradually
        scatter_plot.set_facecolors(colors_data[-curr_len:])
    else:
        # Empty plot if NAN
        scatter_plot.set_offsets(np.empty((0, 2)))
        
    return scatter_plot,

# Setup animation at ~30 FPS
ani = animation.FuncAnimation(fig, update, interval=30, blit=False, save_count=50)

print("Starting live plot... Close window to exit or press Ctrl+C.")
try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
