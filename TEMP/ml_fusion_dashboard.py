"""
═══════════════════════════════════════════════════════════════════════════
  ISAC ML Fusion Dashboard — isac_polar_ml Visualizer
  
  Reads FUSED CSV lines from COM13:
      FUSED,<r_m>,<theta_deg>,<timestamp_ms>,<inference_us>
  
  4-panel layout:
    [TOP-LEFT]    Polar position with trailing ghost
    [TOP-RIGHT]   Dual-axis range + angle timeline
    [BOTTOM-LEFT] Inference latency histogram + rolling line
    [BOTTOM-RIGHT] Live stats panel
═══════════════════════════════════════════════════════════════════════════
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import collections
import time
import sys

# ──────────────────────── CONFIGURATION ──────────────────────────
PORT = "COM13"
BAUD = 115200
MAX_TRAIL     = 25       # polar ghost trail
MAX_TIMELINE  = 300      # rolling timeline points
MAX_LATENCY   = 200      # latency history
FPS           = 25

# ──────────────────────── SERIAL ─────────────────────────────────
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.005)
    print(f"Connected to {PORT}")
except Exception as e:
    print(f"Cannot open {PORT} -- close ESP-IDF monitor first!")
    print(e)
    sys.exit(1)

# ──────────────────────── COLOUR PALETTE ─────────────────────────
BG          = "#06090F"
PANEL       = "#0D1321"
GRID        = "#1B2838"
BORDER      = "#2A3A50"
TEXT_DIM    = "#5A6B80"
TEXT_MED    = "#8899AA"
TEXT_HI     = "#D0DEF0"

FUSED_MAGENTA   = "#E040FB"  # primary fused colour
FUSED_GLOW      = "#CE93D8"
RANGE_CYAN      = "#00E5FF"
ANGLE_AMBER     = "#FFD740"
LATENCY_GREEN   = "#69F0AE"
ACCENT_RED      = "#FF5252"
ACCENT_PURPLE   = "#B388FF"

# ──────────────────────── DATA BUFFERS ───────────────────────────
trail_r     = collections.deque(maxlen=MAX_TRAIL)
trail_theta = collections.deque(maxlen=MAX_TRAIL)

ts_buf      = collections.deque(maxlen=MAX_TIMELINE)
range_buf   = collections.deque(maxlen=MAX_TIMELINE)
angle_buf   = collections.deque(maxlen=MAX_TIMELINE)

lat_buf     = collections.deque(maxlen=MAX_LATENCY)
lat_ts_buf  = collections.deque(maxlen=MAX_LATENCY)
lat_hist    = []

stats = {
    "count": 0,
    "start": time.time(),
    "min_lat": 9999,
    "max_lat": 0,
    "sum_lat": 0,
    "last_r": 0.0,
    "last_theta": 0.0,
    "last_lat": 0,
}

# ──────────────────────── FIGURE SETUP ───────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(17, 10), facecolor=BG, dpi=96)
fig.canvas.manager.set_window_title("ISAC ML Fusion Dashboard")

gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1],
                       width_ratios=[1, 1.5],
                       hspace=0.30, wspace=0.25,
                       left=0.05, right=0.97, top=0.93, bottom=0.05)

# ── [TOP-LEFT] Polar ─────────────────────────────────────────────
ax_polar = fig.add_subplot(gs[0, 0], polar=True, facecolor=PANEL)
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)
ax_polar.set_thetamin(-60)
ax_polar.set_thetamax(60)
ax_polar.set_ylim(0, 5)
ax_polar.set_rticks([1, 2, 3, 4, 5])
ax_polar.set_rlabel_position(25)
ax_polar.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.5)
ax_polar.tick_params(colors=TEXT_DIM, labelsize=7)
for sp in ax_polar.spines.values():
    sp.set_color(BORDER)
ax_polar.set_title("ML Fused Position", color=FUSED_MAGENTA, fontsize=12,
                    fontweight="bold", pad=16, fontfamily="monospace")

ax_polar.scatter([0], [0], c=LATENCY_GREEN, s=80, zorder=5, marker="P",
                  edgecolors="white", linewidths=0.5)

_tc = np.zeros((MAX_TRAIL, 4))
for i in range(MAX_TRAIL):
    a = ((i + 1) / MAX_TRAIL) ** 1.8
    _tc[i] = [0.878, 0.251, 0.984, a]  # FUSED_MAGENTA

sc_trail = ax_polar.scatter([], [], s=40, zorder=3)
sc_head  = ax_polar.scatter([], [], c=FUSED_MAGENTA, s=250, zorder=4,
                             edgecolors="white", linewidths=1.2)
polar_lbl = ax_polar.text(np.radians(52), 4.5, "", fontsize=11,
                           color=FUSED_MAGENTA, fontweight="bold",
                           ha="center", fontfamily="monospace")

# ── [TOP-RIGHT] Range + Angle Timeline ──────────────────────────
ax_time = fig.add_subplot(gs[0, 1], facecolor=PANEL)
ax_time.set_title("Range & Angle Timeline", color=TEXT_HI, fontsize=12,
                   fontweight="bold", fontfamily="monospace")
ax_time.set_xlabel("Time (s)", color=TEXT_DIM, fontsize=8)
ax_time.set_ylabel("Range (m)", color=RANGE_CYAN, fontsize=9)
ax_time.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.4)
ax_time.tick_params(axis="y", colors=RANGE_CYAN, labelsize=8)
ax_time.tick_params(axis="x", colors=TEXT_DIM, labelsize=7)
for s in ("top", "right"):
    ax_time.spines[s].set_visible(False)
ax_time.spines["bottom"].set_color(BORDER)
ax_time.spines["left"].set_color(BORDER)

ax_ang = ax_time.twinx()
ax_ang.set_ylabel("Angle (deg)", color=ANGLE_AMBER, fontsize=9)
ax_ang.tick_params(axis="y", colors=ANGLE_AMBER, labelsize=8)
ax_ang.spines["right"].set_color(BORDER)
for s in ("top", "left", "bottom"):
    ax_ang.spines[s].set_visible(False)

ln_r, = ax_time.plot([], [], color=RANGE_CYAN, linewidth=2.2, alpha=0.9,
                      label="Range (m)")
ln_a, = ax_ang.plot([], [], color=ANGLE_AMBER, linewidth=1.8, alpha=0.75,
                     linestyle="--", label="Angle (deg)")

dot_r = ax_time.scatter([], [], c=RANGE_CYAN, s=50, zorder=4,
                          edgecolors="white", linewidths=0.5)
dot_a = ax_ang.scatter([], [], c=ANGLE_AMBER, s=50, zorder=4,
                         edgecolors="white", linewidths=0.5)

lines_legend = [ln_r, ln_a]
ax_time.legend(lines_legend, [l.get_label() for l in lines_legend],
               loc="upper left", fontsize=7, facecolor=PANEL,
               edgecolor=BORDER, labelcolor=TEXT_MED, framealpha=0.9)

txt_r = ax_time.text(0.98, 0.96, "", transform=ax_time.transAxes,
                      fontsize=18, color=RANGE_CYAN, fontweight="bold",
                      ha="right", va="top", fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor=BG, edgecolor=BORDER, alpha=0.85))
txt_a = ax_time.text(0.98, 0.78, "", transform=ax_time.transAxes,
                      fontsize=14, color=ANGLE_AMBER, fontweight="bold",
                      ha="right", va="top", fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor=BG, edgecolor=BORDER, alpha=0.85))

# ── [BOTTOM-LEFT] Inference Latency ─────────────────────────────
ax_lat = fig.add_subplot(gs[1, 0], facecolor=PANEL)
ax_lat.set_title("Inference Latency", color=LATENCY_GREEN, fontsize=11,
                  fontweight="bold", fontfamily="monospace")
ax_lat.set_xlabel("Time (s)", color=TEXT_DIM, fontsize=8)
ax_lat.set_ylabel("us", color=LATENCY_GREEN, fontsize=9)
ax_lat.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.4)
ax_lat.tick_params(colors=TEXT_DIM, labelsize=7)
for s in ("top", "right"):
    ax_lat.spines[s].set_visible(False)
ax_lat.spines["bottom"].set_color(BORDER)
ax_lat.spines["left"].set_color(BORDER)

ln_lat, = ax_lat.plot([], [], color=LATENCY_GREEN, linewidth=1.6, alpha=0.8)
fill_lat = None
ax_lat.axhline(y=500, color=ACCENT_RED, linewidth=0.8, linestyle=":",
               alpha=0.5, label="500 us budget")
ax_lat.legend(loc="upper right", fontsize=7, facecolor=PANEL,
              edgecolor=BORDER, labelcolor=TEXT_MED)

txt_lat = ax_lat.text(0.02, 0.92, "", transform=ax_lat.transAxes,
                       fontsize=13, color=LATENCY_GREEN, fontweight="bold",
                       va="top", fontfamily="monospace",
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor=BG, edgecolor=BORDER, alpha=0.85))

# ── [BOTTOM-RIGHT] Stats ────────────────────────────────────────
ax_st = fig.add_subplot(gs[1, 1], facecolor=PANEL)
ax_st.set_xlim(0, 1); ax_st.set_ylim(0, 1); ax_st.axis("off")
ax_st.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_st.transAxes,
                                facecolor=PANEL, edgecolor=BORDER,
                                linewidth=1.2, zorder=0))

_ls = dict(fontsize=8, color=TEXT_DIM, va="top",
           transform=ax_st.transAxes, fontfamily="monospace")
_vs = dict(fontsize=16, fontweight="bold", va="bottom",
           transform=ax_st.transAxes, fontfamily="monospace")

ax_st.text(0.06, 0.92, "FUSED EVENTS", **_ls)
st_cnt = ax_st.text(0.06, 0.55, "0", color=FUSED_MAGENTA, **_vs)

ax_st.text(0.30, 0.92, "RATE (Hz)", **_ls)
st_hz  = ax_st.text(0.30, 0.55, "--", color=RANGE_CYAN, **_vs)

ax_st.text(0.54, 0.92, "AVG LATENCY", **_ls)
st_avg = ax_st.text(0.54, 0.55, "--", color=LATENCY_GREEN, **_vs)

ax_st.text(0.78, 0.92, "UPTIME", **_ls)
st_up  = ax_st.text(0.78, 0.55, "0s", color=ACCENT_PURPLE, **_vs)

ax_st.text(0.06, 0.38, "MIN / MAX LAT", **_ls)
st_mm  = ax_st.text(0.06, 0.05, "--", color=TEXT_MED, **_vs)

ax_st.text(0.40, 0.38, "POSITION", **_ls)
st_pos = ax_st.text(0.40, 0.05, "--", color=FUSED_GLOW, **_vs)

ax_st.text(0.78, 0.38, "MODEL", **_ls)
ax_st.text(0.78, 0.05, "F32 MLP\n2474 p", color=TEXT_DIM, **_vs)

# ── Suptitle ─────────────────────────────────────────────────────
fig.suptitle("ISAC  ·  ML Sensor Fusion Dashboard",
             color=TEXT_HI, fontsize=16, fontweight="bold",
             fontfamily="monospace", x=0.5, y=0.97)

# ──────────────────────── ANIMATION ──────────────────────────────
_t0 = None

def update(_frame):
    global _t0, fill_lat

    while ser.in_waiting > 0:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw.startswith("FUSED,"):
                continue
            parts = raw.split(",")
            if len(parts) != 5:
                continue

            r_m       = float(parts[1])
            theta_deg = float(parts[2])
            ts_ms     = int(parts[3])
            lat_us    = int(parts[4])

            if _t0 is None:
                _t0 = ts_ms
            ts_s = (ts_ms - _t0) / 1000.0

            # Buffers
            trail_r.append(r_m)
            trail_theta.append(np.radians(theta_deg))
            ts_buf.append(ts_s)
            range_buf.append(r_m)
            angle_buf.append(theta_deg)
            lat_buf.append(lat_us)
            lat_ts_buf.append(ts_s)

            # Stats
            stats["count"] += 1
            stats["sum_lat"] += lat_us
            if lat_us < stats["min_lat"]: stats["min_lat"] = lat_us
            if lat_us > stats["max_lat"]: stats["max_lat"] = lat_us
            stats["last_r"]     = r_m
            stats["last_theta"] = theta_deg
            stats["last_lat"]   = lat_us

        except Exception:
            pass

    # ── Polar ────────────────────────────────────────────────────
    n = len(trail_r)
    if n > 0:
        offs = np.c_[list(trail_theta), list(trail_r)]
        sc_trail.set_offsets(offs)
        sc_trail.set_sizes(np.linspace(10, 160, n))
        sc_trail.set_facecolors(_tc[-n:])
        sc_head.set_offsets([[trail_theta[-1], trail_r[-1]]])
        polar_lbl.set_text(f"{trail_r[-1]:.2f}m  {np.degrees(trail_theta[-1]):.0f}deg")
    else:
        sc_trail.set_offsets(np.empty((0, 2)))
        sc_head.set_offsets(np.empty((0, 2)))

    # Dynamic polar Y limit
    if n > 0:
        mr = max(trail_r)
        ax_polar.set_ylim(0, max(mr * 1.4, 2.0))

    # ── Timeline ─────────────────────────────────────────────────
    if len(ts_buf) > 1:
        ta = np.array(ts_buf); ra = np.array(range_buf); aa = np.array(angle_buf)
        ln_r.set_data(ta, ra)
        ln_a.set_data(ta, aa)
        dot_r.set_offsets([[ta[-1], ra[-1]]])
        dot_a.set_offsets([[ta[-1], aa[-1]]])

        t_hi = ta[-1] + 0.5
        t_lo = max(0, ta[-1] - 30)
        ax_time.set_xlim(t_lo, t_hi)
        ax_time.set_ylim(max(ra.min() - 0.3, 0), ra.max() + 0.5)
        ax_ang.set_ylim(aa.min() - 5, aa.max() + 5)

        txt_r.set_text(f"{ra[-1]:.3f} m")
        txt_a.set_text(f"{aa[-1]:.1f} deg")

    # ── Latency ──────────────────────────────────────────────────
    if len(lat_ts_buf) > 1:
        lt = np.array(lat_ts_buf); lv = np.array(lat_buf)
        ln_lat.set_data(lt, lv)

        if fill_lat is not None:
            fill_lat.remove()
        fill_lat = ax_lat.fill_between(lt, 0, lv,
                                        color=LATENCY_GREEN, alpha=0.12)

        ax_lat.set_xlim(max(0, lt[-1] - 20), lt[-1] + 0.5)
        ax_lat.set_ylim(0, max(lv.max() * 1.3, 300))
        txt_lat.set_text(f"{lv[-1]} us")

    # ── Stats ────────────────────────────────────────────────────
    c = stats["count"]
    st_cnt.set_text(str(c))

    elapsed = time.time() - stats["start"]
    hz = c / elapsed if elapsed > 0 else 0
    st_hz.set_text(f"{hz:.0f}")

    avg = stats["sum_lat"] / c if c > 0 else 0
    st_avg.set_text(f"{avg:.0f} us")

    m, s = divmod(int(elapsed), 60)
    st_up.set_text(f"{m}m {s:02d}s" if m else f"{s}s")

    if c > 0:
        st_mm.set_text(f"{stats['min_lat']} / {stats['max_lat']} us")
        st_pos.set_text(f"({stats['last_r']:.2f}m, {stats['last_theta']:.0f}deg)")

    return (sc_trail, sc_head, ln_r, ln_a, ln_lat)


ani = animation.FuncAnimation(fig, update, interval=1000 // FPS,
                               blit=False, cache_frame_data=False)

print("+" + "=" * 54 + "+")
print("|  ISAC ML Fusion Dashboard -- Live from COM13        |")
print("|  Close the window or Ctrl+C to exit.                |")
print("+" + "=" * 54 + "+")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nExiting.")
finally:
    ser.close()
    print("Serial closed.")
