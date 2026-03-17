"""
═══════════════════════════════════════════════════════════════════════════
  ISAC Fused Sensor Dashboard — polar_logger_v2 Visualizer
  
  Reads CSV lines from COM18:
      R,<r_m>,<theta_deg>,<ts_ms>
      W,<r_m>,NAN,<ts_ms>
  
  3-panel layout:
    [LEFT]   Polar radar sweep with trailing ghost effect
    [RIGHT]  Dual-line rolling range chart (Radar + Wi-Fi FTM)
    [BOTTOM] Live statistics bar
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
MAX_TRAIL     = 20      # radar polar trail length
MAX_TIMELINE  = 200     # rolling timeline points
HAMPEL_WINDOW = 5
HAMPEL_SIGMA  = 2.5
KALMAN_Q      = 1e-4
KALMAN_R      = 0.04
FPS           = 30

# ──────────────────────── SERIAL ─────────────────────────────────
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.005)
    print(f"✓ Connected to {PORT}")
except Exception as e:
    print(f"✗ Cannot open {PORT} — close ESP-IDF monitor first!")
    print(e)
    sys.exit(1)

# ──────────────────────── COLOUR PALETTE ─────────────────────────
BG          = "#0B0F19"
PANEL_BG    = "#111827"
GRID        = "#1E293B"
BORDER      = "#334155"
TEXT_DIM    = "#64748B"
TEXT_MED    = "#94A3B8"
TEXT_BRIGHT = "#E2E8F0"

RADAR_CYAN      = "#22D3EE"   # cyan-400
RADAR_GLOW      = "#06B6D4"   # cyan-500
WIFI_AMBER      = "#FBBF24"   # amber-400
WIFI_GLOW       = "#F59E0B"   # amber-500
ACCENT_GREEN    = "#34D399"   # emerald-400
ACCENT_RED      = "#F87171"   # red-400
ACCENT_PURPLE   = "#A78BFA"   # violet-400

# ──────────────────────── DATA BUFFERS ───────────────────────────
# Radar polar trail
radar_r     = collections.deque(maxlen=MAX_TRAIL)
radar_theta = collections.deque(maxlen=MAX_TRAIL)

# Timeline
radar_ts      = collections.deque(maxlen=MAX_TIMELINE)
radar_range   = collections.deque(maxlen=MAX_TIMELINE)

wifi_ts       = collections.deque(maxlen=MAX_TIMELINE)
wifi_range_raw = collections.deque(maxlen=MAX_TIMELINE)
wifi_range_sm  = collections.deque(maxlen=MAX_TIMELINE)

# Stats counters
stats = {
    "radar_ok": 0, "radar_nan": 0,
    "wifi_ok": 0,  "wifi_total": 0,
    "start_time": time.time(),
}

# ──────────────────────── KALMAN FILTER ──────────────────────────
class Kalman1D:
    def __init__(self, q=KALMAN_Q, r=KALMAN_R):
        self.q, self.r = q, r
        self.est, self.err = None, 1.0
    def update(self, z):
        if self.est is None:
            self.est = z
            return z
        pred = self.est
        err_p = self.err + self.q
        gain = err_p / (err_p + self.r)
        self.est = pred + gain * (z - pred)
        self.err = (1 - gain) * err_p
        return self.est

kf_wifi = Kalman1D()

def hampel(val, buf, win=HAMPEL_WINDOW, sigma=HAMPEL_SIGMA):
    if len(buf) < win:
        return val
    window = np.array(list(buf)[-win:] + [val])
    med = np.median(window)
    mad = np.median(np.abs(window - med))
    if mad > 0 and abs(val - med) > sigma * mad:
        return float(med)
    return val

# ──────────────────────── FIGURE SETUP ───────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 9), facecolor=BG, dpi=100)
fig.canvas.manager.set_window_title("ISAC Fused Sensor Dashboard")

gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1],
                       width_ratios=[1, 1.4],
                       hspace=0.28, wspace=0.25,
                       left=0.06, right=0.97, top=0.92, bottom=0.06)

# ── [TOP-LEFT] Polar Radar ──────────────────────────────────────
ax_polar = fig.add_subplot(gs[0, 0], polar=True, facecolor=PANEL_BG)
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)
ax_polar.set_thetamin(-60)
ax_polar.set_thetamax(60)
ax_polar.set_ylim(0, 4)
ax_polar.set_rticks([1, 2, 3, 4])
ax_polar.set_rlabel_position(22)
ax_polar.grid(color=GRID, linestyle="--", linewidth=0.6, alpha=0.6)
ax_polar.tick_params(colors=TEXT_DIM, labelsize=8)
for spine in ax_polar.spines.values():
    spine.set_color(BORDER)
ax_polar.set_title("Radar Polar View", color=RADAR_CYAN, fontsize=13,
                    fontweight="bold", pad=18, fontfamily="monospace")

# Radar origin marker
ax_polar.scatter([0], [0], c=ACCENT_GREEN, s=90, zorder=5, marker="P",
                  edgecolors="white", linewidths=0.6)

# Precompute trail colour ramp (transparent → solid cyan)
_trail_colors = np.zeros((MAX_TRAIL, 4))
for i in range(MAX_TRAIL):
    a = ((i + 1) / MAX_TRAIL) ** 1.6        # ease-in curve
    _trail_colors[i] = [0.133, 0.827, 0.933, a]  # #22D3EE

scatter_trail = ax_polar.scatter([], [], s=50, zorder=3)
scatter_head  = ax_polar.scatter([], [], c=RADAR_CYAN, s=220, zorder=4,
                                  edgecolors="white", linewidths=1.0)

polar_label = ax_polar.text(np.radians(55), 3.6, "", fontsize=11,
                             color=RADAR_CYAN, fontweight="bold",
                             ha="center", fontfamily="monospace")

# ── [TOP-RIGHT] Range Timeline ──────────────────────────────────
ax_time = fig.add_subplot(gs[0, 1], facecolor=PANEL_BG)
ax_time.set_title("Range Timeline", color=TEXT_BRIGHT, fontsize=13,
                   fontweight="bold", fontfamily="monospace")
ax_time.set_xlabel("Time (s)", color=TEXT_DIM, fontsize=9)
ax_time.set_ylabel("Range (m)", color=TEXT_DIM, fontsize=9)
ax_time.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.5)
ax_time.tick_params(colors=TEXT_DIM, labelsize=8)
for side in ("top", "right"):
    ax_time.spines[side].set_visible(False)
for side in ("bottom", "left"):
    ax_time.spines[side].set_color(BORDER)
ax_time.set_ylim(-0.1, 5.0)

line_radar, = ax_time.plot([], [], color=RADAR_CYAN, linewidth=2.0,
                            alpha=0.85, label="Radar")
line_wifi_raw, = ax_time.plot([], [], color=WIFI_AMBER, linewidth=1.0,
                               alpha=0.35, linestyle="--", label="Wi-Fi Raw")
line_wifi_sm, = ax_time.plot([], [], color=WIFI_AMBER, linewidth=2.4,
                              alpha=0.9, label="Wi-Fi Smooth")

dot_radar = ax_time.scatter([], [], c=RADAR_CYAN, s=60, zorder=4,
                              edgecolors="white", linewidths=0.6)
dot_wifi  = ax_time.scatter([], [], c=WIFI_AMBER, s=60, zorder=4,
                              edgecolors="white", linewidths=0.6)

ax_time.legend(loc="upper left", fontsize=8, facecolor=PANEL_BG,
               edgecolor=BORDER, labelcolor=TEXT_MED, framealpha=0.9)

time_text_radar = ax_time.text(0.98, 0.96, "", transform=ax_time.transAxes,
                                fontsize=16, color=RADAR_CYAN,
                                fontweight="bold", ha="right", va="top",
                                fontfamily="monospace",
                                bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor=BG, edgecolor=BORDER,
                                          alpha=0.85))
time_text_wifi = ax_time.text(0.98, 0.80, "", transform=ax_time.transAxes,
                               fontsize=16, color=WIFI_AMBER,
                               fontweight="bold", ha="right", va="top",
                               fontfamily="monospace",
                               bbox=dict(boxstyle="round,pad=0.3",
                                         facecolor=BG, edgecolor=BORDER,
                                         alpha=0.85))

# ── [BOTTOM] Status Bar ─────────────────────────────────────────
ax_stat = fig.add_subplot(gs[1, :], facecolor=PANEL_BG)
ax_stat.set_xlim(0, 1)
ax_stat.set_ylim(0, 1)
ax_stat.axis("off")
for side in ax_stat.spines.values():
    side.set_color(BORDER)

# Static decorations
ax_stat.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_stat.transAxes,
                                 facecolor=PANEL_BG, edgecolor=BORDER,
                                 linewidth=1.2, zorder=0))

_lbl_style = dict(fontsize=8, color=TEXT_DIM, va="top",
                   transform=ax_stat.transAxes, fontfamily="monospace")
_val_style = dict(fontsize=14, fontweight="bold", va="bottom",
                   transform=ax_stat.transAxes, fontfamily="monospace")

ax_stat.text(0.04, 0.88, "RADAR HITS", **_lbl_style)
stat_radar_ok = ax_stat.text(0.04, 0.28, "0", color=RADAR_CYAN, **_val_style)

ax_stat.text(0.18, 0.88, "RADAR NaN", **_lbl_style)
stat_radar_nan = ax_stat.text(0.18, 0.28, "0", color=ACCENT_RED, **_val_style)

ax_stat.text(0.32, 0.88, "RADAR HIT %", **_lbl_style)
stat_radar_pct = ax_stat.text(0.32, 0.28, "—", color=ACCENT_GREEN, **_val_style)

ax_stat.text(0.50, 0.88, "WI-FI HITS", **_lbl_style)
stat_wifi_ok = ax_stat.text(0.50, 0.28, "0", color=WIFI_AMBER, **_val_style)

ax_stat.text(0.66, 0.88, "UPTIME", **_lbl_style)
stat_uptime = ax_stat.text(0.66, 0.28, "0s", color=ACCENT_PURPLE, **_val_style)

ax_stat.text(0.82, 0.88, "DATA RATE", **_lbl_style)
stat_rate = ax_stat.text(0.82, 0.28, "—", color=TEXT_BRIGHT, **_val_style)

# ── Suptitle ─────────────────────────────────────────────────────
fig.suptitle("ISAC  ·  Fused Sensor Dashboard",
             color=TEXT_BRIGHT, fontsize=17, fontweight="bold",
             fontfamily="monospace", x=0.5, y=0.97)

# ──────────────────────── ANIMATION LOOP ─────────────────────────
_t0 = None

def update(frame_num):
    global _t0

    # ── Read all pending serial lines ────────────────────────────
    while ser.in_waiting > 0:
        try:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue
            parts = raw_line.split(",")
            if len(parts) != 4:
                continue

            kind = parts[0]
            r_str, th_str, ts_str = parts[1], parts[2], parts[3]

            try:
                ts_ms = int(ts_str)
            except ValueError:
                continue

            if _t0 is None:
                _t0 = ts_ms

            ts_s = (ts_ms - _t0) / 1000.0

            # ── RADAR ────────────────────────────────────────────
            if kind == "R":
                if r_str == "NAN" or th_str == "NAN":
                    stats["radar_nan"] += 1
                    # Clear trail on dropout
                    radar_r.clear()
                    radar_theta.clear()
                else:
                    r_val = float(r_str)
                    th_val = float(th_str)
                    stats["radar_ok"] += 1
                    radar_r.append(r_val)
                    radar_theta.append(np.radians(th_val))
                    radar_ts.append(ts_s)
                    radar_range.append(r_val)

            # ── WI-FI FTM ────────────────────────────────────────
            elif kind == "W":
                if r_str == "NAN":
                    stats["wifi_total"] += 1
                else:
                    r_val = float(r_str)
                    stats["wifi_ok"] += 1
                    stats["wifi_total"] += 1
                    wifi_ts.append(ts_s)
                    wifi_range_raw.append(r_val)
                    r_h = hampel(r_val, wifi_range_raw)
                    r_s = kf_wifi.update(r_h)
                    wifi_range_sm.append(r_s)

        except Exception:
            pass

    # ── UPDATE POLAR ─────────────────────────────────────────────
    n = len(radar_r)
    if n > 0:
        offsets = np.c_[list(radar_theta), list(radar_r)]
        scatter_trail.set_offsets(offsets)
        sizes = np.linspace(15, 180, n)
        scatter_trail.set_sizes(sizes)
        scatter_trail.set_facecolors(_trail_colors[-n:])
        scatter_head.set_offsets([[radar_theta[-1], radar_r[-1]]])
        polar_label.set_text(f"{radar_r[-1]:.2f} m")
    else:
        scatter_trail.set_offsets(np.empty((0, 2)))
        scatter_head.set_offsets(np.empty((0, 2)))
        polar_label.set_text("—")

    # ── UPDATE TIMELINE ──────────────────────────────────────────
    if len(radar_ts) > 1:
        rt = np.array(radar_ts)
        rr = np.array(radar_range)
        line_radar.set_data(rt, rr)
        dot_radar.set_offsets([[rt[-1], rr[-1]]])
        time_text_radar.set_text(f"R {rr[-1]:.3f} m")
    else:
        time_text_radar.set_text("R —")

    if len(wifi_ts) > 1:
        wt = np.array(wifi_ts)
        wr = np.array(wifi_range_raw)
        ws = np.array(wifi_range_sm)
        line_wifi_raw.set_data(wt, wr)
        line_wifi_sm.set_data(wt, ws)
        dot_wifi.set_offsets([[wt[-1], ws[-1]]])
        time_text_wifi.set_text(f"W {ws[-1]:.3f} m")
    else:
        time_text_wifi.set_text("W —")

    # Auto-scale X
    all_ts = list(radar_ts) + list(wifi_ts)
    if all_ts:
        t_max = max(all_ts)
        ax_time.set_xlim(max(0, t_max - 30), t_max + 0.5)

    # Auto-scale Y
    all_r = list(radar_range) + list(wifi_range_sm)
    if all_r:
        y_hi = max(max(all_r) * 1.3, 1.0)
        ax_time.set_ylim(-0.1, min(y_hi, 10.0))

    # ── UPDATE STATUS BAR ────────────────────────────────────────
    rok = stats["radar_ok"]
    rnan = stats["radar_nan"]
    rtot = rok + rnan
    pct = f"{100 * rok / rtot:.1f}%" if rtot > 0 else "—"

    stat_radar_ok.set_text(str(rok))
    stat_radar_nan.set_text(str(rnan))
    stat_radar_pct.set_text(pct)
    stat_wifi_ok.set_text(str(stats["wifi_ok"]))

    elapsed = time.time() - stats["start_time"]
    mins, secs = divmod(int(elapsed), 60)
    stat_uptime.set_text(f"{mins}m {secs:02d}s" if mins else f"{secs}s")

    total_events = rok + rnan + stats["wifi_total"]
    rate = total_events / elapsed if elapsed > 0 else 0
    stat_rate.set_text(f"{rate:.0f} evt/s")

    return (scatter_trail, scatter_head, line_radar, line_wifi_raw,
            line_wifi_sm, dot_radar, dot_wifi)


ani = animation.FuncAnimation(fig, update, interval=1000 // FPS,
                               blit=False, cache_frame_data=False)

print("╔══════════════════════════════════════════════════════╗")
print("║   ISAC Fused Sensor Dashboard — Live from COM18     ║")
print("║   Close the window or Ctrl+C to exit.               ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    plt.show()
except KeyboardInterrupt:
    print("\n✓ Exiting gracefully.")
finally:
    ser.close()
    print("✓ Serial port closed.")
