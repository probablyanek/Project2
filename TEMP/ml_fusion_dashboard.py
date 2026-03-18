"""
═══════════════════════════════════════════════════════════════════════════
  ISAC ML Fusion Dashboard v2 — Verbose Telemetry Visualizer

  Reads FUSED_V2 CSV lines from COM13:
      FUSED_V2,<ts_ms>,<inference_us>,
               <fused_r>,<fused_theta>,
               <radar_r>,<radar_theta>,<radar_fresh>,
               <wifi_r>,<wifi_fresh>

  4-panel layout:
    [TOP-LEFT]    Polar — Fused (magenta) + Radar raw (amber) + WiFi arc
    [TOP-RIGHT]   Range timeline — Fused vs Radar vs WiFi overlaid
    [BOTTOM-LEFT] Inference latency + sensor freshness strip
    [BOTTOM-RIGHT] Expanded live stats + sensor health badges
═══════════════════════════════════════════════════════════════════════════
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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

FUSED_MAGENTA   = "#E040FB"
FUSED_GLOW      = "#CE93D8"
RANGE_CYAN      = "#00E5FF"
ANGLE_AMBER     = "#FFD740"
LATENCY_GREEN   = "#69F0AE"
ACCENT_RED      = "#FF5252"
ACCENT_PURPLE   = "#B388FF"

RADAR_AMBER     = "#FFA726"   # raw radar colour
RADAR_AMBER_DIM = "#7A5018"
WIFI_TEAL       = "#26C6DA"   # raw wifi colour
WIFI_TEAL_DIM   = "#136068"
FRESH_GREEN     = "#00E676"
STALE_RED       = "#FF5252"

# ──────────────────────── DATA BUFFERS ───────────────────────────
# Fused trail (polar)
trail_r     = collections.deque(maxlen=MAX_TRAIL)
trail_theta = collections.deque(maxlen=MAX_TRAIL)

# Timelines
ts_buf        = collections.deque(maxlen=MAX_TIMELINE)
fused_r_buf   = collections.deque(maxlen=MAX_TIMELINE)
fused_a_buf   = collections.deque(maxlen=MAX_TIMELINE)
radar_r_buf   = collections.deque(maxlen=MAX_TIMELINE)
radar_a_buf   = collections.deque(maxlen=MAX_TIMELINE)
wifi_r_buf    = collections.deque(maxlen=MAX_TIMELINE)

# Latency
lat_buf     = collections.deque(maxlen=MAX_LATENCY)
lat_ts_buf  = collections.deque(maxlen=MAX_LATENCY)

# Freshness strips
radar_fresh_buf = collections.deque(maxlen=MAX_TIMELINE)
wifi_fresh_buf  = collections.deque(maxlen=MAX_TIMELINE)

stats = {
    "count": 0,
    "start": time.time(),
    "min_lat": 9999,
    "max_lat": 0,
    "sum_lat": 0,
    "last_fused_r": 0.0,
    "last_fused_theta": 0.0,
    "last_radar_r": 0.0,
    "last_radar_theta": 0.0,
    "last_wifi_r": 0.0,
    "last_lat": 0,
    "radar_fresh": 0.0,
    "wifi_fresh": 0.0,
    "radar_fresh_pct": 0.0,
    "wifi_fresh_pct": 0.0,
    "radar_fresh_sum": 0,
    "wifi_fresh_sum": 0,
}

# ──────────────────────── FIGURE SETUP ───────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(17, 10), facecolor=BG, dpi=96)
fig.canvas.manager.set_window_title("ISAC ML Fusion Dashboard v2")

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
ax_polar.set_title("Sensor Fusion View", color=FUSED_MAGENTA, fontsize=12,
                    fontweight="bold", pad=16, fontfamily="monospace")

# Origin marker
ax_polar.scatter([0], [0], c=LATENCY_GREEN, s=80, zorder=5, marker="P",
                  edgecolors="white", linewidths=0.5)

# Fused trail colours (fading magenta)
_tc = np.zeros((MAX_TRAIL, 4))
for i in range(MAX_TRAIL):
    a = ((i + 1) / MAX_TRAIL) ** 1.8
    _tc[i] = [0.878, 0.251, 0.984, a]

sc_trail = ax_polar.scatter([], [], s=40, zorder=3)
sc_head  = ax_polar.scatter([], [], c=FUSED_MAGENTA, s=250, zorder=4,
                             edgecolors="white", linewidths=1.2)

# Raw radar marker (amber diamond)
sc_radar = ax_polar.scatter([], [], c=RADAR_AMBER, s=100, zorder=3,
                             marker="D", edgecolors="white", linewidths=0.5,
                             alpha=0.7, label="Radar")

# Raw wifi arc marker (teal ring — we'll use a hollow circle at wifi_r)
sc_wifi = ax_polar.scatter([], [], facecolors="none", edgecolors=WIFI_TEAL,
                            s=180, zorder=2, linewidths=1.8, alpha=0.6,
                            label="WiFi FTM")

polar_lbl = ax_polar.text(np.radians(52), 4.5, "", fontsize=10,
                           color=FUSED_MAGENTA, fontweight="bold",
                           ha="center", fontfamily="monospace")

# Small legend inside polar
ax_polar.legend(loc="lower left", fontsize=6, facecolor=PANEL,
                edgecolor=BORDER, labelcolor=TEXT_MED, framealpha=0.8,
                bbox_to_anchor=(0.0, -0.12))

# ── [TOP-RIGHT] Range Timeline ──────────────────────────────────
ax_time = fig.add_subplot(gs[0, 1], facecolor=PANEL)
ax_time.set_title("Range Comparison", color=TEXT_HI, fontsize=12,
                   fontweight="bold", fontfamily="monospace")
ax_time.set_xlabel("Time (s)", color=TEXT_DIM, fontsize=8)
ax_time.set_ylabel("Range (m)", color=TEXT_HI, fontsize=9)
ax_time.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.4)
ax_time.tick_params(axis="y", colors=TEXT_MED, labelsize=8)
ax_time.tick_params(axis="x", colors=TEXT_DIM, labelsize=7)
for s in ("top", "right"):
    ax_time.spines[s].set_visible(False)
ax_time.spines["bottom"].set_color(BORDER)
ax_time.spines["left"].set_color(BORDER)

# Three range lines: fused (bright), radar (dim), wifi (dim)
ln_fused, = ax_time.plot([], [], color=FUSED_MAGENTA, linewidth=2.5,
                          alpha=0.95, label="Fused", zorder=3)
ln_radar, = ax_time.plot([], [], color=RADAR_AMBER, linewidth=1.4,
                          alpha=0.5, linestyle="--", label="Radar",
                          zorder=2)
ln_wifi,  = ax_time.plot([], [], color=WIFI_TEAL, linewidth=1.4,
                          alpha=0.5, linestyle=":", label="WiFi FTM",
                          zorder=2)

dot_fused = ax_time.scatter([], [], c=FUSED_MAGENTA, s=50, zorder=4,
                             edgecolors="white", linewidths=0.5)

ax_time.legend(loc="upper left", fontsize=7, facecolor=PANEL,
               edgecolor=BORDER, labelcolor=TEXT_MED, framealpha=0.9)

# Overlay value badges
txt_fused = ax_time.text(0.98, 0.96, "", transform=ax_time.transAxes,
                          fontsize=16, color=FUSED_MAGENTA, fontweight="bold",
                          ha="right", va="top", fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor=BG, edgecolor=BORDER, alpha=0.85))
txt_radar_v = ax_time.text(0.98, 0.78, "", transform=ax_time.transAxes,
                            fontsize=10, color=RADAR_AMBER, fontweight="bold",
                            ha="right", va="top", fontfamily="monospace",
                            alpha=0.8)
txt_wifi_v = ax_time.text(0.98, 0.66, "", transform=ax_time.transAxes,
                           fontsize=10, color=WIFI_TEAL, fontweight="bold",
                           ha="right", va="top", fontfamily="monospace",
                           alpha=0.8)

# ── [BOTTOM-LEFT] Inference Latency + Freshness ─────────────────
ax_lat = fig.add_subplot(gs[1, 0], facecolor=PANEL)
ax_lat.set_title("Inference Latency", color=LATENCY_GREEN, fontsize=11,
                  fontweight="bold", fontfamily="monospace")
ax_lat.set_xlabel("Time (s)", color=TEXT_DIM, fontsize=8)
ax_lat.set_ylabel("μs", color=LATENCY_GREEN, fontsize=9)
ax_lat.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.4)
ax_lat.tick_params(colors=TEXT_DIM, labelsize=7)
for s in ("top", "right"):
    ax_lat.spines[s].set_visible(False)
ax_lat.spines["bottom"].set_color(BORDER)
ax_lat.spines["left"].set_color(BORDER)

ln_lat, = ax_lat.plot([], [], color=LATENCY_GREEN, linewidth=1.6, alpha=0.8)
fill_lat = None
ax_lat.axhline(y=500, color=ACCENT_RED, linewidth=0.8, linestyle=":",
               alpha=0.5, label="500 μs budget")
ax_lat.legend(loc="upper right", fontsize=7, facecolor=PANEL,
              edgecolor=BORDER, labelcolor=TEXT_MED)

txt_lat = ax_lat.text(0.02, 0.92, "", transform=ax_lat.transAxes,
                       fontsize=13, color=LATENCY_GREEN, fontweight="bold",
                       va="top", fontfamily="monospace",
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor=BG, edgecolor=BORDER, alpha=0.85))

# ── [BOTTOM-RIGHT] Expanded Stats ───────────────────────────────
ax_st = fig.add_subplot(gs[1, 1], facecolor=PANEL)
ax_st.set_xlim(0, 1); ax_st.set_ylim(0, 1); ax_st.axis("off")
ax_st.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_st.transAxes,
                                facecolor=PANEL, edgecolor=BORDER,
                                linewidth=1.2, zorder=0))

_ls = dict(fontsize=7, color=TEXT_DIM, va="top",
           transform=ax_st.transAxes, fontfamily="monospace")
_vs = dict(fontsize=14, fontweight="bold", va="bottom",
           transform=ax_st.transAxes, fontfamily="monospace")
_vs_sm = dict(fontsize=11, fontweight="bold", va="bottom",
              transform=ax_st.transAxes, fontfamily="monospace")

# Row 1: Counters
ax_st.text(0.04, 0.95, "EVENTS", **_ls)
st_cnt = ax_st.text(0.04, 0.72, "0", color=FUSED_MAGENTA, **_vs)

ax_st.text(0.22, 0.95, "RATE", **_ls)
st_hz  = ax_st.text(0.22, 0.72, "--", color=RANGE_CYAN, **_vs)

ax_st.text(0.38, 0.95, "AVG LAT", **_ls)
st_avg = ax_st.text(0.38, 0.72, "--", color=LATENCY_GREEN, **_vs)

ax_st.text(0.58, 0.95, "MIN/MAX", **_ls)
st_mm  = ax_st.text(0.58, 0.72, "--", color=TEXT_MED, **_vs_sm)

ax_st.text(0.82, 0.95, "UPTIME", **_ls)
st_up  = ax_st.text(0.82, 0.72, "0s", color=ACCENT_PURPLE, **_vs)

# Row 2: Fused position
ax_st.text(0.04, 0.62, "FUSED POS", **_ls)
st_pos = ax_st.text(0.04, 0.42, "--", color=FUSED_GLOW, **_vs_sm)

# Row 2: Sensor health badges
ax_st.text(0.40, 0.62, "RADAR", **_ls)
st_radar_badge = ax_st.text(0.40, 0.42, "WAIT", color=TEXT_DIM, **_vs_sm)
st_radar_val   = ax_st.text(0.40, 0.34, "", fontsize=8, color=RADAR_AMBER,
                             transform=ax_st.transAxes, fontfamily="monospace",
                             va="top", alpha=0.8)

ax_st.text(0.66, 0.62, "WiFi FTM", **_ls)
st_wifi_badge = ax_st.text(0.66, 0.42, "WAIT", color=TEXT_DIM, **_vs_sm)
st_wifi_val   = ax_st.text(0.66, 0.34, "", fontsize=8, color=WIFI_TEAL,
                            transform=ax_st.transAxes, fontfamily="monospace",
                            va="top", alpha=0.8)

# Row 2: Model info
ax_st.text(0.86, 0.62, "MODEL", **_ls)
ax_st.text(0.86, 0.42, "F32 MLP", color=TEXT_DIM, **_vs_sm)

# Row 3: Trust bars
ax_st.text(0.04, 0.22, "TRUST", **_ls)

# Radar trust bar background
_bar_y = 0.08
_bar_h = 0.06
_bar_w = 0.35
ax_st.add_patch(plt.Rectangle((0.04, _bar_y), _bar_w, _bar_h,
                                transform=ax_st.transAxes,
                                facecolor=RADAR_AMBER_DIM, edgecolor=BORDER,
                                linewidth=0.5, zorder=1))
trust_radar_bar = ax_st.add_patch(
    plt.Rectangle((0.04, _bar_y), 0.001, _bar_h,
                   transform=ax_st.transAxes,
                   facecolor=RADAR_AMBER, alpha=0.7, zorder=2))
ax_st.text(0.04, _bar_y + _bar_h + 0.01, "RADAR", fontsize=6,
           color=RADAR_AMBER, transform=ax_st.transAxes, va="bottom",
           fontfamily="monospace", alpha=0.7)
st_radar_pct = ax_st.text(0.04 + _bar_w + 0.02, _bar_y + _bar_h / 2, "",
                           fontsize=8, color=RADAR_AMBER,
                           transform=ax_st.transAxes, va="center",
                           fontfamily="monospace")

# WiFi trust bar background
ax_st.add_patch(plt.Rectangle((0.55, _bar_y), _bar_w, _bar_h,
                                transform=ax_st.transAxes,
                                facecolor=WIFI_TEAL_DIM, edgecolor=BORDER,
                                linewidth=0.5, zorder=1))
trust_wifi_bar = ax_st.add_patch(
    plt.Rectangle((0.55, _bar_y), 0.001, _bar_h,
                   transform=ax_st.transAxes,
                   facecolor=WIFI_TEAL, alpha=0.7, zorder=2))
ax_st.text(0.55, _bar_y + _bar_h + 0.01, "WiFi FTM", fontsize=6,
           color=WIFI_TEAL, transform=ax_st.transAxes, va="bottom",
           fontfamily="monospace", alpha=0.7)
st_wifi_pct = ax_st.text(0.55 + _bar_w + 0.02, _bar_y + _bar_h / 2, "",
                          fontsize=8, color=WIFI_TEAL,
                          transform=ax_st.transAxes, va="center",
                          fontfamily="monospace")

# ── Suptitle ─────────────────────────────────────────────────────
fig.suptitle("ISAC  ·  ML Sensor Fusion Dashboard  v2",
             color=TEXT_HI, fontsize=16, fontweight="bold",
             fontfamily="monospace", x=0.5, y=0.97)

# ──────────────────────── ANIMATION ──────────────────────────────
_t0 = None

def update(_frame):
    global _t0, fill_lat

    while ser.in_waiting > 0:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw.startswith("FUSED_V2,"):
                continue
            parts = raw.split(",")
            if len(parts) != 10:
                continue

            ts_ms         = int(parts[1])
            lat_us        = int(parts[2])
            fused_r       = float(parts[3])
            fused_theta   = float(parts[4])
            radar_r       = float(parts[5])
            radar_theta   = float(parts[6])
            radar_fresh   = float(parts[7])
            wifi_r        = float(parts[8])
            wifi_fresh    = float(parts[9])

            if _t0 is None:
                _t0 = ts_ms
            ts_s = (ts_ms - _t0) / 1000.0

            # ── Fused trail ──
            trail_r.append(fused_r)
            trail_theta.append(np.radians(fused_theta))

            # ── Timeline buffers ──
            ts_buf.append(ts_s)
            fused_r_buf.append(fused_r)
            fused_a_buf.append(fused_theta)
            radar_r_buf.append(radar_r)
            radar_a_buf.append(radar_theta)
            wifi_r_buf.append(wifi_r)

            # ── Freshness ──
            radar_fresh_buf.append(radar_fresh)
            wifi_fresh_buf.append(wifi_fresh)

            # ── Latency ──
            lat_buf.append(lat_us)
            lat_ts_buf.append(ts_s)

            # ── Stats ──
            stats["count"] += 1
            stats["sum_lat"] += lat_us
            if lat_us < stats["min_lat"]: stats["min_lat"] = lat_us
            if lat_us > stats["max_lat"]: stats["max_lat"] = lat_us
            stats["last_fused_r"]     = fused_r
            stats["last_fused_theta"] = fused_theta
            stats["last_radar_r"]     = radar_r
            stats["last_radar_theta"] = radar_theta
            stats["last_wifi_r"]      = wifi_r
            stats["last_lat"]         = lat_us
            stats["radar_fresh"]      = radar_fresh
            stats["wifi_fresh"]       = wifi_fresh
            stats["radar_fresh_sum"] += int(radar_fresh)
            stats["wifi_fresh_sum"]  += int(wifi_fresh)

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
        polar_lbl.set_text(f"{trail_r[-1]:.2f}m  {np.degrees(trail_theta[-1]):.0f}°")

        # Raw radar marker (amber diamond)
        rr = stats["last_radar_r"]
        rt = np.radians(stats["last_radar_theta"])
        sc_radar.set_offsets([[rt, rr]])
        sc_radar.set_alpha(0.8 if stats["radar_fresh"] > 0 else 0.2)

        # WiFi range ring: show as a set of dots forming an arc at wifi_r
        wr = stats["last_wifi_r"]
        if wr > 0.01:
            # 5 points along the visible arc
            arc_angles = np.linspace(np.radians(-50), np.radians(50), 7)
            sc_wifi.set_offsets(np.c_[arc_angles, np.full_like(arc_angles, wr)])
            sc_wifi.set_alpha(0.6 if stats["wifi_fresh"] > 0 else 0.15)
        else:
            sc_wifi.set_offsets(np.empty((0, 2)))

    else:
        sc_trail.set_offsets(np.empty((0, 2)))
        sc_head.set_offsets(np.empty((0, 2)))
        sc_radar.set_offsets(np.empty((0, 2)))
        sc_wifi.set_offsets(np.empty((0, 2)))

    # Dynamic polar Y limit
    if n > 0:
        mr = max(list(trail_r) + [stats["last_radar_r"], stats["last_wifi_r"]])
        ax_polar.set_ylim(0, max(mr * 1.4, 2.0))

    # ── Timeline ─────────────────────────────────────────────────
    if len(ts_buf) > 1:
        ta = np.array(ts_buf)
        fr = np.array(fused_r_buf)
        rr = np.array(radar_r_buf)
        wr = np.array(wifi_r_buf)

        ln_fused.set_data(ta, fr)
        ln_radar.set_data(ta, rr)
        ln_wifi.set_data(ta, wr)
        dot_fused.set_offsets([[ta[-1], fr[-1]]])

        t_hi = ta[-1] + 0.5
        t_lo = max(0, ta[-1] - 30)
        ax_time.set_xlim(t_lo, t_hi)

        all_r = np.concatenate([fr, rr, wr])
        ax_time.set_ylim(max(all_r.min() - 0.3, 0), all_r.max() + 0.5)

        txt_fused.set_text(f"{fr[-1]:.3f} m")
        txt_radar_v.set_text(
            f"RAD {rr[-1]:.2f}m {'●' if stats['radar_fresh'] > 0 else '○'}")
        txt_wifi_v.set_text(
            f"WiFi {wr[-1]:.2f}m {'●' if stats['wifi_fresh'] > 0 else '○'}")

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
        txt_lat.set_text(f"{lv[-1]} μs")

    # ── Stats ────────────────────────────────────────────────────
    c = stats["count"]
    st_cnt.set_text(str(c))

    elapsed = time.time() - stats["start"]
    hz = c / elapsed if elapsed > 0 else 0
    st_hz.set_text(f"{hz:.0f} Hz")

    avg = stats["sum_lat"] / c if c > 0 else 0
    st_avg.set_text(f"{avg:.0f} μs")

    m, s = divmod(int(elapsed), 60)
    st_up.set_text(f"{m}m {s:02d}s" if m else f"{s}s")

    if c > 0:
        st_mm.set_text(f"{stats['min_lat']}/{stats['max_lat']}")

        # Fused position
        st_pos.set_text(
            f"{stats['last_fused_r']:.2f}m {stats['last_fused_theta']:.0f}°")

        # Radar badge
        if stats["radar_fresh"] > 0:
            st_radar_badge.set_text("● LIVE")
            st_radar_badge.set_color(FRESH_GREEN)
        else:
            st_radar_badge.set_text("○ STALE")
            st_radar_badge.set_color(STALE_RED)
        st_radar_val.set_text(
            f"{stats['last_radar_r']:.2f}m  {stats['last_radar_theta']:.0f}°")

        # WiFi badge
        if stats["wifi_fresh"] > 0:
            st_wifi_badge.set_text("● LIVE")
            st_wifi_badge.set_color(FRESH_GREEN)
        else:
            st_wifi_badge.set_text("○ STALE")
            st_wifi_badge.set_color(STALE_RED)
        st_wifi_val.set_text(f"{stats['last_wifi_r']:.2f}m")

        # Trust bars (rolling freshness percentage over buffer)
        r_pct = (stats["radar_fresh_sum"] / c) if c > 0 else 0
        w_pct = (stats["wifi_fresh_sum"] / c) if c > 0 else 0

        trust_radar_bar.set_width(_bar_w * r_pct)
        trust_wifi_bar.set_width(_bar_w * w_pct)
        st_radar_pct.set_text(f"{r_pct*100:.0f}%")
        st_wifi_pct.set_text(f"{w_pct*100:.0f}%")

    return (sc_trail, sc_head, sc_radar, sc_wifi,
            ln_fused, ln_radar, ln_wifi, ln_lat)


ani = animation.FuncAnimation(fig, update, interval=1000 // FPS,
                               blit=False, cache_frame_data=False)

print("+" + "=" * 58 + "+")
print("|  ISAC ML Fusion Dashboard v2 -- Live from COM13          |")
print("|  Verbose telemetry: FUSED + RADAR + WiFi FTM             |")
print("|  Close the window or Ctrl+C to exit.                     |")
print("+" + "=" * 58 + "+")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nExiting.")
finally:
    ser.close()
    print("Serial closed.")
