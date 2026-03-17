#!/usr/bin/env python3
"""
train_polar_mlp_f32.py
-----------------------
Train a small MLP for static indoor localization using fused radar + WiFi
features prepared by prepare_features.py.

Outputs:
    polar_mlp_f32.keras          - Keras native format
    polar_mlp_f32_saved/         - TF SavedModel format
    diagnostics/training_*.png   - Diagnostic plots
"""

import os
import sys
import json

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════ CONFIGURATION ═══════════════════════════════ #
EPOCHS       = 200
BATCH_SIZE   = 256
LR           = 0.001
PATIENCE_ES  = 15          # EarlyStopping patience
PATIENCE_LR  = 7           # ReduceLROnPlateau patience
LR_FACTOR    = 0.5
DIAG_DIR     = "diagnostics"
# ═════════════════════════════════════════════════════════════════════════ #


def load_data():
    """Load .npy files and norm_params.json."""
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val   = np.load("X_val.npy")
    y_val   = np.load("y_val.npy")
    X_test  = np.load("X_test.npy")
    y_test  = np.load("y_test.npy")

    with open("norm_params.json") as f:
        norm = json.load(f)

    print("=== LOADED DATA ===")
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, norm


def flatten(X):
    """Flatten (N, W, F) to (N, W*F)."""
    return X.reshape(X.shape[0], -1)


def build_model(input_dim):
    """Build the small MLP: input_dim -> 48 -> 24 -> 2."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(2,  activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
    )
    return model


def denormalize(pred, norm):
    """
    Denormalize predictions from [0,1] back to physical units.
    pred shape: (N, 2)  columns: [r_norm, theta_norm]
    Returns: r_m, theta_deg
    """
    r_min = norm["R_MIN_M"]
    r_max = norm["R_MAX_M"]
    t_min = norm["THETA_MIN_DEG"]
    t_max = norm["THETA_MAX_DEG"]

    r_m       = pred[:, 0] * (r_max - r_min) + r_min
    theta_deg = pred[:, 1] * (t_max - t_min) + t_min
    return r_m, theta_deg


def polar_to_cart(r, theta_deg):
    """Convert polar (r, theta_deg) to Cartesian (x, y)."""
    theta_rad = np.deg2rad(theta_deg)
    x = r * np.sin(theta_rad)
    y = r * np.cos(theta_rad)
    return x, y


def compute_metrics(pred, gt, norm, label=""):
    """Compute and print MAE_r, MAE_theta, Euclidean error."""
    pred_r, pred_theta = denormalize(pred, norm)
    gt_r,   gt_theta   = denormalize(gt,   norm)

    mae_r     = np.mean(np.abs(pred_r     - gt_r))
    mae_theta = np.mean(np.abs(pred_theta - gt_theta))

    pred_x, pred_y = polar_to_cart(pred_r, pred_theta)
    gt_x,   gt_y   = polar_to_cart(gt_r,   gt_theta)
    euclid = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    mean_euclid = np.mean(euclid)
    p95_euclid  = np.percentile(euclid, 95)

    print(f"\n--- {label} Metrics ---")
    print(f"  MAE  r         : {mae_r:.4f} m")
    print(f"  MAE  theta     : {mae_theta:.4f} deg")
    print(f"  Euclidean mean : {mean_euclid:.4f} m")
    print(f"  Euclidean 95%  : {p95_euclid:.4f} m")

    return {
        "pred_r": pred_r, "pred_theta": pred_theta,
        "gt_r": gt_r, "gt_theta": gt_theta,
        "mae_r": mae_r, "mae_theta": mae_theta,
        "euclid": euclid, "mean_euclid": mean_euclid,
        "p95_euclid": p95_euclid,
    }


# ──────────────────── Plot helpers ──────────────────────────────────────── #

def plot_training_curves(history):
    """(a) Training loss vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.history["loss"],     label="Train MSE", linewidth=1.5)
    ax.plot(history.history["val_loss"], label="Val MSE",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised)")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_segment_predictions(X_test_flat, y_test, model, norm):
    """
    (b) For each test segment, plot predicted vs gt (r, theta) over time,
    with radar_fresh / wifi_fresh colour bars underneath.

    Because we flattened the windows, we reconstruct which rows
    belong to which segment by using the gt labels (constant per segment).
    """
    pred_test = model.predict(X_test_flat, verbose=0)
    pred_r, pred_theta = denormalize(pred_test, norm)
    gt_r,   gt_theta   = denormalize(y_test,    norm)

    # Identify test segments by unique (gt_r, gt_theta) pairs
    seg_labels = list(zip(np.round(gt_r, 4), np.round(gt_theta, 4)))
    unique_segs = sorted(set(seg_labels))

    window_size = norm["WINDOW_SIZE"]
    num_feat    = norm["NUM_FEATURES"]

    n_segs = len(unique_segs)
    fig, axes = plt.subplots(n_segs, 1, figsize=(14, 4 * n_segs), squeeze=False)

    for idx, (sr, st) in enumerate(unique_segs):
        mask = [(r, t) == (sr, st) for r, t in seg_labels]
        mask = np.array(mask)

        ax = axes[idx, 0]

        p_r = pred_r[mask]
        p_t = pred_theta[mask]
        g_r = gt_r[mask]
        g_t = gt_theta[mask]

        timesteps = np.arange(len(p_r))

        # Reconstruct the last row of each window to get radar_fresh and wifi_fresh
        # X_test_flat shape: (N, W*F).  For the last row, features are at
        # positions [(W-1)*F .. W*F-1]
        seg_X = X_test_flat[mask]
        last_row_start = (window_size - 1) * num_feat
        radar_fresh = seg_X[:, last_row_start + 2]   # f3 = radar_fresh
        wifi_fresh  = seg_X[:, last_row_start + 4]   # f5 = wifi_fresh

        # Plot r predictions
        ax.plot(timesteps, p_r, "b-", alpha=0.7, label="Pred r (m)", linewidth=1)
        ax.axhline(sr, color="b", linestyle="--", alpha=0.5, label=f"GT r = {sr:.1f} m")

        ax2 = ax.twinx()
        ax2.plot(timesteps, p_t, "r-", alpha=0.7, label="Pred theta (deg)", linewidth=1)
        ax2.axhline(st, color="r", linestyle="--", alpha=0.5, label=f"GT theta = {st:.1f} deg")
        ax2.set_ylabel("Theta (deg)", color="r")

        # Colour bar for sensor activity at the bottom
        for t_i in timesteps:
            if radar_fresh[t_i] > 0.5:
                ax.axvspan(t_i - 0.4, t_i + 0.4, ymin=0, ymax=0.03,
                           color="green", alpha=0.6)
            if wifi_fresh[t_i] > 0.5:
                ax.axvspan(t_i - 0.4, t_i + 0.4, ymin=0.03, ymax=0.06,
                           color="orange", alpha=0.6)

        ax.set_xlabel("Window index")
        ax.set_ylabel("Range (m)", color="b")
        ax.set_title(f"Segment gt=({sr:.1f} m, {st:.1f} deg)  |  "
                     f"green=radar, orange=wifi")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "training_segment_predictions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_r(m):
    """(c) Scatter: predicted_r vs gt_r."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m["gt_r"], m["pred_r"], s=8, alpha=0.4, edgecolors="none")
    lims = [min(m["gt_r"].min(), m["pred_r"].min()) - 0.2,
            max(m["gt_r"].max(), m["pred_r"].max()) + 0.2]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
    ax.set_xlabel("GT r (m)")
    ax.set_ylabel("Predicted r (m)")
    ax.set_title("Predicted vs Ground-Truth Range")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "training_scatter_r.png"), dpi=150)
    plt.close(fig)


def plot_scatter_theta(m):
    """(d) Scatter: predicted_theta vs gt_theta."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m["gt_theta"], m["pred_theta"], s=8, alpha=0.4, edgecolors="none")
    lims = [min(m["gt_theta"].min(), m["pred_theta"].min()) - 2,
            max(m["gt_theta"].max(), m["pred_theta"].max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
    ax.set_xlabel("GT theta (deg)")
    ax.set_ylabel("Predicted theta (deg)")
    ax.set_title("Predicted vs Ground-Truth Angle")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "training_scatter_theta.png"), dpi=150)
    plt.close(fig)


def plot_euclidean_hist(m):
    """(e) Euclidean error histogram with mean & 95th pctl lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(m["euclid"], bins=60, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(m["mean_euclid"], color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {m['mean_euclid']:.3f} m")
    ax.axvline(m["p95_euclid"],  color="orange", linestyle="--", linewidth=1.5,
               label=f"95th pctl = {m['p95_euclid']:.3f} m")
    ax.set_xlabel("Euclidean Error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Euclidean Error Distribution (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "training_euclidean_hist.png"), dpi=150)
    plt.close(fig)


def plot_error_by_distance(m):
    """(f) Error breakdown table by distance band, rendered as a table figure."""
    bands = [(0, 1), (1, 2), (2, 3), (3, 5), (5, np.inf)]
    band_names = ["0-1 m", "1-2 m", "2-3 m", "3-5 m", "5+ m"]

    rows = []
    for (lo, hi), name in zip(bands, band_names):
        if hi == 1:
            mask = (m["gt_r"] >= lo) & (m["gt_r"] <= hi)  # Include 1.0 exactly
        else:
            mask = (m["gt_r"] > lo) & (m["gt_r"] <= hi)
        n = mask.sum()
        if n == 0:
            rows.append([name, 0, "-", "-", "-"])
            continue
        mae_r     = np.mean(np.abs(m["pred_r"][mask]     - m["gt_r"][mask]))
        mae_theta = np.mean(np.abs(m["pred_theta"][mask]  - m["gt_theta"][mask]))
        euc       = np.mean(m["euclid"][mask])
        rows.append([name, n, f"{mae_r:.4f}", f"{mae_theta:.4f}", f"{euc:.4f}"])

    # Console table
    print("\n--- Error by Distance Band (Test) ---")
    header = f"{'Band':>8s}  {'N':>6s}  {'MAE_r(m)':>10s}  {'MAE_th(d)':>10s}  {'Euc(m)':>10s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row[0]:>8s}  {row[1]:>6}  {row[2]:>10s}  {row[3]:>10s}  {row[4]:>10s}")

    # Figure table
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    col_labels = ["Band", "N", "MAE r (m)", "MAE theta (deg)", "Euclidean (m)"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title("Error Breakdown by Distance Band (Test)", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "training_error_by_distance.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar_availability(m, X_test_flat, norm):
    """
    (g) Error breakdown: 'radar available' vs 'radar occluded'.
    A window is considered 'radar available' if the last row has radar_fresh==1.
    """
    window_size = norm["WINDOW_SIZE"]
    num_feat    = norm["NUM_FEATURES"]
    last_row_start = (window_size - 1) * num_feat

    radar_fresh = X_test_flat[:, last_row_start + 2]  # f3
    radar_avail = radar_fresh > 0.5

    labels_list = ["Radar Available", "Radar Occluded"]
    masks_list  = [radar_avail, ~radar_avail]

    rows = []
    for lbl, mask in zip(labels_list, masks_list):
        n = mask.sum()
        if n == 0:
            rows.append([lbl, 0, "-", "-", "-"])
            continue
        mae_r     = np.mean(np.abs(m["pred_r"][mask]     - m["gt_r"][mask]))
        mae_theta = np.mean(np.abs(m["pred_theta"][mask]  - m["gt_theta"][mask]))
        euc       = np.mean(m["euclid"][mask])
        rows.append([lbl, n, f"{mae_r:.4f}", f"{mae_theta:.4f}", f"{euc:.4f}"])

    print("\n--- Radar Available vs Occluded (Test) ---")
    header = f"{'Condition':>18s}  {'N':>6s}  {'MAE_r(m)':>10s}  {'MAE_th(d)':>10s}  {'Euc(m)':>10s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row[0]:>18s}  {row[1]:>6}  {row[2]:>10s}  {row[3]:>10s}  {row[4]:>10s}")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    col_labels = ["Condition", "N", "MAE r (m)", "MAE theta (deg)", "Euclidean (m)"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title("Error: Radar Available vs Occluded (Test)", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "training_radar_availability.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════ MAIN ═══════════════════════════════════════ #

def main():
    os.makedirs(DIAG_DIR, exist_ok=True)

    # ── 1. Load ──
    X_train, y_train, X_val, y_val, X_test, y_test, norm = load_data()

    # ── 2. Flatten ──
    X_train_flat = flatten(X_train)
    X_val_flat   = flatten(X_val)
    X_test_flat  = flatten(X_test)
    input_dim    = X_train_flat.shape[1]

    print(f"\nFlattened: {X_train.shape} --> ({X_train_flat.shape[0]}, {input_dim})")
    print(f"Input dim = {input_dim}")

    # ── 3. Build model ──
    model = build_model(input_dim)
    model.summary()

    total_params = model.count_params()
    size_kb = total_params * 4 / 1024   # float32
    print(f"\nTotal parameters: {total_params}  (~{size_kb:.1f} KB in Float32)")

    # ── 4-5. Train ──
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_ES,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=PATIENCE_LR,
            verbose=1,
        ),
    ]

    print("\n=== TRAINING ===")
    history = model.fit(
        X_train_flat, y_train,
        validation_data=(X_val_flat, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # ── 6. Evaluate ──
    print("\n=== EVALUATION ===")
    train_mse = model.evaluate(X_train_flat, y_train, verbose=0)
    val_mse   = model.evaluate(X_val_flat,   y_val,   verbose=0)
    test_mse  = model.evaluate(X_test_flat,  y_test,  verbose=0)
    print(f"Train MSE (norm): {train_mse:.6f}")
    print(f"Val   MSE (norm): {val_mse:.6f}")
    print(f"Test  MSE (norm): {test_mse:.6f}")

    pred_train = model.predict(X_train_flat, verbose=0)
    pred_val   = model.predict(X_val_flat,   verbose=0)
    pred_test  = model.predict(X_test_flat,  verbose=0)

    m_train = compute_metrics(pred_train, y_train, norm, "Train")
    m_val   = compute_metrics(pred_val,   y_val,   norm, "Val")
    m_test  = compute_metrics(pred_test,  y_test,  norm, "Test")

    # ── Plots ──
    print("\nGenerating plots ...")

    # (a) Training curves
    plot_training_curves(history)

    # (b) Segment predictions
    plot_segment_predictions(X_test_flat, y_test, model, norm)

    # (c) Scatter r
    plot_scatter_r(m_test)

    # (d) Scatter theta
    plot_scatter_theta(m_test)

    # (e) Euclidean histogram
    plot_euclidean_hist(m_test)

    # (f) Error by distance band
    plot_error_by_distance(m_test)

    # (g) Radar availability
    plot_radar_availability(m_test, X_test_flat, norm)

    print(f"\nAll plots saved to '{DIAG_DIR}/'")

    # ── 7. Save model ──
    model.save("polar_mlp_f32.keras")
    model.export("polar_mlp_f32_saved")
    print("\nModel saved:")
    print("  polar_mlp_f32.keras")
    print("  polar_mlp_f32_saved/")

    # ── 8. Summary ──
    print("\n=== MODEL SUMMARY ===")
    model.summary()

    print("\n[OK] All done.")


if __name__ == "__main__":
    main()
