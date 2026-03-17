#!/usr/bin/env python3
"""
prepare_features.py
───────────────────
Processes a sensor-fusion dataset (radar + WiFi-FTM) into windowed features
suitable for training a static indoor-localization MLP.

Author : auto-generated
Date   : 2026-03-17
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt

# ───────────────────────── USER-EDITABLE CONSTANTS ──────────────────────── #
R_MIN_M       = 0.0
R_MAX_M       = 6.0          # maximum range of the room in metres
THETA_MIN_DEG = -60.0
THETA_MAX_DEG = 60.0
WINDOW_SIZE   = 5
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
RANDOM_SEED   = 42
CSV_PATH      = "polar_labeled_dataset.csv"
# ──────────────────────────────────────────────────────────────────────────── #


# ====================================================================== #
#  Step 1 — Load and parse                                                #
# ====================================================================== #
def load_and_parse(csv_path: str) -> pd.DataFrame:
    """Read the CSV, coerce NAN strings to np.nan, drop relative_ms."""
    df = pd.read_csv(csv_path, dtype={"sensor": str})

    # Coerce string "NAN" to real NaN for measurement columns
    df["r_m"]       = pd.to_numeric(df["r_m"],       errors="coerce")
    df["theta_deg"] = pd.to_numeric(df["theta_deg"], errors="coerce")

    # Drop relative_ms — it must not be used
    if "relative_ms" in df.columns:
        df.drop(columns=["relative_ms"], inplace=True)

    # Verify ground-truth columns have zero NaN values
    for col in ("gt_r_m", "gt_theta_deg"):
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"ERROR: column '{col}' contains {n_nan} NaN value(s). Aborting.")
            sys.exit(1)

    return df


# ====================================================================== #
#  Step 2 — Process each segment with forward-fill                        #
# ====================================================================== #
def process_segments(df: pd.DataFrame):
    """
    Walk every row in each segment, maintaining forward-fill state for both
    sensors.  Returns a list of raw feature rows (5,), a list of labels (2,),
    and per-segment statistics.
    """
    all_features = []          # list of (5,) arrays
    all_labels   = []          # list of (2,) arrays
    segment_info = []          # per-segment stats dict
    clip_events  = []          # (seg_id, row_idx, feat_name, raw_val)

    for seg_id, seg_df in df.groupby("segment_id", sort=True):
        seg_df = seg_df.reset_index(drop=True)

        # Initialise forward-fill state for this segment
        last_valid_radar_r     = R_MAX_M
        last_valid_radar_theta = 0.0
        last_valid_wifi_r      = R_MAX_M

        seg_features = []
        seg_labels   = []

        # Stats accumulators
        radar_total = 0
        radar_valid = 0
        radar_nan   = 0
        wifi_total  = 0
        wifi_valid  = 0
        wifi_nan    = 0

        for row_idx, row in seg_df.iterrows():
            sensor   = row["sensor"]
            r_val    = row["r_m"]
            t_val    = row["theta_deg"]
            gt_r     = row["gt_r_m"]
            gt_theta = row["gt_theta_deg"]

            if sensor == "R":
                radar_total += 1
                if not np.isnan(r_val) and not np.isnan(t_val):
                    radar_r     = r_val
                    radar_theta = t_val
                    radar_fresh = 1.0
                    last_valid_radar_r     = r_val
                    last_valid_radar_theta = t_val
                    radar_valid += 1
                else:
                    radar_r     = last_valid_radar_r
                    radar_theta = last_valid_radar_theta
                    radar_fresh = 0.0
                    radar_nan  += 1

                f1 = radar_r
                f2 = radar_theta
                f3 = radar_fresh
                f4 = last_valid_wifi_r
                f5 = 0.0                    # wifi not fresh on a radar row

            elif sensor == "W":
                wifi_total += 1
                if not np.isnan(r_val):
                    wifi_r     = r_val
                    wifi_fresh = 1.0
                    last_valid_wifi_r = r_val
                    wifi_valid += 1
                else:
                    wifi_r     = last_valid_wifi_r
                    wifi_fresh = 0.0
                    wifi_nan  += 1

                f1 = last_valid_radar_r
                f2 = last_valid_radar_theta
                f3 = 0.0                    # radar not fresh on a wifi row
                f4 = wifi_r
                f5 = wifi_fresh
            else:
                print(f"WARNING: unknown sensor '{sensor}' in seg {seg_id} row {row_idx}")
                continue

            seg_features.append([f1, f2, f3, f4, f5])
            seg_labels.append([gt_r, gt_theta])

        all_features.extend(seg_features)
        all_labels.extend(seg_labels)

        segment_info.append({
            "segment_id":       seg_id,
            "gt_r_m":           seg_df["gt_r_m"].iloc[0],
            "gt_theta_deg":     seg_df["gt_theta_deg"].iloc[0],
            "total_rows":       len(seg_df),
            "radar_total":      radar_total,
            "radar_valid":      radar_valid,
            "radar_nan":        radar_nan,
            "wifi_total":       wifi_total,
            "wifi_valid":       wifi_valid,
            "wifi_nan":         wifi_nan,
        })

    features = np.array(all_features, dtype=np.float64)
    labels   = np.array(all_labels,   dtype=np.float64)
    return features, labels, segment_info


# ====================================================================== #
#  Step 3 — Normalize                                                     #
# ====================================================================== #
def normalize(features: np.ndarray, labels: np.ndarray, segment_info, df):
    """Min-max normalise features and labels; clip to [0, 1]."""
    clip_events = []

    r_range     = R_MAX_M - R_MIN_M
    theta_range = THETA_MAX_DEG - THETA_MIN_DEG

    feat_norm = features.copy()
    lab_norm  = labels.copy()

    # f1 — radar_r
    feat_norm[:, 0] = (features[:, 0] - R_MIN_M) / r_range
    # f2 — radar_theta
    feat_norm[:, 1] = (features[:, 1] - THETA_MIN_DEG) / theta_range
    # f3 — radar_fresh (already 0/1)
    # f4 — wifi_r
    feat_norm[:, 3] = (features[:, 3] - R_MIN_M) / r_range
    # f5 — wifi_fresh (already 0/1)

    # labels
    lab_norm[:, 0] = (labels[:, 0] - R_MIN_M) / r_range
    lab_norm[:, 1] = (labels[:, 1] - THETA_MIN_DEG) / theta_range

    # Build a cumulative row offset per segment for clip-event reporting
    seg_offsets = {}
    offset = 0
    for info in segment_info:
        seg_offsets[info["segment_id"]] = offset
        offset += info["total_rows"]

    feat_names = ["radar_r_norm", "radar_theta_norm", "radar_fresh",
                  "wifi_r_norm", "wifi_fresh"]
    lab_names  = ["gt_r_norm", "gt_theta_norm"]

    # Clip features
    for col_idx, fname in enumerate(feat_names):
        vals = feat_norm[:, col_idx]
        mask = (vals < 0.0) | (vals > 1.0)
        if mask.any():
            idxs = np.where(mask)[0]
            for i in idxs:
                # Determine segment_id for this global row
                seg_id_found = None
                for info in segment_info:
                    s = seg_offsets[info["segment_id"]]
                    e = s + info["total_rows"]
                    if s <= i < e:
                        seg_id_found = info["segment_id"]
                        local_row = i - s
                        break
                clip_events.append((seg_id_found, local_row, fname, float(vals[i])))
            feat_norm[:, col_idx] = np.clip(vals, 0.0, 1.0)

    # Clip labels
    for col_idx, lname in enumerate(lab_names):
        vals = lab_norm[:, col_idx]
        mask = (vals < 0.0) | (vals > 1.0)
        if mask.any():
            idxs = np.where(mask)[0]
            for i in idxs:
                seg_id_found = None
                for info in segment_info:
                    s = seg_offsets[info["segment_id"]]
                    e = s + info["total_rows"]
                    if s <= i < e:
                        seg_id_found = info["segment_id"]
                        local_row = i - s
                        break
                clip_events.append((seg_id_found, local_row, lname, float(vals[i])))
            lab_norm[:, col_idx] = np.clip(vals, 0.0, 1.0)

    return feat_norm, lab_norm, clip_events


# ====================================================================== #
#  Step 4 — Build sliding windows per segment                             #
# ====================================================================== #
def build_windows(feat_norm, lab_norm, segment_info):
    """
    Create (WINDOW_SIZE, 5) windows per segment; label = last row's label.
    Returns X_windows, y_windows, n_discarded.
    """
    X_windows = []
    y_windows = []
    n_discarded = 0

    offset = 0
    for info in segment_info:
        n = info["total_rows"]
        seg_feat = feat_norm[offset : offset + n]
        seg_lab  = lab_norm[offset : offset + n]

        if n < WINDOW_SIZE:
            n_discarded += n
            offset += n
            continue

        n_discarded += WINDOW_SIZE - 1   # first (W-1) rows can't start a window

        for i in range(n - WINDOW_SIZE + 1):
            X_windows.append(seg_feat[i : i + WINDOW_SIZE])
            y_windows.append(seg_lab[i + WINDOW_SIZE - 1])

        offset += n

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_windows, dtype=np.float32)
    return X, y, n_discarded


# ====================================================================== #
#  Step 5 — Split chronologically within each segment                     #
# ====================================================================== #
def split_by_segment(X_all, y_all, segment_info):
    """
    Split windows from EACH segment chronologically:
    First 70% -> train
    Next 15% -> val
    Last 15% -> test
    This ensures all distances are represented in train/val/test.
    Returns dict with keys train / val / test, each holding (X, y, seg_ids_represented).
    """
    train_xs, train_ys = [], []
    val_xs, val_ys     = [], []
    test_xs, test_ys   = [], []

    offset = 0
    seg_ids = []
    
    for info in segment_info:
        sid = info["segment_id"]
        seg_ids.append(sid)
        
        n_rows = info["total_rows"]
        n_win = max(0, n_rows - WINDOW_SIZE + 1)
        
        if n_win == 0:
            continue
            
        # Chronological split within this segment
        n_train = int(np.floor(n_win * TRAIN_RATIO))
        n_val   = int(np.floor(n_win * VAL_RATIO))
        
        seg_X = X_all[offset : offset + n_win]
        seg_y = y_all[offset : offset + n_win]
        
        if n_train > 0:
            train_xs.append(seg_X[:n_train])
            train_ys.append(seg_y[:n_train])
        if n_val > 0:
            val_xs.append(seg_X[n_train : n_train + n_val])
            val_ys.append(seg_y[n_train : n_train + n_val])
        if n_train + n_val < n_win:
            test_xs.append(seg_X[n_train + n_val :])
            test_ys.append(seg_y[n_train + n_val :])
        
        offset += n_win

    def concat(xs, ys):
        if xs:
            return np.concatenate(xs), np.concatenate(ys)
        return np.empty((0, WINDOW_SIZE, 5), dtype=np.float32), \
               np.empty((0, 2), dtype=np.float32)

    X_train, y_train = concat(train_xs, train_ys)
    X_val,   y_val   = concat(val_xs, val_ys)
    X_test,  y_test  = concat(test_xs, test_ys)

    return {
        "train": (X_train, y_train, seg_ids),
        "val":   (X_val,   y_val,   seg_ids),
        "test":  (X_test,  y_test,  seg_ids),
    }


# ====================================================================== #
#  Step 6 — Save outputs                                                  #
# ====================================================================== #
def save_outputs(splits):
    """Save .npy files and norm_params.json."""
    for name in ("train", "val", "test"):
        X, y, _ = splits[name]
        np.save(f"X_{name}.npy", X)
        np.save(f"y_{name}.npy", y)

    params = {
        "R_MIN_M":       R_MIN_M,
        "R_MAX_M":       R_MAX_M,
        "THETA_MIN_DEG": THETA_MIN_DEG,
        "THETA_MAX_DEG": THETA_MAX_DEG,
        "WINDOW_SIZE":   WINDOW_SIZE,
        "NUM_FEATURES":  5,
        "features":      ["radar_r_norm", "radar_theta_norm", "radar_fresh",
                          "wifi_r_norm", "wifi_fresh"],
        "labels":        ["gt_r_norm", "gt_theta_norm"],
    }
    with open("norm_params.json", "w") as f:
        json.dump(params, f, indent=4)


# ====================================================================== #
#  Step 7 — Print comprehensive summary                                   #
# ====================================================================== #
def print_summary(df, segment_info, clip_events, n_discarded,
                  total_windows, splits):
    total_rows = len(df)
    n_seg = len(segment_info)
    rows_per_seg = [info["total_rows"] for info in segment_info]

    print("\n" + "=" * 60)
    print("=== DATA LOADING ===")
    print("=" * 60)
    print(f"Total rows loaded:  {total_rows}")
    print(f"Total segments:     {n_seg}")
    print(f"Rows per segment (min / max / mean): "
          f"{min(rows_per_seg)} / {max(rows_per_seg)} / "
          f"{np.mean(rows_per_seg):.1f}")

    # ── sensor statistics ──
    total_radar   = sum(i["radar_total"] for i in segment_info)
    total_wifi    = sum(i["wifi_total"]  for i in segment_info)
    radar_valid   = sum(i["radar_valid"] for i in segment_info)
    radar_nan     = sum(i["radar_nan"]   for i in segment_info)
    wifi_valid    = sum(i["wifi_valid"]  for i in segment_info)
    wifi_nan      = sum(i["wifi_nan"]    for i in segment_info)

    def pct(a, b):
        return f"{100*a/b:.1f}" if b else "N/A"

    print("\n" + "=" * 60)
    print("=== SENSOR STATISTICS ===")
    print("=" * 60)
    print(f"Total radar events:                {total_radar}")
    print(f"Total wifi events:                 {total_wifi}")
    print(f"Radar valid (non-NaN) events:      {radar_valid} ({pct(radar_valid, total_radar)}% of radar events)")
    print(f"Radar NaN (dropout) events:        {radar_nan} ({pct(radar_nan, total_radar)}% of radar events)")
    print(f"WiFi valid events:                 {wifi_valid} ({pct(wifi_valid, total_wifi)}% of wifi events)")
    print(f"WiFi NaN events:                   {wifi_nan} ({pct(wifi_nan, total_wifi)}% of wifi events)")

    # ── per-segment table ──
    print("\n" + "=" * 60)
    print("=== PER-SEGMENT TABLE ===")
    print("=" * 60)
    header = (f"{'seg':>4s}  {'gt_r':>6s}  {'gt_th':>6s}  {'rows':>5s}  "
              f"{'R_ok':>5s}  {'R_nan':>5s}  {'R_%':>6s}  "
              f"{'W_tot':>5s}  {'W_%':>6s}  {'flag':>6s}")
    print(header)
    print("-" * len(header))
    for info in segment_info:
        r_pct = 100 * info["radar_valid"] / info["radar_total"] if info["radar_total"] else 0
        w_pct = 100 * info["wifi_valid"]  / info["wifi_total"]  if info["wifi_total"]  else 0

        # Suspicious flag heuristic
        gt_theta = info["gt_theta_deg"]
        gt_r     = info["gt_r_m"]
        suspicious = (r_pct < 30
                      and abs(gt_theta) <= 55
                      and gt_r < R_MAX_M)
        flag = "! WARN" if suspicious else ""

        print(f"{info['segment_id']:4d}  {info['gt_r_m']:6.2f}  {info['gt_theta_deg']:6.1f}  "
              f"{info['total_rows']:5d}  {info['radar_valid']:5d}  {info['radar_nan']:5d}  "
              f"{r_pct:5.1f}%  {info['wifi_total']:5d}  {w_pct:5.1f}%  {flag}")
        if suspicious:
            print(f"       WARNING: segment {info['segment_id']} has radar_valid_pct "
                  f"= {r_pct:.1f}% < 30% — this is suspicious!")

    # ── clipping ──
    print("\n" + "=" * 60)
    print("=== NORMALIZATION CLIPS ===")
    print("=" * 60)
    print(f"Total clipping events: {len(clip_events)}")
    if clip_events:
        print("Top 5 clip events:")
        for seg_id, row_idx, feat, raw in clip_events[:5]:
            print(f"  seg={seg_id}, row={row_idx}, feature={feat}, raw_norm_value={raw:.6f}")

    # ── windowing ──
    print("\n" + "=" * 60)
    print("=== WINDOWING ===")
    print("=" * 60)
    print(f"Total windows created:                            {total_windows}")
    print(f"Windows discarded (incomplete at segment starts): {n_discarded}")

    # ── split ──
    X_train, y_train, train_ids = splits["train"]
    X_val,   y_val,   val_ids   = splits["val"]
    X_test,  y_test,  test_ids  = splits["test"]

    print("\n" + "=" * 60)
    print("=== TRAIN / VAL / TEST SPLIT ===")
    print("=" * 60)
    print(f"Training segments:   {len(train_ids):3d} --> {len(X_train)} windows")
    print(f"Validation segments: {len(val_ids):3d} --> {len(X_val)} windows")
    print(f"Testing segments:    {len(test_ids):3d} --> {len(X_test)} windows")
    print(f"\nTraining segment IDs:   {[int(x) for x in train_ids]}")
    print(f"Validation segment IDs: {[int(x) for x in val_ids]}")
    print(f"Testing segment IDs:    {[int(x) for x in test_ids]}")

    # ── shapes ──
    print("\n" + "=" * 60)
    print("=== ARRAY SHAPES ===")
    print("=" * 60)
    for name, X, y in [("train", X_train, y_train),
                        ("val",   X_val,   y_val),
                        ("test",  X_test,  y_test)]:
        print(f"X_{name}: {X.shape}  dtype={X.dtype}")
        print(f"y_{name}: {y.shape}  dtype={y.dtype}")

    # ── sanity ──
    all_X = np.concatenate([X_train, X_val, X_test]) if (len(X_train) + len(X_val) + len(X_test)) > 0 else np.empty((0,))
    all_y = np.concatenate([y_train, y_val, y_test]) if (len(y_train) + len(y_val) + len(y_test)) > 0 else np.empty((0,))

    print("\n" + "=" * 60)
    print("=== SANITY CHECKS ===")
    print("=" * 60)
    if all_X.size:
        print(f"X value range: [{all_X.min():.6f}, {all_X.max():.6f}]  (should be within [0.0, 1.0])")
    else:
        print("X is empty!")
    if all_y.size:
        print(f"y value range: [{all_y.min():.6f}, {all_y.max():.6f}]  (should be within [0.0, 1.0])")
    else:
        print("y is empty!")
    print(f"Any NaN in X: {bool(np.isnan(all_X).any())}  (MUST be False)")
    print(f"Any NaN in y: {bool(np.isnan(all_y).any())}  (MUST be False)")
    print()


# ====================================================================== #
#  Step 8 — Diagnostic plots                                              #
# ====================================================================== #
def generate_plots(feat_norm, lab_norm, segment_info, splits):
    """Generate and save the four required diagnostic PNGs."""
    diag_dir = "diagnostics"
    os.makedirs(diag_dir, exist_ok=True)

    feat_names = ["radar_r_norm", "radar_theta_norm", "radar_fresh",
                  "wifi_r_norm", "wifi_fresh"]
    lab_names  = ["gt_r_norm", "gt_theta_norm"]

    # ── Plot 1: feature distributions ──
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (ax, name) in enumerate(zip(axes, feat_names)):
        ax.hist(feat_norm[:, i], bins=50, color="steelblue", edgecolor="black",
                alpha=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.suptitle("Feature Distributions (normalised)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "feature_distributions.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: label distributions ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (ax, name) in enumerate(zip(axes, lab_names)):
        ax.hist(lab_norm[:, i], bins=30, color="coral", edgecolor="black",
                alpha=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.suptitle("Label Distributions (normalised)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "label_distributions.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: segment radar validity ──
    seg_ids  = [info["segment_id"] for info in segment_info]
    r_pcts   = [100 * info["radar_valid"] / info["radar_total"]
                if info["radar_total"] else 0
                for info in segment_info]
    colours  = ["red" if p < 30 else "green" for p in r_pcts]

    fig, ax = plt.subplots(figsize=(max(8, len(seg_ids)), 5))
    ax.bar([str(s) for s in seg_ids], r_pcts, color=colours, edgecolor="black")
    ax.set_xlabel("segment_id")
    ax.set_ylabel("radar_valid %")
    ax.set_title("Radar Validity per Segment")
    ax.axhline(30, color="orange", linestyle="--", linewidth=1, label="30% threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "segment_radar_validity.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: sample windows (heatmaps) ──
    X_train, y_train, _ = splits["train"]
    if len(X_train) >= 3:
        rng = np.random.RandomState(RANDOM_SEED)
        idxs = rng.choice(len(X_train), size=3, replace=False)

        r_range     = R_MAX_M - R_MIN_M
        theta_range = THETA_MAX_DEG - THETA_MIN_DEG

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, idx in zip(axes, idxs):
            win = X_train[idx]            # (5, 5)
            lab = y_train[idx]            # (2,)
            gt_r     = lab[0] * r_range + R_MIN_M
            gt_theta = lab[1] * theta_range + THETA_MIN_DEG

            im = ax.imshow(win, aspect="auto", cmap="viridis")
            ax.set_title(f"Window {idx}\ngt_r={gt_r:.2f} m, gt_th={gt_theta:.1f} deg",
                         fontsize=10)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Time step")
            ax.set_xticks(range(5))
            ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Sample Training Windows", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(diag_dir, "sample_windows.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"Diagnostic plots saved to '{diag_dir}/'")


# ====================================================================== #
#  Main                                                                   #
# ====================================================================== #
def main():
    # Step 1
    print("Loading data ...")
    df = load_and_parse(CSV_PATH)
    total_rows = len(df)
    print(f"  {total_rows} rows loaded.\n")

    # Step 2
    print("Processing segments (forward-fill) ...")
    features, labels, segment_info = process_segments(df)
    print(f"  {len(features)} feature vectors built.\n")

    # Step 3
    print("Normalising ...")
    feat_norm, lab_norm, clip_events = normalize(features, labels,
                                                  segment_info, df)
    for seg_id, row_idx, feat_name, raw_val in clip_events:
        print(f"  CLIP WARNING: seg={seg_id}, row={row_idx}, "
              f"feature={feat_name}, normalised_value={raw_val:.6f}")
    print(f"  {len(clip_events)} clip event(s).\n")

    # Step 4
    print("Building sliding windows ...")
    X_all, y_all, n_discarded = build_windows(feat_norm, lab_norm,
                                               segment_info)
    total_windows = len(X_all)
    print(f"  {total_windows} windows created, {n_discarded} rows discarded.\n")

    # Step 5
    print("Splitting chronologically within each segment ...")
    splits = split_by_segment(X_all, y_all, segment_info)
    print("  Done.\n")

    # Step 6
    print("Saving outputs ...")
    save_outputs(splits)
    print("  Saved .npy files and norm_params.json.\n")

    # Step 7
    print_summary(df, segment_info, clip_events, n_discarded,
                  total_windows, splits)

    # Step 8
    print("Generating diagnostic plots ...")
    generate_plots(feat_norm, lab_norm, segment_info, splits)
    print("\n[OK] All done.")


if __name__ == "__main__":
    main()
