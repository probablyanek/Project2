#ifndef POLAR_NORM_PARAMS_H
#define POLAR_NORM_PARAMS_H

// Normalization: normalized = (value - MIN) / (MAX - MIN)
// Denormalization: value = normalized * (MAX - MIN) + MIN

#define R_MIN_M       0.0f
#define R_MAX_M       6.0f
#define THETA_MIN_DEG -60.0f
#define THETA_MAX_DEG 60.0f

#define WINDOW_SIZE   5
#define NUM_FEATURES  5

// Feature order (per time-step in the window):
//   [0] radar_r_norm      - normalised radar range
//   [1] radar_theta_norm  - normalised radar angle
//   [2] radar_fresh       - 1.0 if radar reading is fresh, 0.0 if stale
//   [3] wifi_r_norm       - normalised WiFi-FTM range
//   [4] wifi_fresh        - 1.0 if WiFi reading is fresh, 0.0 if stale

// MLP input length (flattened window)
#define MLP_INPUT_LEN (WINDOW_SIZE * NUM_FEATURES)  // 25

// Output: 2 floats [r_norm, theta_norm]
// Denormalize: r_m = r_norm * (R_MAX_M - R_MIN_M) + R_MIN_M
//              theta_deg = theta_norm * (THETA_MAX_DEG - THETA_MIN_DEG) + THETA_MIN_DEG

#endif  // POLAR_NORM_PARAMS_H
