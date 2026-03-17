/* ═══════════════════════════════════════════════════════════════════════════
 *  isac_polar_ml  —  Real-Time Sensor Fusion Firmware
 *
 *  ESP32-S3  ·  ESP-IDF v5.5  ·  TFLite Micro Float32 MLP
 *
 *  3-task FreeRTOS architecture:
 *    Task 1  wifi_ftm_task     Core 0  Prio 5  Wi-Fi STA + FTM + ESP-NOW
 *    Task 2  uart_radar_task   Core 1  Prio 5  HLK-LD2450 UART parser
 *    Task 3  ml_inference_task Core 1  Prio 3  Sliding window → MLP → FUSED
 *
 *  Radar + Wi-Fi FTM events flow through a FreeRTOS queue as
 *  sensor_event_t structs.  The ML task consumes them, maintains a
 *  forward-fill sliding window (matching prepare_features.py exactly),
 *  runs inference, and prints FUSED CSV lines to serial.
 * ═══════════════════════════════════════════════════════════════════════════*/

#include <cstdio>
#include <cmath>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "driver/uart.h"
#include "driver/gpio.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_mac.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "polar_model_data.h"
#include "polar_norm_params.h"

/* ───────────────────────── Constants ────────────────────────── */
static const char *TAG = "ISAC_FUSION";

#define SSID        "ISAC_TAG"
#define PASSWORD    "12345678"

#define RADAR_TXD   GPIO_NUM_17
#define RADAR_RXD   GPIO_NUM_16

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define C_LIGHT     299792458.0f
#define QUEUE_LEN   64

/* ───────────────────────── Data types ───────────────────────── */
typedef enum { SENSOR_RADAR = 0, SENSOR_WIFI = 1 } sensor_type_t;

typedef struct {
    sensor_type_t type;
    float         r_m;
    float         theta_deg;
    int64_t       timestamp_ms;
} sensor_event_t;

typedef struct {
    /* Last known valid readings (forward-fill sources) */
    float last_radar_r;
    float last_radar_theta;
    float last_wifi_r;

    /* Sliding window circular buffer */
    float window[WINDOW_SIZE][NUM_FEATURES];
    int   window_head;
    int   window_count;

    /* Last good prediction for sanity checks */
    float last_pred_r;
    float last_pred_theta;
} fusion_state_t;

/* ───────────────────────── Shared queue ─────────────────────── */
static QueueHandle_t sensor_queue = NULL;

/* ───────────────── Normalization helpers ────────────────────── */
static inline float norm_range(float r_m) {
    float v = (r_m - R_MIN_M) / (R_MAX_M - R_MIN_M);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v;
}

static inline float norm_angle(float theta_deg) {
    float v = (theta_deg - THETA_MIN_DEG) / (THETA_MAX_DEG - THETA_MIN_DEG);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v;
}

static inline float denorm_range(float v) {
    return v * (R_MAX_M - R_MIN_M) + R_MIN_M;
}

static inline float denorm_angle(float v) {
    return v * (THETA_MAX_DEG - THETA_MIN_DEG) + THETA_MIN_DEG;
}

/* ═════════════════════════════════════════════════════════════════
 *  TASK 1 — Wi-Fi FTM  (Core 0, Priority 5)
 * ═════════════════════════════════════════════════════════════════*/
static uint8_t responder_mac[6] = {0};
static volatile bool wifi_connected = false;

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi disconnected – reconnecting");
        wifi_connected = false;
        esp_wifi_connect();

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
        ESP_LOGI(TAG, "Connected to ISAC_TAG");
        esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40);

        wifi_ap_record_t ap;
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            memcpy(responder_mac, ap.bssid, 6);
            ESP_LOGI(TAG, "Responder MAC: %02x:%02x:%02x:%02x:%02x:%02x",
                     responder_mac[0], responder_mac[1], responder_mac[2],
                     responder_mac[3], responder_mac[4], responder_mac[5]);
            wifi_connected = true;
        }

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_FTM_REPORT) {
        auto *rpt = (wifi_event_ftm_report_t *)data;
        int64_t ts = esp_timer_get_time() / 1000LL;

        if (rpt->status == FTM_STATUS_SUCCESS &&
            rpt->ftm_report_num_entries > 0 &&
            rpt->ftm_report_data != NULL)
        {
            float min_r = 99999.0f;
            for (int i = 0; i < rpt->ftm_report_num_entries; i++) {
                float r = (float)rpt->ftm_report_data[i].rtt
                          * 1e-12f * C_LIGHT / 2.0f;
                if (r < min_r) min_r = r;
            }

            sensor_event_t evt = {};
            evt.type         = SENSOR_WIFI;
            evt.r_m          = min_r;
            evt.theta_deg    = NAN;
            evt.timestamp_ms = ts;
            xQueueSendToBack(sensor_queue, &evt, 0);
        }

        if (rpt->ftm_report_data) free(rpt->ftm_report_data);
    }
}

static void espnow_recv_cb(const esp_now_recv_info_t *info,
                            const uint8_t *data, int len)
{
    if (len == 4 && memcmp(data, "LIVE", 4) == 0) {
        ESP_LOGI(TAG, "TAG_ALIVE");
    }
}

static void ftm_timer_cb(void *arg)
{
    if (!wifi_connected) return;

    wifi_ftm_initiator_cfg_t cfg = {};
    cfg.channel            = 6;
    cfg.frm_count          = 16;
    cfg.burst_period       = 2;
    cfg.use_get_report_api = false;
    memcpy(cfg.resp_mac, responder_mac, 6);
    esp_wifi_ftm_initiate_session(&cfg);
}

static void wifi_ftm_task(void *arg)
{
    /* --- Networking infra --- */
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t wcfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wcfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    wifi_config_t sta_cfg = {};
    strcpy((char *)sta_cfg.sta.ssid, SSID);
    strcpy((char *)sta_cfg.sta.password, PASSWORD);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* --- ESP-NOW --- */
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_recv_cb(espnow_recv_cb));

    /* --- FTM periodic timer (500 ms) --- */
    const esp_timer_create_args_t tmr = {
        .callback             = &ftm_timer_cb,
        .arg                  = NULL,
        .dispatch_method      = ESP_TIMER_TASK,
        .name                 = "ftm_timer",
        .skip_unhandled_events = true
    };
    esp_timer_handle_t timer;
    ESP_ERROR_CHECK(esp_timer_create(&tmr, &timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(timer, 500000));

    ESP_LOGI(TAG, "wifi_ftm_task running on core %d", xPortGetCoreID());

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

/* ═════════════════════════════════════════════════════════════════
 *  TASK 2 — UART Radar  (Core 1, Priority 5)
 * ═════════════════════════════════════════════════════════════════*/

/** Convert LD2450 raw 16-bit coordinate to signed mm.
 *  Bit 15 = sign flag: 1 → positive, 0 → negative. */
static inline int16_t ld2450_to_signed(uint16_t raw)
{
    int16_t val = raw & 0x7FFF;
    return (raw & 0x8000) ? val : -val;
}

static void uart_radar_task(void *arg)
{
    const uart_config_t ucfg = {
        .baud_rate  = 256000,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_1, 8192, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_1, &ucfg));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM_1,
                                 RADAR_TXD, RADAR_RXD,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    ESP_LOGI(TAG, "uart_radar_task running on core %d", xPortGetCoreID());

    uint8_t pkt[30] = {0};

    for (;;) {
        uint8_t chunk[128];
        int len = uart_read_bytes(UART_NUM_1, chunk, sizeof(chunk),
                                  pdMS_TO_TICKS(50));

        for (int i = 0; i < len; i++) {
            memmove(pkt, pkt + 1, 29);
            pkt[29] = chunk[i];

            if (pkt[0] != 0xAA || pkt[29] != 0xCC) continue;
            if (pkt[1] != 0xFF || pkt[2] != 0x03 || pkt[3] != 0x00) {
                memset(pkt, 0, sizeof(pkt));
                continue;
            }

            int64_t ts = esp_timer_get_time() / 1000LL;

            uint16_t resolution = pkt[10] | ((uint16_t)pkt[11] << 8);

            sensor_event_t evt = {};
            evt.type = SENSOR_RADAR;
            evt.timestamp_ms = ts;

            if (resolution == 0) {
                evt.r_m       = NAN;
                evt.theta_deg = NAN;
            } else {
                int16_t x_mm = ld2450_to_signed(
                                   pkt[4] | ((uint16_t)pkt[5] << 8));
                int16_t y_mm = ld2450_to_signed(
                                   pkt[6] | ((uint16_t)pkt[7] << 8));

                float x_m = x_mm / 1000.0f;
                float y_m = y_mm / 1000.0f;
                float r_m       = sqrtf(x_m * x_m + y_m * y_m);
                float theta_deg = atan2f(x_m, y_m) * (180.0f / (float)M_PI);

                if (fabsf(theta_deg) <= 60.0f && r_m >= 0.1f) {
                    evt.r_m       = r_m;
                    evt.theta_deg = theta_deg;
                } else {
                    evt.r_m       = NAN;
                    evt.theta_deg = NAN;
                }
            }

            xQueueSendToBack(sensor_queue, &evt, 0);
            memset(pkt, 0, sizeof(pkt));
        }

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

/* ═════════════════════════════════════════════════════════════════
 *  TASK 3 — ML Inference  (Core 1, Priority 3)
 * ═════════════════════════════════════════════════════════════════*/
static void ml_inference_task(void *arg)
{
    /* ────────── 1. Load & verify model ────────── */
    const tflite::Model *model = tflite::GetModel(polar_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch: got %lu, expected %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "TFLite model loaded (%u bytes)", polar_model_tflite_len);

    /* ────────── 2. Register ops ────────── */
    /* Float32 MLP: Dense(relu) → Dense(relu) → Dense(linear)
     * FullyConnected handles the fused ReLU activation internally.
     * Reshape may be inserted by the converter for input flattening. */
    tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddReshape();

    /* ────────── 3. Allocate tensor arena in internal SRAM ────────── */
    constexpr int kTensorArenaSize = 32 * 1024;
    uint8_t *tensor_arena = (uint8_t *)heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate %d byte tensor arena!", kTensorArenaSize);
        vTaskDelete(NULL);
        return;
    }

    /* ────────── 4. Create interpreter ────────── */
    tflite::MicroInterpreter interpreter(model, resolver,
                                         tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed!");
        vTaskDelete(NULL);
        return;
    }

    TfLiteTensor *input  = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    ESP_LOGI(TAG, "Input:  dims=%d shape=[%d,%d] type=%d",
             input->dims->size,
             input->dims->data[0], input->dims->data[1],
             input->type);
    ESP_LOGI(TAG, "Output: dims=%d shape=[%d,%d] type=%d",
             output->dims->size,
             output->dims->data[0], output->dims->data[1],
             output->type);
    ESP_LOGI(TAG, "Arena used: %zu / %d bytes",
             interpreter.arena_used_bytes(), kTensorArenaSize);

    /* ────────── 5. Initialise fusion state ────────── */
    fusion_state_t state;
    state.last_radar_r     = R_MAX_M;
    state.last_radar_theta = 0.0f;
    state.last_wifi_r      = R_MAX_M;
    state.window_head      = 0;
    state.window_count     = 0;
    state.last_pred_r      = R_MAX_M / 2.0f;
    state.last_pred_theta  = 0.0f;
    memset(state.window, 0, sizeof(state.window));

    ESP_LOGI(TAG, "ml_inference_task running on core %d", xPortGetCoreID());

    /* ────────── 6. Main loop ────────── */
    sensor_event_t event;
    for (;;) {
        if (xQueueReceive(sensor_queue, &event, portMAX_DELAY) != pdTRUE)
            continue;

        /* ── Forward-fill (matches prepare_features.py exactly) ── */
        float feat_radar_r, feat_radar_theta, feat_radar_fresh;
        float feat_wifi_r, feat_wifi_fresh;

        if (event.type == SENSOR_RADAR) {
            if (!std::isnan(event.r_m) && !std::isnan(event.theta_deg)) {
                feat_radar_r     = event.r_m;
                feat_radar_theta = event.theta_deg;
                feat_radar_fresh = 1.0f;
                state.last_radar_r     = event.r_m;
                state.last_radar_theta = event.theta_deg;
            } else {
                feat_radar_r     = state.last_radar_r;
                feat_radar_theta = state.last_radar_theta;
                feat_radar_fresh = 0.0f;
            }
            feat_wifi_r     = state.last_wifi_r;
            feat_wifi_fresh = 0.0f;

        } else { /* SENSOR_WIFI */
            if (!std::isnan(event.r_m)) {
                feat_wifi_r     = event.r_m;
                feat_wifi_fresh = 1.0f;
                state.last_wifi_r = event.r_m;
            } else {
                feat_wifi_r     = state.last_wifi_r;
                feat_wifi_fresh = 0.0f;
            }
            feat_radar_r     = state.last_radar_r;
            feat_radar_theta = state.last_radar_theta;
            feat_radar_fresh = 0.0f;
        }

        /* ── Push 5-feature row into circular window ── */
        float *row = state.window[state.window_head];
        row[0] = norm_range(feat_radar_r);
        row[1] = norm_angle(feat_radar_theta);
        row[2] = feat_radar_fresh;
        row[3] = norm_range(feat_wifi_r);
        row[4] = feat_wifi_fresh;

        state.window_head = (state.window_head + 1) % WINDOW_SIZE;
        if (state.window_count < WINDOW_SIZE)
            state.window_count++;

        /* ── Skip inference until window is full ── */
        if (state.window_count < WINDOW_SIZE) {
            ESP_LOGD(TAG, "Window filling: %d/%d",
                     state.window_count, WINDOW_SIZE);
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }

        /* ── Flatten circular window into input tensor (oldest first) ── */
        float *input_data = input->data.f;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int idx = (state.window_head + i) % WINDOW_SIZE;
            for (int j = 0; j < NUM_FEATURES; j++) {
                input_data[i * NUM_FEATURES + j] = state.window[idx][j];
            }
        }

        /* ── Run inference and time it ── */
        int64_t t0 = esp_timer_get_time();
        TfLiteStatus status = interpreter.Invoke();
        int64_t t1 = esp_timer_get_time();
        int64_t inference_us = t1 - t0;

        if (status != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed!");
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }

        /* ── Denormalize and clamp ── */
        float *out = output->data.f;
        float pred_r_m       = denorm_range(out[0]);
        float pred_theta_deg = denorm_angle(out[1]);

        if (pred_r_m < R_MIN_M)           pred_r_m = R_MIN_M;
        if (pred_r_m > R_MAX_M)           pred_r_m = R_MAX_M;
        if (pred_theta_deg < THETA_MIN_DEG) pred_theta_deg = THETA_MIN_DEG;
        if (pred_theta_deg > THETA_MAX_DEG) pred_theta_deg = THETA_MAX_DEG;

        state.last_pred_r     = pred_r_m;
        state.last_pred_theta = pred_theta_deg;

        /* ── Print fused output ── */
        printf("FUSED,%.3f,%.1f,%lld,%lld\n",
               pred_r_m, pred_theta_deg,
               event.timestamp_ms, inference_us);

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

/* ═════════════════════════════════════════════════════════════════
 *  app_main
 * ═════════════════════════════════════════════════════════════════*/
extern "C" void app_main(void)
{
    /* --- NVS (required by Wi-Fi) --- */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    /* --- Shared sensor queue --- */
    sensor_queue = xQueueCreate(QUEUE_LEN, sizeof(sensor_event_t));
    if (!sensor_queue) {
        ESP_LOGE(TAG, "Failed to create sensor queue!");
        return;
    }
    ESP_LOGI(TAG, "Sensor queue created (%d entries)", QUEUE_LEN);

    /* --- Core 0: Wi-Fi + FTM (sole owner of radio hardware) --- */
    xTaskCreatePinnedToCore(wifi_ftm_task,      "wifi_ftm",
                            8192, NULL, 5, NULL, 0);

    /* --- Core 1: UART radar (high priority; ISR isolated from Wi-Fi) --- */
    xTaskCreatePinnedToCore(uart_radar_task,     "radar",
                            4096, NULL, 5, NULL, 1);

    /* --- Core 1: ML inference (lower priority; UART preempts) --- */
    xTaskCreatePinnedToCore(ml_inference_task,   "ml_infer",
                            16384, NULL, 3, NULL, 1);

    ESP_LOGI(TAG, "All tasks started. Core 0: WiFi/FTM. Core 1: UART + ML.");
}
