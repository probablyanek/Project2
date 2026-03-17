/* ═══════════════════════════════════════════════════════════════════════════
 *  polar_logger_v2  —  ISAC Fused Sensor Logger
 *
 *  3-task FreeRTOS architecture on ESP32-S3
 *    Task 1  wifi_ftm_task   Core 0   Prio 5   Wi-Fi STA + FTM + ESP-NOW
 *    Task 2  uart_radar_task Core 1   Prio 5   HLK-LD2450 UART parser
 *    Task 3  csv_print_task  Core 1   Prio 2   Serialises queue → UART0
 *
 *  All sensor readings go through a single FreeRTOS queue as
 *  sensor_event_t structs, then get printed as CSV on the USB serial.
 * ═══════════════════════════════════════════════════════════════════════════*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "driver/uart.h"
#include "driver/gpio.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_mac.h"
#include "esp_now.h"
#include "nvs_flash.h"

/* ───────────────────────── Constants ────────────────────────── */
#define TAG_WIFI   "WIFI_FTM"
#define TAG_RADAR  "RADAR"
#define TAG_CSV    "CSV"

#define SSID       "ISAC_TAG"
#define PASSWORD   "12345678"

#define RADAR_TXD  (GPIO_NUM_17)
#define RADAR_RXD  (GPIO_NUM_16)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* Speed of light (m/s) */
#define C_LIGHT    299792458.0f

/* Queue depth */
#define QUEUE_LEN  64

/* ───────────────────────── Data types ───────────────────────── */
typedef enum { SENSOR_RADAR = 0, SENSOR_WIFI = 1 } sensor_type_t;

typedef struct {
    sensor_type_t type;
    float         r_m;            /* range in metres              */
    float         theta_deg;      /* angle in degrees (NAN for Wi-Fi) */
    int64_t       timestamp_ms;
} sensor_event_t;

/* ───────────────────────── Shared queue ─────────────────────── */
static QueueHandle_t sensor_queue = NULL;

/* ═════════════════════════════════════════════════════════════════
 *  TASK 1 — Wi-Fi FTM  (Core 0, Priority 5)
 * ═════════════════════════════════════════════════════════════════*/
static uint8_t responder_mac[6] = {0};
static volatile bool wifi_connected = false;

/* ---- Wi-Fi + FTM event handler ---- */
static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG_WIFI, "Disconnected – reconnecting…");
        wifi_connected = false;
        esp_wifi_connect();

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
        ESP_LOGI(TAG_WIFI, "Connected to ISAC_TAG");
        esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40);

        wifi_ap_record_t ap;
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            memcpy(responder_mac, ap.bssid, 6);
            ESP_LOGI(TAG_WIFI, "Responder MAC: %02x:%02x:%02x:%02x:%02x:%02x",
                     responder_mac[0], responder_mac[1], responder_mac[2],
                     responder_mac[3], responder_mac[4], responder_mac[5]);
            wifi_connected = true;
        }

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_FTM_REPORT) {
        /* ── Process FTM burst ── */
        wifi_event_ftm_report_t *rpt = (wifi_event_ftm_report_t *)data;
        int64_t ts = esp_timer_get_time() / 1000LL;

        if (rpt->status == FTM_STATUS_SUCCESS &&
            rpt->ftm_report_num_entries > 0   &&
            rpt->ftm_report_data != NULL)
        {
            float min_r = 99999.0f;
            for (int i = 0; i < rpt->ftm_report_num_entries; i++) {
                float r = (float)rpt->ftm_report_data[i].rtt
                          * 1e-12f * C_LIGHT / 2.0f;
                if (r < min_r) min_r = r;
            }

            sensor_event_t evt = {
                .type         = SENSOR_WIFI,
                .r_m          = min_r,
                .theta_deg    = NAN,
                .timestamp_ms = ts
            };
            xQueueSendToBack(sensor_queue, &evt, 0);
        }

        if (rpt->ftm_report_data) free(rpt->ftm_report_data);
    }
}

/* ---- ESP-NOW heartbeat receiver ---- */
static void espnow_recv_cb(const esp_now_recv_info_t *info,
                            const uint8_t *data, int len)
{
    if (len == 4 && memcmp(data, "LIVE", 4) == 0) {
        ESP_LOGD(TAG_WIFI, "TAG_ALIVE");
    }
}

/* ---- FTM periodic timer callback ---- */
static void ftm_timer_cb(void *arg)
{
    if (!wifi_connected) return;

    wifi_ftm_initiator_cfg_t cfg = {
        .channel              = 6,
        .frm_count            = 16,
        .burst_period         = 2,
        .use_get_report_api   = false
    };
    memcpy(cfg.resp_mac, responder_mac, 6);
    esp_wifi_ftm_initiate_session(&cfg);
}

/* ---- Task body ---- */
static void wifi_ftm_task(void *arg)
{
    /* --- NVS --- */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

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

    wifi_config_t sta_cfg = {
        .sta = {
            .ssid     = SSID,
            .password = PASSWORD,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* --- ESP-NOW --- */
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_recv_cb(espnow_recv_cb));

    /* --- FTM periodic timer (500 ms) --- */
    const esp_timer_create_args_t tmr = {
        .callback             = &ftm_timer_cb,
        .name                 = "ftm_timer",
        .skip_unhandled_events = true
    };
    esp_timer_handle_t timer;
    ESP_ERROR_CHECK(esp_timer_create(&tmr, &timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(timer, 500000));   /* 500 ms */

    ESP_LOGI(TAG_WIFI, "wifi_ftm_task running on core %d", xPortGetCoreID());

    /* Keep the task alive so the timer and callbacks remain valid */
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
    /* --- UART1 init (ISR registers on Core 1) --- */
    const uart_config_t ucfg = {
        .baud_rate  = 256000,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_1, 8192, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_1, &ucfg));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM_1,
                                 RADAR_TXD, RADAR_RXD,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    ESP_LOGI(TAG_RADAR, "uart_radar_task running on core %d", xPortGetCoreID());

    uint8_t pkt[30] = {0};          /* sliding-window packet buffer */

    for (;;) {
        uint8_t chunk[128];
        int len = uart_read_bytes(UART_NUM_1, chunk, sizeof(chunk),
                                  pdMS_TO_TICKS(50));

        for (int i = 0; i < len; i++) {
            /* Slide window by 1 byte */
            memmove(pkt, pkt + 1, 29);
            pkt[29] = chunk[i];

            /* LD2450 frame: header 0xAA at [0], trailer 0xCC at [29] */
            if (pkt[0] != 0xAA || pkt[29] != 0xCC) continue;

            /* Verify sub-header bytes (0xFF 0x03 0x00) */
            if (pkt[1] != 0xFF || pkt[2] != 0x03 || pkt[3] != 0x00) {
                memset(pkt, 0, sizeof(pkt));
                continue;
            }

            int64_t ts = esp_timer_get_time() / 1000LL;

            /* Target-1 resolution field at bytes [10..11] —
               zero means "no target detected" */
            uint16_t resolution = pkt[10] | ((uint16_t)pkt[11] << 8);

            if (resolution == 0) {
                /* No target → still log so we can measure dropout rate */
                sensor_event_t evt = {
                    .type         = SENSOR_RADAR,
                    .r_m          = NAN,
                    .theta_deg    = NAN,
                    .timestamp_ms = ts
                };
                xQueueSendToBack(sensor_queue, &evt, 0);
            } else {
                int16_t x_mm = ld2450_to_signed(
                                   pkt[4] | ((uint16_t)pkt[5] << 8));
                int16_t y_mm = ld2450_to_signed(
                                   pkt[6] | ((uint16_t)pkt[7] << 8));

                float x_m = x_mm / 1000.0f;
                float y_m = y_mm / 1000.0f;
                float r_m       = sqrtf(x_m * x_m + y_m * y_m);
                float theta_deg = atan2f(x_m, y_m) * (180.0f / M_PI);

                sensor_event_t evt = { .type = SENSOR_RADAR, .timestamp_ms = ts };

                if (theta_deg > 60.0f || theta_deg < -60.0f) {
                    /* Outside ±60° FoV → mark as dropout */
                    evt.r_m       = NAN;
                    evt.theta_deg = NAN;
                } else {
                    evt.r_m       = r_m;
                    evt.theta_deg = theta_deg;
                }
                xQueueSendToBack(sensor_queue, &evt, 0);
            }

            /* Clear buffer after a successful parse */
            memset(pkt, 0, sizeof(pkt));
        }

        /* Feed IDLE watchdog */
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

/* ═════════════════════════════════════════════════════════════════
 *  TASK 3 — CSV Printer  (Core 1, Priority 2)
 * ═════════════════════════════════════════════════════════════════*/
static void csv_print_task(void *arg)
{
    ESP_LOGI(TAG_CSV, "csv_print_task running on core %d", xPortGetCoreID());

    sensor_event_t evt;
    for (;;) {
        if (xQueueReceive(sensor_queue, &evt, portMAX_DELAY) != pdTRUE)
            continue;

        const char *label = (evt.type == SENSOR_RADAR) ? "R" : "W";

        /* Build range string */
        char r_str[16];
        if (isnan(evt.r_m))
            strcpy(r_str, "NAN");
        else
            snprintf(r_str, sizeof(r_str), "%.3f", evt.r_m);

        /* Build angle string */
        char t_str[16];
        if (isnan(evt.theta_deg))
            strcpy(t_str, "NAN");
        else
            snprintf(t_str, sizeof(t_str), "%.1f", evt.theta_deg);

        /* sensor_type, r_m, theta_deg, timestamp_ms */
        printf("%s,%s,%s,%lld\n", label, r_str, t_str, evt.timestamp_ms);
    }
}

/* ═════════════════════════════════════════════════════════════════
 *  app_main
 * ═════════════════════════════════════════════════════════════════*/
void app_main(void)
{
    /* Create the shared sensor queue */
    sensor_queue = xQueueCreate(QUEUE_LEN, sizeof(sensor_event_t));
    assert(sensor_queue != NULL);

    /*  Task 1 — Wi-Fi + FTM   → Core 0, prio 5  */
    xTaskCreatePinnedToCore(wifi_ftm_task,   "wifi_ftm_task",
                            8192, NULL, 5, NULL, 0);

    /*  Task 2 — UART radar    → Core 1, prio 5  */
    xTaskCreatePinnedToCore(uart_radar_task, "uart_radar_task",
                            4096, NULL, 5, NULL, 1);

    /*  Task 3 — CSV printer   → Core 1, prio 2  */
    xTaskCreatePinnedToCore(csv_print_task,  "csv_print_task",
                            4096, NULL, 2, NULL, 1);
}
