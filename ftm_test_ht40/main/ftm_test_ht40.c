#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_mac.h"

#define TAG "FTM_INITIATOR"
#define SSID "ISAC_TAG"

static uint8_t responder_mac[6] = {0};
static bool is_connected = false;

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected from AP! Reconnecting...");
        is_connected = false;
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        // Technically this gets triggered once we're fully connected and got an IP
        // BUT wait, we don't need IP necessarily for FTM, but let's just trigger FTM after we get connected.
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
        ESP_LOGI(TAG, "Connected to AP!");
        
        // Ensure HT40 on the STA interface
        esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40);

        wifi_ap_record_t ap_info;
        if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
            memcpy(responder_mac, ap_info.bssid, 6);
            ESP_LOGI(TAG, "Responder MAC: %02x:%02x:%02x:%02x:%02x:%02x", 
                     responder_mac[0], responder_mac[1], responder_mac[2], 
                     responder_mac[3], responder_mac[4], responder_mac[5]);
            is_connected = true;
        }
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_FTM_REPORT) {
        wifi_event_ftm_report_t *report = (wifi_event_ftm_report_t *)event_data;
        uint32_t timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);

        if (report->status == FTM_STATUS_SUCCESS) {
            if (report->ftm_report_num_entries > 0 && report->ftm_report_data != NULL) {
                float min_r_m = 99999.0f;
                char raw_str[512] = "WIFI_RAW";
                char tmp[32];
                
                for (int i = 0; i < report->ftm_report_num_entries; i++) {
                    uint32_t rtt_ps = report->ftm_report_data[i].rtt;
                    float r_m = (float)rtt_ps * 1e-12f * 299792458.0f / 2.0f;
                    
                    if (r_m < min_r_m) {
                        min_r_m = r_m;
                    }
                    
                    snprintf(tmp, sizeof(tmp), ",%.3f", r_m);
                    strncat(raw_str, tmp, sizeof(raw_str) - strlen(raw_str) - 1);
                }
                
                printf("WIFI,%.3f,%lu\n", min_r_m, timestamp_ms);
                printf("%s,%lu\n", raw_str, timestamp_ms);
            }
        } else {
            // ESP_LOGD(TAG, "FTM Report failed with status: %d", report->status);
        }
        
        if (report->ftm_report_data) {
            free(report->ftm_report_data);
        }
    }
}

static void espnow_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len) {
    if (len == 4 && memcmp(data, "LIVE", 4) == 0) {
        uint32_t timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);
        printf("TAG_ALIVE,%lu\n", timestamp_ms);
    }
}

static void timer_callback(void* arg) {
    if (is_connected) {
        wifi_ftm_initiator_cfg_t ftm_cfg = {
            .channel = 6, 
            .frm_count = 16,
            .burst_period = 2,
            .use_get_report_api = false
        };
        memcpy(ftm_cfg.resp_mac, responder_mac, 6);
        esp_wifi_ftm_initiate_session(&ftm_cfg);
    }
}

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = SSID,
            .password = "12345678", // Must match responder
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_recv_cb(espnow_recv_cb));

    // Create and start periodic timer for FTM
    const esp_timer_create_args_t timer_args = {
        .callback = &timer_callback,
        .name = "ftm_timer",
        .skip_unhandled_events = true
    };
    esp_timer_handle_t timer;
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(timer, 500000)); // 500 ms
}
