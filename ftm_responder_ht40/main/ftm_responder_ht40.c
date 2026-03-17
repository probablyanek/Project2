#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_mac.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_now.h"

#define TAG "FTM_RESPONDER"

// ESP-NOW Broadcast MAC Address
static const uint8_t broadcast_mac[ESP_NOW_ETH_ALEN] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

static void wifi_init_softap(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = "ISAC_TAG",
            .ssid_len = strlen("ISAC_TAG"),
            .channel = 6,
            .password = "12345678",
            .max_connection = 4,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .ftm_responder = true,
            .pmf_cfg = {
                .required = false,
            },
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    
    // Set HT40 bandwidth on AP interface
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT40));
    
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "SoftAP initialized. SSID: ISAC_TAG, Channel: 6, BW: HT40, FTM: Enabled");
}

static void heartbeat_task(void *pvParameter) {
    const char *heartbeat_msg = "LIVE";
    
    while (1) {
        esp_err_t err = esp_now_send(broadcast_mac, (const uint8_t *)heartbeat_msg, 4);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "ESP-NOW send error: %s", esp_err_to_name(err));
        } else {
            ESP_LOGD(TAG, "Broadcasted LIVE");
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void app_main(void) {
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialize WiFi in SoftAP mode with HT40 and FTM Responder configured
    wifi_init_softap();

    // Initialize ESP-NOW
    ESP_ERROR_CHECK(esp_now_init());
    
    // Add broadcast peer for ESP-NOW on the SoftAP interface
    esp_now_peer_info_t peer_info = {0};
    peer_info.channel = 6;
    peer_info.ifidx = WIFI_IF_AP;
    peer_info.encrypt = false;
    memcpy(peer_info.peer_addr, broadcast_mac, ESP_NOW_ETH_ALEN);
    
    ESP_ERROR_CHECK(esp_now_add_peer(&peer_info));

    // Spawn Heartbeat Task
    xTaskCreate(heartbeat_task, "heartbeat_task", 2048, NULL, 5, NULL);
}
