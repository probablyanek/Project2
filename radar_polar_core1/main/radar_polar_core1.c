#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"

#define TXD_PIN (GPIO_NUM_17)
#define RXD_PIN (GPIO_NUM_16)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

int16_t convertCoord(uint16_t raw) {
    int16_t val = raw & 0x7FFF;
    return (raw & 0x8000) ? val : -val;
}

void radar_task(void *arg) {
    const uart_config_t uart_config = {
        .baud_rate = 256000,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_1, 8192, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_1, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM_1, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    uint8_t packetBuffer[30] = {0};

    while (1) {
        uint8_t chunk[128];
        int len = uart_read_bytes(UART_NUM_1, chunk, sizeof(chunk), pdMS_TO_TICKS(10));
        
        for (int i = 0; i < len; i++) {
            // Sliding window 
            memmove(packetBuffer, packetBuffer + 1, 29);
            packetBuffer[29] = chunk[i];
            
            if (packetBuffer[0] == 0xAA && packetBuffer[29] == 0xCC) {
                uint32_t timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);
                
                // Target 1 checking (Resolution is at index 10 and 11)
                if ((packetBuffer[10] | (packetBuffer[11] << 8)) == 0) {
                    printf("RADAR,NAN,NAN,%lu\n", timestamp_ms);
                } else {
                    int16_t tx = convertCoord(packetBuffer[4] | (packetBuffer[5] << 8));
                    int16_t ty = convertCoord(packetBuffer[6] | (packetBuffer[7] << 8));
                    
                    float x_m = tx / 1000.0f;
                    float y_m = ty / 1000.0f;
                    float r_m = sqrtf(x_m * x_m + y_m * y_m);
                    float theta_deg = atan2f(x_m, y_m) * (180.0f / M_PI);
                    
                    if (theta_deg > 60.0f || theta_deg < -60.0f) {
                        printf("RADAR,NAN,NAN,%lu\n", timestamp_ms);
                    } else {
                        printf("RADAR,%.3f,%.1f,%lu\n", r_m, theta_deg, timestamp_ms);
                    }
                }
                
                // Clear buffer after successful parse
                memset(packetBuffer, 0, sizeof(packetBuffer));
            }
        }
        
        // Feed watchdog
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void app_main(void) {
    xTaskCreatePinnedToCore(radar_task, "radar_task", 4096, NULL, 5, NULL, 1);
}
