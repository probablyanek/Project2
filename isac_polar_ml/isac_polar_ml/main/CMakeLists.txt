idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro driver esp_wifi esp_timer nvs_flash esp_event
)