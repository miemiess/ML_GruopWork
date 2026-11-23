idf_component_register(SRCS "main.c"
                       PRIV_REQUIRES ads1292 esp_driver_spi esp_driver_gpio
                       INCLUDE_DIRS ".")
