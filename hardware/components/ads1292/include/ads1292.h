/*
 * ADS1292 driver for ESP-IDF (SPI master).
 *
 * The SPI bus must be initialized by the caller (spi_bus_initialize) before
 * ads1292_init(). CS is driven as a plain GPIO by this driver so the
 * datasheet's 4 tCLK command-decode gaps can be honoured.
 *
 * Threading: ads1292_start/stop/wait_sample are meant to be called from a
 * single acquisition task. Register access is only allowed while stopped
 * (the chip ignores RREG/WREG in RDATAC mode).
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Register map */
#define ADS1292_REG_ID        0x00
#define ADS1292_REG_CONFIG1   0x01
#define ADS1292_REG_CONFIG2   0x02
#define ADS1292_REG_LOFF      0x03
#define ADS1292_REG_CH1SET    0x04
#define ADS1292_REG_CH2SET    0x05
#define ADS1292_REG_RLD_SENS  0x06
#define ADS1292_REG_LOFF_SENS 0x07
#define ADS1292_REG_LOFF_STAT 0x08
#define ADS1292_REG_RESP1     0x09
#define ADS1292_REG_RESP2     0x0A
#define ADS1292_REG_GPIO      0x0B
#define ADS1292_REG_COUNT     12

/* Values of the ID register */
#define ADS1292_ID_ADS1292    0x53
#define ADS1292_ID_ADS1292R   0x73

/* Output data rate (CONFIG1.DR[2:0], internal 512 kHz clock) */
typedef enum {
    ADS1292_RATE_125 = 0,
    ADS1292_RATE_250,
    ADS1292_RATE_500,
    ADS1292_RATE_1000,
    ADS1292_RATE_2000,
    ADS1292_RATE_4000,
    ADS1292_RATE_8000,
} ads1292_rate_t;

/* PGA gain (CHnSET.GAIN[2:0]); note code 0 is gain 6 */
typedef enum {
    ADS1292_PGA_6 = 0,
    ADS1292_PGA_1,
    ADS1292_PGA_2,
    ADS1292_PGA_3,
    ADS1292_PGA_4,
    ADS1292_PGA_8,
    ADS1292_PGA_12,
} ads1292_pga_t;

/* Channel input (CHnSET.MUX[3:0]) */
typedef enum {
    ADS1292_INPUT_NORMAL  = 0x0, /* electrodes */
    ADS1292_INPUT_SHORTED = 0x1, /* inputs shorted: offset / noise check */
    ADS1292_INPUT_TEST    = 0x5, /* internal 1 Hz square wave, about +/-1 mV */
} ads1292_input_t;

typedef struct {
    spi_host_device_t host; /* bus already initialized by the caller */
    int clock_hz;           /* SCLK, e.g. 2 MHz */
    gpio_num_t cs;          /* required */
    gpio_num_t drdy;        /* required, falling edge = new sample */
    gpio_num_t start;       /* optional (GPIO_NUM_NC -> START/STOP commands) */
    gpio_num_t pwdn;        /* optional (GPIO_NUM_NC -> left to module pull-up) */
    gpio_num_t clksel;      /* optional, driven high = internal oscillator */
} ads1292_config_t;

typedef struct {
    ads1292_rate_t rate;
    ads1292_pga_t pga;      /* applied to both channels */
    ads1292_input_t input;  /* applied to both channels */
    bool rld_enable;        /* right-leg drive from both channels */
} ads1292_settings_t;

typedef struct {
    uint32_t status;        /* 24-bit status word, top nibble 0xC when valid */
    int32_t ch[2];          /* sign-extended 24-bit ADC codes */
} ads1292_sample_t;

typedef struct ads1292_dev_t *ads1292_handle_t;

/* Add the device to the bus, reset it, leave it in SDATAC mode and check the ID. */
esp_err_t ads1292_init(const ads1292_config_t *cfg, ads1292_handle_t *out_dev);
esp_err_t ads1292_deinit(ads1292_handle_t dev);

esp_err_t ads1292_read_id(ads1292_handle_t dev, uint8_t *id);
esp_err_t ads1292_read_regs(ads1292_handle_t dev, uint8_t addr, uint8_t *buf, size_t count);
esp_err_t ads1292_write_regs(ads1292_handle_t dev, uint8_t addr, const uint8_t *buf, size_t count);

/* Write all configuration registers and read them back for verification. Device must be stopped. */
esp_err_t ads1292_configure(ads1292_handle_t dev, const ads1292_settings_t *settings);

/* Enter RDATAC and start conversions. */
esp_err_t ads1292_start(ads1292_handle_t dev);
/* Stop conversions and leave RDATAC (registers accessible again). */
esp_err_t ads1292_stop(ads1292_handle_t dev);

/*
 * Block until the next DRDY, then read one frame.
 * Returns ESP_ERR_TIMEOUT if no DRDY arrives, ESP_ERR_INVALID_RESPONSE if the
 * status word is malformed (sample is still filled in).
 */
esp_err_t ads1292_wait_sample(ads1292_handle_t dev, ads1292_sample_t *sample, uint32_t timeout_ms);

/* Number of DRDY pulses that were not read since ads1292_start(). */
uint32_t ads1292_dropped_samples(ads1292_handle_t dev);

uint32_t ads1292_rate_hz(ads1292_rate_t rate);
int ads1292_pga_gain(ads1292_pga_t pga);
/* Input-referred microvolts, VREF = 2.42 V. */
float ads1292_code_to_uv(int32_t code, ads1292_pga_t pga);

#ifdef __cplusplus
}
#endif
