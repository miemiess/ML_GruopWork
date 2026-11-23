/*
 * ADS1292 driver for ESP-IDF. See include/ads1292.h and the TI datasheet
 * (vendor/ADS1292模块资料包/5 电子硬件/ads1292 芯片手册.pdf).
 */
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "ads1292.h"

static const char *TAG = "ads1292";

/* Opcodes */
#define CMD_WAKEUP  0x02
#define CMD_STANDBY 0x04
#define CMD_RESET   0x06
#define CMD_START   0x08
#define CMD_STOP    0x0A
#define CMD_RDATAC  0x10
#define CMD_SDATAC  0x11
#define CMD_RDATA   0x12
#define CMD_RREG    0x20
#define CMD_WREG    0x40

/* 4 tCLK at the internal 512 kHz clock = 7.8 us: gap between command bytes and before CS high. */
#define T_DECODE_US 8
#define FRAME_BYTES 9
#define VREF_UV     2420000.0f

struct ads1292_dev_t {
    spi_device_handle_t spi;
    ads1292_config_t cfg;
    SemaphoreHandle_t drdy_sem;
    volatile uint32_t drdy_count; /* incremented in ISR */
    uint32_t consumed;            /* drdy_count value of the last frame read */
    uint32_t dropped;
    bool running;
};

static inline bool pin_used(gpio_num_t pin)
{
    return pin >= 0;
}

static void IRAM_ATTR drdy_isr(void *arg)
{
    struct ads1292_dev_t *dev = arg;
    BaseType_t woken = pdFALSE;
    dev->drdy_count++;
    xSemaphoreGiveFromISR(dev->drdy_sem, &woken);
    if (woken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

static esp_err_t xfer_byte(struct ads1292_dev_t *dev, uint8_t tx, uint8_t *rx)
{
    spi_transaction_t t = {
        .length = 8,
        .flags = SPI_TRANS_USE_TXDATA | SPI_TRANS_USE_RXDATA,
    };
    t.tx_data[0] = tx;
    esp_err_t err = spi_device_polling_transmit(dev->spi, &t);
    if (err == ESP_OK && rx) {
        *rx = t.rx_data[0];
    }
    return err;
}

/* Send a multi-byte command with a decode gap after every byte, CS held low throughout. */
static esp_err_t send_bytes(struct ads1292_dev_t *dev, const uint8_t *tx, uint8_t *rx, size_t count)
{
    ESP_RETURN_ON_ERROR(spi_device_acquire_bus(dev->spi, portMAX_DELAY), TAG, "acquire bus");
    gpio_set_level(dev->cfg.cs, 0);
    esp_err_t err = ESP_OK;
    for (size_t i = 0; i < count && err == ESP_OK; i++) {
        err = xfer_byte(dev, tx[i], rx ? &rx[i] : NULL);
        esp_rom_delay_us(T_DECODE_US);
    }
    gpio_set_level(dev->cfg.cs, 1);
    spi_device_release_bus(dev->spi);
    esp_rom_delay_us(2);
    return err;
}

static esp_err_t send_cmd(struct ads1292_dev_t *dev, uint8_t cmd)
{
    return send_bytes(dev, &cmd, NULL, 1);
}

static esp_err_t read_frame(struct ads1292_dev_t *dev, uint8_t frame[FRAME_BYTES])
{
    static const uint8_t zeros[FRAME_BYTES] = {0};
    spi_transaction_t t = {
        .length = FRAME_BYTES * 8,
        .tx_buffer = zeros,
        .rx_buffer = frame,
    };
    ESP_RETURN_ON_ERROR(spi_device_acquire_bus(dev->spi, portMAX_DELAY), TAG, "acquire bus");
    gpio_set_level(dev->cfg.cs, 0);
    esp_err_t err = spi_device_polling_transmit(dev->spi, &t);
    esp_rom_delay_us(T_DECODE_US);
    gpio_set_level(dev->cfg.cs, 1);
    spi_device_release_bus(dev->spi);
    return err;
}

static inline int32_t sign_extend24(const uint8_t *p)
{
    uint32_t raw = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
    return (int32_t)(raw << 8) >> 8;
}

esp_err_t ads1292_read_regs(ads1292_handle_t dev, uint8_t addr, uint8_t *buf, size_t count)
{
    ESP_RETURN_ON_FALSE(dev && buf && count > 0 && addr + count <= ADS1292_REG_COUNT,
                        ESP_ERR_INVALID_ARG, TAG, "bad register range");
    ESP_RETURN_ON_FALSE(!dev->running, ESP_ERR_INVALID_STATE, TAG, "stop the device before register access");
    uint8_t tx[2 + ADS1292_REG_COUNT] = {CMD_RREG | addr, (uint8_t)(count - 1)};
    uint8_t rx[2 + ADS1292_REG_COUNT];
    ESP_RETURN_ON_ERROR(send_bytes(dev, tx, rx, 2 + count), TAG, "RREG");
    memcpy(buf, rx + 2, count);
    return ESP_OK;
}

esp_err_t ads1292_write_regs(ads1292_handle_t dev, uint8_t addr, const uint8_t *buf, size_t count)
{
    ESP_RETURN_ON_FALSE(dev && buf && count > 0 && addr >= ADS1292_REG_CONFIG1 && addr + count <= ADS1292_REG_COUNT,
                        ESP_ERR_INVALID_ARG, TAG, "bad register range");
    ESP_RETURN_ON_FALSE(!dev->running, ESP_ERR_INVALID_STATE, TAG, "stop the device before register access");
    uint8_t tx[2 + ADS1292_REG_COUNT] = {CMD_WREG | addr, (uint8_t)(count - 1)};
    memcpy(tx + 2, buf, count);
    return send_bytes(dev, tx, NULL, 2 + count);
}

esp_err_t ads1292_read_id(ads1292_handle_t dev, uint8_t *id)
{
    return ads1292_read_regs(dev, ADS1292_REG_ID, id, 1);
}

esp_err_t ads1292_init(const ads1292_config_t *cfg, ads1292_handle_t *out_dev)
{
    ESP_RETURN_ON_FALSE(cfg && out_dev && pin_used(cfg->cs) && pin_used(cfg->drdy) && cfg->clock_hz > 0,
                        ESP_ERR_INVALID_ARG, TAG, "CS, DRDY and clock are required");
    esp_err_t ret = ESP_OK;
    bool isr_added = false;
    struct ads1292_dev_t *dev = calloc(1, sizeof(*dev));
    ESP_RETURN_ON_FALSE(dev, ESP_ERR_NO_MEM, TAG, "no memory");
    dev->cfg = *cfg;

    uint64_t out_mask = BIT64(cfg->cs);
    if (pin_used(cfg->start)) out_mask |= BIT64(cfg->start);
    if (pin_used(cfg->pwdn)) out_mask |= BIT64(cfg->pwdn);
    if (pin_used(cfg->clksel)) out_mask |= BIT64(cfg->clksel);
    gpio_config_t out_conf = {.pin_bit_mask = out_mask, .mode = GPIO_MODE_OUTPUT};
    ESP_GOTO_ON_ERROR(gpio_config(&out_conf), err, TAG, "output pins");
    gpio_set_level(cfg->cs, 1);
    if (pin_used(cfg->start)) gpio_set_level(cfg->start, 0);
    if (pin_used(cfg->clksel)) gpio_set_level(cfg->clksel, 1);
    if (pin_used(cfg->pwdn)) gpio_set_level(cfg->pwdn, 1);

    gpio_config_t drdy_conf = {
        .pin_bit_mask = BIT64(cfg->drdy),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    ESP_GOTO_ON_ERROR(gpio_config(&drdy_conf), err, TAG, "DRDY pin");

    spi_device_interface_config_t dev_conf = {
        .mode = 1, /* CPOL = 0, CPHA = 1 */
        .clock_speed_hz = cfg->clock_hz,
        .spics_io_num = -1, /* CS handled manually */
        .queue_size = 1,
    };
    ESP_GOTO_ON_ERROR(spi_bus_add_device(cfg->host, &dev_conf, &dev->spi), err, TAG, "add SPI device");

    dev->drdy_sem = xSemaphoreCreateBinary();
    ESP_GOTO_ON_FALSE(dev->drdy_sem, ESP_ERR_NO_MEM, err, TAG, "semaphore");
    ret = gpio_install_isr_service(0);
    ESP_GOTO_ON_FALSE(ret == ESP_OK || ret == ESP_ERR_INVALID_STATE, ret, err, TAG, "ISR service");
    ESP_GOTO_ON_ERROR(gpio_isr_handler_add(cfg->drdy, drdy_isr, dev), err, TAG, "DRDY ISR");
    isr_added = true;

    /* Power-on reset time plus margin, then software reset (needs 18 tCLK afterwards). */
    vTaskDelay(pdMS_TO_TICKS(150));
    ESP_GOTO_ON_ERROR(send_cmd(dev, CMD_RESET), err, TAG, "RESET");
    vTaskDelay(pdMS_TO_TICKS(2));
    /* The chip powers up in RDATAC mode; registers are only accessible after SDATAC. */
    ESP_GOTO_ON_ERROR(send_cmd(dev, CMD_SDATAC), err, TAG, "SDATAC");

    uint8_t id = 0;
    ESP_GOTO_ON_ERROR(ads1292_read_id(dev, &id), err, TAG, "read ID");
    if (id != ADS1292_ID_ADS1292 && id != ADS1292_ID_ADS1292R) {
        ESP_LOGE(TAG, "unexpected ID 0x%02X (0x00/0xFF usually means wiring, power or SPI mode problem)", id);
        ret = ESP_ERR_NOT_FOUND;
        goto err;
    }
    ESP_LOGI(TAG, "found %s (ID 0x%02X)", id == ADS1292_ID_ADS1292R ? "ADS1292R" : "ADS1292", id);
    *out_dev = dev;
    return ESP_OK;

err:
    if (isr_added) gpio_isr_handler_remove(cfg->drdy);
    if (dev->drdy_sem) vSemaphoreDelete(dev->drdy_sem);
    if (dev->spi) spi_bus_remove_device(dev->spi);
    free(dev);
    return ret;
}

esp_err_t ads1292_deinit(ads1292_handle_t dev)
{
    ESP_RETURN_ON_FALSE(dev, ESP_ERR_INVALID_ARG, TAG, "null handle");
    if (dev->running) {
        ads1292_stop(dev);
    }
    gpio_isr_handler_remove(dev->cfg.drdy);
    spi_bus_remove_device(dev->spi);
    vSemaphoreDelete(dev->drdy_sem);
    free(dev);
    return ESP_OK;
}

esp_err_t ads1292_configure(ads1292_handle_t dev, const ads1292_settings_t *s)
{
    ESP_RETURN_ON_FALSE(dev && s && s->rate <= ADS1292_RATE_8000 && s->pga <= ADS1292_PGA_12,
                        ESP_ERR_INVALID_ARG, TAG, "bad settings");
    ESP_RETURN_ON_FALSE(!dev->running, ESP_ERR_INVALID_STATE, TAG, "stop the device first");

    const uint8_t chset = (uint8_t)((s->pga << 4) | s->input);
    /* Registers 0x01..0x07 */
    const uint8_t block[] = {
        (uint8_t)s->rate,                                   /* CONFIG1: continuous conversion */
        (uint8_t)(0xA0 | (s->input == ADS1292_INPUT_TEST ? 0x03 : 0x00)), /* CONFIG2: bit7=1, ref buffer on, VREF 2.42 V, [1 Hz test] */
        0x10,                                               /* LOFF: bit4 must be 1, lead-off off */
        chset,                                              /* CH1SET */
        chset,                                              /* CH2SET */
        (uint8_t)(s->rld_enable ? 0x2F : 0x00),             /* RLD_SENS: buffer on, sense IN1P/N, IN2P/N */
        0x00,                                               /* LOFF_SENS */
    };
    /* Registers 0x09..0x0A (skip LOFF_STAT) */
    const uint8_t resp[] = {
        0x02,                                               /* RESP1: bit1 must be 1 */
        0x03,                                               /* RESP2: bit0 must be 1, RLDREF internal (AVDD+AVSS)/2 */
    };
    ESP_RETURN_ON_ERROR(ads1292_write_regs(dev, ADS1292_REG_CONFIG1, block, sizeof(block)), TAG, "write config");
    ESP_RETURN_ON_ERROR(ads1292_write_regs(dev, ADS1292_REG_RESP1, resp, sizeof(resp)), TAG, "write RESP");
    vTaskDelay(pdMS_TO_TICKS(50)); /* reference buffer settling */

    uint8_t check[sizeof(block)];
    ESP_RETURN_ON_ERROR(ads1292_read_regs(dev, ADS1292_REG_CONFIG1, check, sizeof(check)), TAG, "read back");
    for (size_t i = 0; i < sizeof(block); i++) {
        if (check[i] != block[i]) {
            ESP_LOGE(TAG, "reg 0x%02X wrote 0x%02X read 0x%02X", (unsigned)(ADS1292_REG_CONFIG1 + i), block[i], check[i]);
            return ESP_ERR_INVALID_RESPONSE;
        }
    }
    return ESP_OK;
}

esp_err_t ads1292_start(ads1292_handle_t dev)
{
    ESP_RETURN_ON_FALSE(dev, ESP_ERR_INVALID_ARG, TAG, "null handle");
    ESP_RETURN_ON_FALSE(!dev->running, ESP_ERR_INVALID_STATE, TAG, "already running");
    ESP_RETURN_ON_ERROR(send_cmd(dev, CMD_RDATAC), TAG, "RDATAC");
    xSemaphoreTake(dev->drdy_sem, 0);
    dev->consumed = dev->drdy_count;
    dev->dropped = 0;
    dev->running = true;
    if (pin_used(dev->cfg.start)) {
        gpio_set_level(dev->cfg.start, 1);
        return ESP_OK;
    }
    esp_err_t err = send_cmd(dev, CMD_START);
    if (err != ESP_OK) {
        dev->running = false;
    }
    return err;
}

esp_err_t ads1292_stop(ads1292_handle_t dev)
{
    ESP_RETURN_ON_FALSE(dev, ESP_ERR_INVALID_ARG, TAG, "null handle");
    if (pin_used(dev->cfg.start)) {
        gpio_set_level(dev->cfg.start, 0);
    } else {
        ESP_RETURN_ON_ERROR(send_cmd(dev, CMD_STOP), TAG, "STOP");
    }
    ESP_RETURN_ON_ERROR(send_cmd(dev, CMD_SDATAC), TAG, "SDATAC");
    dev->running = false;
    return ESP_OK;
}

esp_err_t ads1292_wait_sample(ads1292_handle_t dev, ads1292_sample_t *sample, uint32_t timeout_ms)
{
    ESP_RETURN_ON_FALSE(dev && sample, ESP_ERR_INVALID_ARG, TAG, "null argument");
    if (!dev->running) {
        return ESP_ERR_INVALID_STATE;
    }
    if (xSemaphoreTake(dev->drdy_sem, pdMS_TO_TICKS(timeout_ms)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }
    uint32_t count = dev->drdy_count;
    if (count - dev->consumed > 1) {
        dev->dropped += count - dev->consumed - 1;
    }
    dev->consumed = count;

    uint8_t frame[FRAME_BYTES];
    ESP_RETURN_ON_ERROR(read_frame(dev, frame), TAG, "read frame");
    sample->status = ((uint32_t)frame[0] << 16) | ((uint32_t)frame[1] << 8) | frame[2];
    sample->ch[0] = sign_extend24(&frame[3]);
    sample->ch[1] = sign_extend24(&frame[6]);
    return (frame[0] & 0xF0) == 0xC0 ? ESP_OK : ESP_ERR_INVALID_RESPONSE;
}

uint32_t ads1292_dropped_samples(ads1292_handle_t dev)
{
    return dev ? dev->dropped : 0;
}

uint32_t ads1292_rate_hz(ads1292_rate_t rate)
{
    return rate <= ADS1292_RATE_8000 ? 125u << rate : 0;
}

int ads1292_pga_gain(ads1292_pga_t pga)
{
    static const int gains[] = {6, 1, 2, 3, 4, 8, 12};
    return pga <= ADS1292_PGA_12 ? gains[pga] : 0;
}

float ads1292_code_to_uv(int32_t code, ads1292_pga_t pga)
{
    return (float)code * VREF_UV / ((float)ads1292_pga_gain(pga) * 8388607.0f);
}
