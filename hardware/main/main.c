/*
 * EMGInsight hardware bring-up firmware: initializes the ADS1292, prints its registers,
 * then reports one line of statistics per second (rate, drops, bad frames,
 * per-channel min/max/mean in microvolts).
 *
 * Expected with "Internal test signal": min about -1000 uV, max about +1000 uV.
 */
#include <inttypes.h>
#include <float.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/spi_master.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "ads1292.h"

static const char *TAG = "main";

#define ADS_HOST SPI3_HOST

#if CONFIG_ADS1292_INPUT_TEST
#define ADS_INPUT ADS1292_INPUT_TEST
#elif CONFIG_ADS1292_INPUT_SHORTED
#define ADS_INPUT ADS1292_INPUT_SHORTED
#else
#define ADS_INPUT ADS1292_INPUT_NORMAL
#endif

typedef struct {
    float min, max;
    double sum;
} channel_stats_t;

static void stats_reset(channel_stats_t *s)
{
    s->min = FLT_MAX;
    s->max = -FLT_MAX;
    s->sum = 0;
}

static void stats_add(channel_stats_t *s, float v)
{
    if (v < s->min) s->min = v;
    if (v > s->max) s->max = v;
    s->sum += v;
}

static void dump_registers(ads1292_handle_t dev)
{
    uint8_t regs[ADS1292_REG_COUNT];
    if (ads1292_read_regs(dev, 0, regs, sizeof(regs)) != ESP_OK) {
        ESP_LOGE(TAG, "register read failed");
        return;
    }
    for (int i = 0; i < ADS1292_REG_COUNT; i++) {
        ESP_LOGI(TAG, "reg 0x%02X = 0x%02X", i, regs[i]);
    }
}

static void acquisition_task(void *arg)
{
    ads1292_handle_t dev = arg;
    const ads1292_pga_t pga = (ads1292_pga_t)CONFIG_ADS1292_GAIN_CODE;
    const uint32_t rate = ads1292_rate_hz((ads1292_rate_t)CONFIG_ADS1292_RATE_CODE);
    channel_stats_t st[2];
    uint32_t n = 0, bad = 0, timeouts = 0;
    stats_reset(&st[0]);
    stats_reset(&st[1]);

    ESP_ERROR_CHECK(ads1292_start(dev));
    while (true) {
        ads1292_sample_t s;
        esp_err_t err = ads1292_wait_sample(dev, &s, 100);
        if (err == ESP_ERR_TIMEOUT) {
            timeouts++;
            ESP_LOGW(TAG, "no DRDY for 100 ms (check DRDY/START wiring), timeouts=%" PRIu32, timeouts);
            continue;
        }
        if (err != ESP_OK) {
            bad++;
        }
        stats_add(&st[0], ads1292_code_to_uv(s.ch[0], pga));
        stats_add(&st[1], ads1292_code_to_uv(s.ch[1], pga));
        if (++n >= rate) {
            ESP_LOGI(TAG, "%" PRIu32 " sps, dropped %" PRIu32 ", bad %" PRIu32
                     " | CH1 min %.1f max %.1f mean %.1f uV | CH2 min %.1f max %.1f mean %.1f uV",
                     n, ads1292_dropped_samples(dev), bad,
                     st[0].min, st[0].max, st[0].sum / n, st[1].min, st[1].max, st[1].sum / n);
            n = 0;
            stats_reset(&st[0]);
            stats_reset(&st[1]);
        }
    }
}

void app_main(void)
{
    spi_bus_config_t bus = {
        .sclk_io_num = CONFIG_ADS1292_PIN_SCLK,
        .mosi_io_num = CONFIG_ADS1292_PIN_MOSI,
        .miso_io_num = CONFIG_ADS1292_PIN_MISO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 32,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(ADS_HOST, &bus, SPI_DMA_DISABLED));

    ads1292_config_t cfg = {
        .host = ADS_HOST,
        .clock_hz = CONFIG_ADS1292_SPI_CLOCK_KHZ * 1000,
        .cs = (gpio_num_t)CONFIG_ADS1292_PIN_CS,
        .drdy = (gpio_num_t)CONFIG_ADS1292_PIN_DRDY,
        .start = (gpio_num_t)CONFIG_ADS1292_PIN_START,
        .pwdn = (gpio_num_t)CONFIG_ADS1292_PIN_PWDN,
        .clksel = (gpio_num_t)CONFIG_ADS1292_PIN_CLKSEL,
    };
    ads1292_handle_t dev;
    ESP_ERROR_CHECK(ads1292_init(&cfg, &dev));

    ads1292_settings_t settings = {
        .rate = (ads1292_rate_t)CONFIG_ADS1292_RATE_CODE,
        .pga = (ads1292_pga_t)CONFIG_ADS1292_GAIN_CODE,
        .input = ADS_INPUT,
#if CONFIG_ADS1292_RLD_ENABLE
        .rld_enable = true,
#else
        .rld_enable = false,
#endif
    };
    ESP_ERROR_CHECK(ads1292_configure(dev, &settings));
    dump_registers(dev);
    ESP_LOGI(TAG, "rate %" PRIu32 " SPS, PGA %d, input mux 0x%X",
             ads1292_rate_hz(settings.rate), ads1292_pga_gain(settings.pga), (unsigned)settings.input);

    xTaskCreatePinnedToCore(acquisition_task, "ads1292", 4096, dev, configMAX_PRIORITIES - 2, NULL, 1);
}
