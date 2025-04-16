#include "Particle.h"

#include <Proximity_Gesture_Detection_inferencing.h>
#include <Adafruit_SH110X.h>
#include <Adafruit_VCNL4040.h>

#define SAMPLING_FREQ_HZ 50                        // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS 1000 / SAMPLING_FREQ_HZ // Sampling period (ms)
#define NUM_SAMPLES 100                            // 100 samples at 50 Hz is 2 sec window

#define BUTTON_A D4
#define BUTTON_B D3
#define BUTTON_C D22

Adafruit_SH1107 display = Adafruit_SH1107(64, 128, &Wire);
Adafruit_VCNL4040 vcnl4040 = Adafruit_VCNL4040();

ei_impulse_result_t result = {0};

float ambientLightMin = 0.0;
float ambientLightMax = 0.0; // dynamically determined based on training environment

float proximityMin = 0.0;
float proximityMax = 1000.0; // experimentally determined

SerialLogHandler logHandler(LOG_LEVEL_ERROR);

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr);
void setup();
void loop();

static float features[200];

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void print_inference_result(ei_impulse_result_t result);

void setup()
{
    SystemPowerConfiguration powerConfig = System.getPowerConfiguration();
    powerConfig.auxiliaryPowerControlPin(D23).interruptPin(A6);
    System.setPowerConfiguration(powerConfig);

    pinMode(BUTTON_A, INPUT_PULLUP);
    pinMode(BUTTON_B, INPUT_PULLUP);
    pinMode(BUTTON_C, INPUT_PULLUP);

    if (!vcnl4040.begin())
    {
        ei_printf("Couldn't find VCNL4040 chip");
    }
    vcnl4040.setProximityIntegrationTime(VCNL4040_PROXIMITY_INTEGRATION_TIME_8T);
    vcnl4040.setAmbientIntegrationTime(VCNL4040_AMBIENT_INTEGRATION_TIME_80MS);
    vcnl4040.setProximityLEDCurrent(VCNL4040_LED_CURRENT_120MA);
    vcnl4040.setProximityLEDDutyCycle(VCNL4040_LED_DUTY_1_40);

    // Put initialization like pinMode and begin functions here
    display.begin(0x3C, true); // Address 0x3C default

    // Clear the buffer.
    display.clearDisplay();
    display.display();
    display.setRotation(1);
}

void loop()
{
    unsigned long timestamp;
    while (digitalRead(BUTTON_A) == 1)
    {
        uint16_t proximity = vcnl4040.getProximity();
        uint16_t ambientLight = vcnl4040.getAmbientLight();

        if (ambientLight > ambientLightMax)
            ambientLightMax = ambientLight;

        float normProximity =
            proximity < proximityMin   ? 0.0
            : proximity > proximityMax ? 1.0
                                       : (float)(proximity - proximityMin) / (proximityMax - proximityMin);

        float normAmbientLight =
            ambientLight < ambientLightMin   ? 0.0
            : ambientLight > ambientLightMax ? 1.0
                                             : (float)(ambientLight - ambientLightMin) / (ambientLightMax - ambientLightMin);

        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SH110X_WHITE);
        display.setCursor(0, 0);
        display.print("Norm Prox: ");
        display.println(normProximity);
        display.print("Norm Light: ");
        display.println(normAmbientLight);
        display.println("------");

        // Print how long it took to perform inference
        ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
                  result.timing.dsp,
                  result.timing.classification,
                  result.timing.anomaly);

        ei_printf("Predictions:\r\n");
        for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
        {
            ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
            ei_printf("%.5f\r\n", result.classification[i].value);
            display.print(ei_classifier_inferencing_categories[i]);
            display.print(": ");
            display.println(result.classification[i].value);
        }
        display.display();
    }

    // Record samples in buffer
    int j = 0;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        // Take timestamp so we can hit our target frequency
        timestamp = millis();

        uint16_t proximity = vcnl4040.getProximity();
        uint16_t ambientLight = vcnl4040.getAmbientLight();

        float normProximity =
            proximity < proximityMin   ? 0.0
            : proximity > proximityMax ? 1.0
                                       : (float)(proximity - proximityMin) / (proximityMax - proximityMin);

        float normAmbientLight =
            ambientLight < ambientLightMin   ? 0.0
            : ambientLight > ambientLightMax ? 1.0
                                             : (float)(ambientLight - ambientLightMin) / (ambientLightMax - ambientLightMin);

        features[j] = normProximity;
        features[j + 1] = normAmbientLight;
        j += 2;

        // Wait just long enough for our sampling period
        while (millis() < timestamp + SAMPLING_PERIOD_MS)
            ;
    }

    // the features are stored into flash, and we don't want to load everything into RAM
    signal_t features_signal;
    features_signal.total_length = sizeof(features) / sizeof(features[0]);
    features_signal.get_data = &raw_feature_get_data;

    // invoke the impulse
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
    if (res != EI_IMPULSE_OK)
    {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        return;
    }

    // print inference return code
    ei_printf("run_classifier returned: %d\r\n", res);

    // Make sure the button has been released for a few milliseconds
    while (digitalRead(BUTTON_A) == 0)
        ;
}