// =====================================================
//  OSMOS + ARGUS на ESP32-C3 Super Mini
//  Цикл: 30 замеров (1/с) с помпой -> MAD-фильтр -> Rs -> Xlog
//        -> нейросеть 3-16-8-1 -> PPM -> моль/л -> статус на OLED
//  Алгоритм и веса те же, что на Arduino Nano.
// =====================================================
#include <Wire.h>
#include <string.h>
#include <math.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>
#include "argus_weights.h"   // W1,B1,W2,B2,W3,B3,SC_MEAN,SC_SCALE

// ---------- Пины ----------
#define AO_PIN     0     // MQ-135 AO через делитель 100к/100к
#define DHT_PIN    1     // DHT22 DATA
#define PUMP_PIN   3     // затвор IRL3103 через 100 Ом
#define OLED_SDA   5
#define OLED_SCL   6
#define OLED_ADDR  0x3C

// ---------- Параметры измерения ----------
#define SAMPLES        30        // замеров за цикл (1 в секунду)
#define OVERSAMPLE     16        // усреднение внутри одного замера
#define RESULT_HOLD_MS 10000UL   // сколько держать итог на экране

// ---------- Физика датчика (как в live_argus.py) ----------
#define VCC_SENSOR  5.0f     // питание MQ-135 от блока
#define DIV_RATIO   2.0f     // делитель (100к+100к)/100к
#define RL_KOHM     9.52f    // 10 кОм модуля || 200 кОм делителя
#define R0_KOHM     76.8f    // как в обученной модели, не менять

// ---------- Термодинамика и пороги ----------
#define P_ATM        101325.0f
#define R_GAS        8.314f
#define THRESH_WARN  4.0e-7f   // моль/л: ниже = свежий
#define THRESH_BAD   1.0e-6f   // моль/л: выше = испорчено

Adafruit_SSD1306 display(128, 64, &Wire, -1);
DHT dht(DHT_PIN, DHT22);
bool oledOk = false;

// ---------- Статистика ----------
float medianOf(const float *a, int n) {
  float t[SAMPLES];
  memcpy(t, a, n * sizeof(float));
  for (int i = 1; i < n; i++) {
    float k = t[i];
    int j = i - 1;
    while (j >= 0 && t[j] > k) { t[j + 1] = t[j]; j--; }
    t[j + 1] = k;
  }
  return (n % 2) ? t[n / 2] : (t[n / 2 - 1] + t[n / 2]) / 2.0f;
}

// MAD z-score фильтр (порог 2.5), как filter_mad_zscore в Python
float madMean(const float *a, int n, int &kept) {
  float med = medianOf(a, n);
  float dev[SAMPLES];
  for (int i = 0; i < n; i++) dev[i] = fabsf(a[i] - med);
  float mad = medianOf(dev, n);

  float sum = 0;
  kept = 0;
  if (mad == 0) {                       // все значения почти одинаковые
    for (int i = 0; i < n; i++) sum += a[i];
    kept = n;
    return sum / n;
  }
  for (int i = 0; i < n; i++) {
    float z = 0.6745f * fabsf(a[i] - med) / mad;
    if (z <= 2.5f) { sum += a[i]; kept++; }
  }
  if (kept == 0) { kept = n; return med; }
  return sum / kept;
}

// ---------- Нейросеть: MLP 3-16-8-1, ReLU, выход = log10(PPM) ----------
float predictLogPPM(float xlog, float t, float h) {
  float x[3] = {xlog, t, h};
  float xs[3], h1[16], h2[8];

  for (int i = 0; i < 3; i++) xs[i] = (x[i] - SC_MEAN[i]) / SC_SCALE[i];

  for (int j = 0; j < 16; j++) {
    float s = B1[j];
    for (int i = 0; i < 3; i++) s += xs[i] * W1[i * 16 + j];
    h1[j] = s > 0 ? s : 0;
  }
  for (int j = 0; j < 8; j++) {
    float s = B2[j];
    for (int i = 0; i < 16; i++) s += h1[i] * W2[i * 8 + j];
    h2[j] = s > 0 ? s : 0;
  }
  float y = B3[0];
  for (int j = 0; j < 8; j++) y += h2[j] * W3[j];
  return y;
}

// ---------- Чтение АЦП: среднее из OVERSAMPLE отсчётов, вольты на пине ----------
float readPinVolts() {
  uint32_t sum = 0;
  for (int i = 0; i < OVERSAMPLE; i++) {
    sum += analogReadMilliVolts(AO_PIN);
    delayMicroseconds(200);
  }
  return (sum / (float)OVERSAMPLE) / 1000.0f;
}

// ---------- OLED ----------
void oledMsg(const char *l1, const char *l2 = "", const char *l3 = "") {
  if (!oledOk) return;
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);  display.println(l1);
  display.setCursor(0, 16); display.println(l2);
  display.setCursor(0, 32); display.println(l3);
  display.display();
}

void oledProgress(int n) {
  if (!oledOk) return;
  char a[24], b[24];
  snprintf(a, sizeof(a), "Sample %d/%d", n, SAMPLES);
  snprintf(b, sizeof(b), "%d s left", SAMPLES - n);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);  display.println("OSMOS: pumping...");
  display.setCursor(0, 20); display.println(a);
  display.setCursor(0, 32); display.println(b);
  // полоса прогресса
  display.drawRect(0, 50, 128, 10, SSD1306_WHITE);
  display.fillRect(2, 52, (124 * n) / SAMPLES, 6, SSD1306_WHITE);
  display.display();
}

void oledResult(float ppm, float cm, float t, float h, const char *status) {
  if (!oledOk) return;
  char a[24], b[24];
  snprintf(a, sizeof(a), "NH3 %.1f ppm", ppm);
  snprintf(b, sizeof(b), "%.1f umol/L", cm * 1e6f);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);  display.println("ARGUS RESULT (30 s)");
  display.setCursor(0, 12); display.println(a);
  display.setCursor(0, 22); display.println(b);
  display.setTextSize(2);
  display.setCursor(0, 36); display.println(status);
  display.setTextSize(1);
  display.setCursor(0, 56);
  display.printf("T%.1fC H%.0f%%", t, h);
  display.display();
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);               // помпа выключена при старте

  analogSetPinAttenuation(AO_PIN, ADC_11db); // диапазон до ~3.1 В
  dht.begin();

  Wire.begin(OLED_SDA, OLED_SCL);
  oledOk = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oledOk) Serial.println("OLED not found (check 0x3C/0x3D, SDA/SCL)");

  oledMsg("OSMOS + ARGUS", "ESP32-C3", "starting...");
  Serial.println("OSMOS + ARGUS ESP32-C3 started");
  delay(3000);
}

// ---------- Основной цикл ----------
void loop() {
  unsigned long cycleStart = millis();
  float buf[SAMPLES];

  // 1. Прокачка и сбор 30 замеров ровно за 30 секунд
  digitalWrite(PUMP_PIN, HIGH);
  for (int i = 0; i < SAMPLES; i++) {
    buf[i] = readPinVolts();
    oledProgress(i + 1);
    while (millis() - cycleStart < (unsigned long)(i + 1) * 1000UL) delay(5);
  }
  digitalWrite(PUMP_PIN, LOW);

  // 2. Фильтрация и физика: напряжение пина -> напряжение датчика -> Rs -> Xlog
  int kept = 0;
  float vpin = madMean(buf, SAMPLES, kept);
  float vout = vpin * DIV_RATIO;

  if (vout < 0.05f || vout > VCC_SENSOR - 0.05f) {
    oledMsg("SENSOR ERROR", "check MQ-135 and", "divider on GPIO0");
    Serial.printf("SENSOR ERROR: Vpin=%.3f Vout=%.3f\n", vpin, vout);
    delay(5000);
    return;
  }

  float rs   = RL_KOHM * (VCC_SENSOR - vout) / vout;
  float xlog = log10f(rs / R0_KOHM);
  float adc10 = vout / VCC_SENSOR * 1023.0f;   // для сверки со старым Nano

  // 3. Климат (одна повторная попытка)
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (isnan(t) || isnan(h)) {
    delay(2200);
    t = dht.readTemperature();
    h = dht.readHumidity();
  }
  if (isnan(t) || isnan(h)) {
    oledMsg("DHT22 ERROR", "check DATA wire", "GPIO1 + 10k");
    Serial.println("DHT22 read error");
    delay(5000);
    return;
  }

  // 4. Нейросеть -> PPM -> моль/л -> статус
  float logPpm = predictLogPPM(xlog, t, h);
  float ppm = powf(10.0f, logPpm);
  float cm  = (ppm * 1e-6f * P_ATM) / (R_GAS * (t + 273.15f)) / 1000.0f;

  const char *status = (cm < THRESH_WARN) ? "FRESH"
                     : (cm < THRESH_BAD)  ? "EARLY"
                                          : "SPOILED";

  // 5. Итог на OLED и в Serial
  oledResult(ppm, cm, t, h, status);
  Serial.printf("ADC10eq=%.1f Vpin=%.3f Vout=%.3f kept=%d/%d Rs=%.2f Xlog=%.4f "
                "T=%.1f H=%.1f PPM=%.3f Cm=%.3e %s\n",
                adc10, vpin, vout, kept, SAMPLES, rs, xlog, t, h, ppm, cm, status);

  delay(RESULT_HOLD_MS);
}
