#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// Camera pins for ESP32-CAM
#define CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// WiFi credentials
const char* ssid = "YourWiFiSSID";
const char* password = "YourWiFiPassword";

WebServer server(80);

// Function to detect hand gesture
String detectGesture(camera_fb_t *fb) {
  // Convert image to grayscale
  uint8_t *buf = fb->buf;
  int width = fb->width;
  int height = fb->height;
  
  // Simple thresholding
  int threshold = 128;
  int whitePixels = 0;
  int totalPixels = width * height;
  
  for(int i = 0; i < totalPixels; i++) {
    uint8_t r = buf[i * 3];
    uint8_t g = buf[i * 3 + 1];
    uint8_t b = buf[i * 3 + 2];
    uint8_t gray = (r + g + b) / 3;
    
    if(gray > threshold) {
      whitePixels++;
    }
  }
  
  // Calculate percentage of white pixels
  float whitePercentage = (float)whitePixels / totalPixels * 100;
  
  // Simple gesture detection based on white pixel percentage
  if(whitePercentage < 20) {
    return "0"; // Fist
  } else if(whitePercentage < 40) {
    return "1"; // One finger
  } else if(whitePercentage < 60) {
    return "2"; // Two fingers
  } else if(whitePercentage < 80) {
    return "3"; // Three fingers
  } else {
    return "5"; // Open hand
  }
}

void setup() {
  Serial.begin(115200);
  
  // Initialize camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  
  // Setup web server
  server.on("/", HTTP_GET, [](){
    camera_fb_t *fb = esp_camera_fb_get();
    if(!fb) {
      server.send(500, "text/plain", "Camera capture failed");
      return;
    }
    
    String gesture = detectGesture(fb);
    server.send(200, "text/plain", gesture);
    
    esp_camera_fb_return(fb);
  });
  
  server.begin();
}

void loop() {
  server.handleClient();
} 