# ESP32 Sign Language Recognition

This project implements a simple sign language recognition system using ESP32-CAM. It can recognize basic hand gestures and numbers.

## Hardware Requirements

1. ESP32-CAM module
2. USB to TTL converter (for programming)
3. Jumper wires
4. Power supply (5V)

## Software Requirements

1. Arduino IDE
2. ESP32 board support package
3. Required libraries:
   - esp_camera.h
   - WiFi.h
   - WebServer.h

## Setup Instructions

1. Install ESP32 board support in Arduino IDE:
   - Open Arduino IDE
   - Go to File > Preferences
   - Add `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json` to Additional Boards Manager URLs
   - Go to Tools > Board > Boards Manager
   - Search for "esp32" and install

2. Install required libraries:
   - Go to Sketch > Include Library > Manage Libraries
   - Search for and install:
     - ESP32 Camera
     - WebServer

3. Configure the code:
   - Open `esp32_sign_language.ino`
   - Update WiFi credentials:
     ```cpp
     const char* ssid = "YourWiFiSSID";
     const char* password = "YourWiFiPassword";
     ```

4. Upload the code:
   - Select board: "ESP32 Wrover Module"
   - Select port
   - Click Upload

## Usage

1. Power up the ESP32-CAM
2. Wait for WiFi connection (check Serial Monitor)
3. Note the IP address shown in Serial Monitor
4. Open a web browser and go to `http://[ESP32_IP]`
5. Show hand gestures to the camera
6. The webpage will display the recognized gesture

## Recognized Gestures

The system can recognize:
- 0: Fist
- 1: One finger
- 2: Two fingers
- 3: Three fingers
- 5: Open hand

## Tips for Better Recognition

1. Ensure good lighting
2. Keep hand in center of frame
3. Make clear, distinct gestures
4. Hold each gesture steady
5. Keep background simple and uniform

## Troubleshooting

1. If camera fails to initialize:
   - Check power supply (needs 5V)
   - Verify all connections
   - Check camera module compatibility

2. If WiFi connection fails:
   - Verify WiFi credentials
   - Check signal strength
   - Ensure 2.4GHz network

3. If gesture recognition is poor:
   - Adjust lighting
   - Modify threshold values in code
   - Ensure hand fills most of frame

## Limitations

1. Basic gesture recognition only
2. Requires good lighting
3. Limited to simple hand shapes
4. May need calibration for different environments

## Future Improvements

1. Add more gesture recognition
2. Implement machine learning
3. Add visual feedback
4. Improve accuracy
5. Add gesture history 