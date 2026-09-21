// XIAO ESP32S3 Sense - hosts its own WiFi AP and streams the camera as MJPEG
// MacBook connects directly to the ESP32's WiFi network (no router needed)
// Stream URL: http://192.168.4.1/stream

#include <WiFi.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <esp_http_server.h>

// TB6612FNG wiring on the XIAO ESP32S3 Sense exposed GPIOs.
#ifndef ROBOT_LEFT_IN1   // AN1
#define ROBOT_LEFT_IN1 1 // D0
#endif
#ifndef ROBOT_LEFT_IN2   // AN2
#define ROBOT_LEFT_IN2 2 // D1
#endif
#ifndef ROBOT_LEFT_PWM   // PWMA
#define ROBOT_LEFT_PWM 3 // D2
#endif
#ifndef ROBOT_RIGHT_IN1   // BIN1
#define ROBOT_RIGHT_IN1 4 // D3
#endif
#ifndef ROBOT_RIGHT_IN2   // BIN2
#define ROBOT_RIGHT_IN2 5 // D4
#endif
#ifndef ROBOT_RIGHT_PWM   // PWMB
#define ROBOT_RIGHT_PWM 6 // D5
#endif
#ifndef ROBOT_STBY   // STBY
#define ROBOT_STBY 9 // D10
#endif

WiFiServer commandServer(8080);
String lastRobotCommand = "stop";
int lastRobotSpeed = 0;

// ---- EDIT THESE ----
const char *AP_SSID = "XIAO-ESP32S3";
const char *AP_PASS = "12345678"; // min 8 chars, or "" for open network
// ---------------------

// XIAO ESP32S3 Sense camera pin map
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39
#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13

static httpd_handle_t stream_httpd = nullptr;

constexpr size_t MAX_STREAM_JPEG_SIZE = 350 * 1024;
constexpr TickType_t STREAM_FRAME_WAIT = pdMS_TO_TICKS(250);
static uint8_t *latestJpeg = nullptr;
static size_t latestJpegLength = 0;
static uint32_t latestJpegSequence = 0;
static SemaphoreHandle_t frameMutex = nullptr;
static TaskHandle_t cameraTaskHandle = nullptr;
static TaskHandle_t commandTaskHandle = nullptr;

static const char *STREAM_BOUNDARY = "\r\n--frame\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static void camera_capture_task(void *parameter)
{
    (void)parameter;
    for (;;)
    {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb != nullptr)
        {
            if (fb->len <= MAX_STREAM_JPEG_SIZE &&
                xSemaphoreTake(frameMutex, pdMS_TO_TICKS(100)) == pdTRUE)
            {
                memcpy(latestJpeg, fb->buf, fb->len);
                latestJpegLength = fb->len;
                latestJpegSequence++;
                xSemaphoreGive(frameMutex);
            }
            esp_camera_fb_return(fb);
        }
        else
        {
            Serial.println("Camera capture failed");
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    esp_err_t res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
    if (res != ESP_OK)
        return res;

    char part_buf[64];
    uint32_t sentSequence = 0;

    while (true)
    {
        size_t jpegLength = 0;
        uint32_t jpegSequence = 0;
        if (xSemaphoreTake(frameMutex, STREAM_FRAME_WAIT) != pdTRUE)
        {
            continue;
        }

        if (latestJpegLength == 0 || latestJpegSequence == sentSequence)
        {
            xSemaphoreGive(frameMutex);
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        jpegLength = latestJpegLength;
        jpegSequence = latestJpegSequence;
        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (res == ESP_OK)
        {
            size_t headerLength = snprintf(part_buf, sizeof(part_buf), STREAM_PART, jpegLength);
            res = httpd_resp_send_chunk(req, part_buf, headerLength);
        }
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char *)latestJpeg, jpegLength);
        xSemaphoreGive(frameMutex);
        sentSequence = jpegSequence;
        if (res != ESP_OK)
            break;
    }
    return res;
}

static void start_stream_server()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    httpd_uri_t stream_uri = {
        .uri = "/stream",
        .method = HTTP_GET,
        .handler = stream_handler,
        .user_ctx = nullptr};

    if (httpd_start(&stream_httpd, &config) == ESP_OK)
    {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
    }
}

static void stop_robot()
{
    lastRobotCommand = "stop";
    lastRobotSpeed = 0;

    analogWrite(ROBOT_LEFT_PWM, 0);
    analogWrite(ROBOT_RIGHT_PWM, 0);
    digitalWrite(ROBOT_LEFT_IN1, LOW);
    digitalWrite(ROBOT_LEFT_IN2, LOW);
    digitalWrite(ROBOT_RIGHT_IN1, LOW);
    digitalWrite(ROBOT_RIGHT_IN2, LOW);
    digitalWrite(ROBOT_STBY, LOW);
}

static void set_motor(int in1, int in2, int pwm, bool forward, int speed)
{
    if (speed == 0)
    {
        analogWrite(pwm, 0);
        digitalWrite(in1, LOW);
        digitalWrite(in2, LOW);
        return;
    }

    digitalWrite(in1, forward ? HIGH : LOW);
    digitalWrite(in2, forward ? LOW : HIGH);
    analogWrite(pwm, speed);
}

static void move_robot(String command, int speed)
{
    speed = constrain(speed, 0, 255);
    lastRobotCommand = command;
    lastRobotSpeed = speed;

    digitalWrite(ROBOT_STBY, HIGH);

    if (command == "w")
    {
        Serial.printf("Robot cmd: forward speed=%d\n", speed);
        set_motor(ROBOT_LEFT_IN1, ROBOT_LEFT_IN2, ROBOT_LEFT_PWM, true, speed);
        set_motor(ROBOT_RIGHT_IN1, ROBOT_RIGHT_IN2, ROBOT_RIGHT_PWM, true, speed);
    }
    else if (command == "s")
    {
        Serial.printf("Robot cmd: backward speed=%d\n", speed);
        set_motor(ROBOT_LEFT_IN1, ROBOT_LEFT_IN2, ROBOT_LEFT_PWM, false, speed);
        set_motor(ROBOT_RIGHT_IN1, ROBOT_RIGHT_IN2, ROBOT_RIGHT_PWM, false, speed);
    }
    else if (command == "a")
    {
        Serial.printf("Robot cmd: tank left turn speed=%d\n", speed);
        set_motor(ROBOT_LEFT_IN1, ROBOT_LEFT_IN2, ROBOT_LEFT_PWM, false, speed);
        set_motor(ROBOT_RIGHT_IN1, ROBOT_RIGHT_IN2, ROBOT_RIGHT_PWM, true, speed);
    }
    else if (command == "d")
    {
        Serial.printf("Robot cmd: tank right turn speed=%d\n", speed);
        set_motor(ROBOT_LEFT_IN1, ROBOT_LEFT_IN2, ROBOT_LEFT_PWM, true, speed);
        set_motor(ROBOT_RIGHT_IN1, ROBOT_RIGHT_IN2, ROBOT_RIGHT_PWM, false, speed);
    }
    else if (command == "p")
    {
        Serial.printf("Robot cmd: stop\n");
        stop_robot();
    }
}

static void process_command(String raw)
{
    raw.trim();
    if (raw.length() == 0)
        return;

    String command = raw;
    int speedValue = 100;

    int commaIndex = raw.indexOf(',');
    if (commaIndex != -1)
    {
        command = raw.substring(0, commaIndex);
        command.trim();
        String speedText = raw.substring(commaIndex + 1);
        speedText.trim();
        if (speedText.length() > 0)
            speedValue = speedText.toInt();
    }

    command.toLowerCase();
    if (command == "w" || command == "s" || command == "a" || command == "d" || command == "p")
    {
        move_robot(command, speedValue);
    }
    else
    {
        Serial.printf("Unknown command: %s\n", raw.c_str());
    }
}

static void command_receive_task(void *parameter)
{
    (void)parameter;
    for (;;)
    {
        WiFiClient client = commandServer.available();
        if (!client)
        {
            vTaskDelay(pdMS_TO_TICKS(2));
            continue;
        }

        String line;
        unsigned long readDeadline = millis() + 100;
        while (client.connected() && millis() < readDeadline)
        {
            if (!client.available())
            {
                vTaskDelay(pdMS_TO_TICKS(1));
                continue;
            }

            char c = client.read();
            if (c == '\n' || c == '\r')
            {
                if (line.length() > 0)
                {
                    process_command(line);
                    line = "";
                }
            }
            else if (line.length() < 32)
            {
                line += c;
            }
            readDeadline = millis() + 100;
        }

        if (line.length() > 0)
            process_command(line);
        client.stop();
    }
}

static bool init_camera()
{
    camera_config_t config = {};
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
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound())
    {
        config.frame_size = FRAMESIZE_VGA; // 640x480
        config.jpeg_quality = 12;          // lower number = higher quality
        config.fb_count = 2;
    }
    else
    {
        config.frame_size = FRAMESIZE_QVGA; // 320x240
        config.jpeg_quality = 15;
        config.fb_count = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
    {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }
    return true;
}

void setup()
{
    Serial.begin(115200);

    if (!init_camera())
    {
        Serial.println("Camera init failed, halting.");
        while (true)
            yield();
    }

    WiFi.mode(WIFI_AP);
    bool ok = WiFi.softAP(AP_SSID, AP_PASS);
    if (!ok)
    {
        Serial.println("Failed to start AP.");
        while (true)
            yield();
    }

    Serial.println("AP started.");
    Serial.print("SSID: ");
    Serial.println(AP_SSID);
    Serial.print("Stream URL: http://");
    Serial.print(WiFi.softAPIP()); // default: 192.168.4.1
    Serial.println("/stream");
    Serial.println("Robot command TCP server: tcp://192.168.4.1:8080");
    Serial.println("Commands: w/s/a/d/p, speed 0-255");

    pinMode(ROBOT_LEFT_IN1, OUTPUT);
    pinMode(ROBOT_LEFT_IN2, OUTPUT);
    pinMode(ROBOT_LEFT_PWM, OUTPUT);
    pinMode(ROBOT_RIGHT_IN1, OUTPUT);
    pinMode(ROBOT_RIGHT_IN2, OUTPUT);
    pinMode(ROBOT_RIGHT_PWM, OUTPUT);
    pinMode(ROBOT_STBY, OUTPUT);

    stop_robot();
    frameMutex = xSemaphoreCreateMutex();
    latestJpeg = static_cast<uint8_t *>(heap_caps_malloc(
        MAX_STREAM_JPEG_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (latestJpeg == nullptr)
    {
        latestJpeg = static_cast<uint8_t *>(heap_caps_malloc(
            MAX_STREAM_JPEG_SIZE, MALLOC_CAP_8BIT));
    }
    if (frameMutex == nullptr || latestJpeg == nullptr)
    {
        Serial.println("Failed to allocate stream resources.");
        while (true)
            yield();
    }

    xTaskCreatePinnedToCore(
        camera_capture_task, "camera_capture", 4096, nullptr, 2,
        &cameraTaskHandle, 0);
    start_stream_server();
    commandServer.begin();
    xTaskCreatePinnedToCore(
        command_receive_task, "command_receive", 4096, nullptr, 3,
        &commandTaskHandle, 1);
}

void loop()
{
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 5000)
    {
        lastPrint = millis();
        Serial.printf("Connected stations: %d\n", WiFi.softAPgetStationNum());
    }
    vTaskDelay(pdMS_TO_TICKS(100));
}