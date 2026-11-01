
#include "GetSpoofPhoto.h"
#include "Identification.hpp"
#include "ImGuiUI.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#ifdef _WIN32
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
#define WM_TRAYICON (WM_USER + 100)
LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if(ImGui_ImplWin32_WndProcHandler) {

    }
}
#endif

namespace {
    // YOLOv5n 的输入尺寸、COCO 人类别编号和检测阈值。
    constexpr int modelInputSize = 640;
    constexpr int personClassId = 0;

    // 将模型输出转换为原始画面中的人物矩形框。
    std::vector<cv::Rect> DetectPeople(
        cv::dnn::Net & detector,
        const cv::Mat & frame,
        float confidenceThreshold,
        float nmsThreshold) {
        std::vector<cv::Rect> people;
        if(frame.empty()) {
            return people;
        }

        // 缩放并归一化输入画面，转换成 YOLO 所需的 NCHW 张量。
        cv::Mat blob;
        cv::dnn::blobFromImage(
            frame,
            blob,
            1.0 / 255.0,
            cv::Size(modelInputSize, modelInputSize),
            cv::Scalar(),
            true,
            false);
        detector.setInput(blob);

        std::vector<cv::Mat> outputs;
        detector.forward(outputs, detector.getUnconnectedOutLayersNames());
        if(outputs.empty()) {
            return people;
        }

        // YOLOv5 常见输出形状为 [1, 25200, 85]，也兼容转置后的排列。
        const cv::Mat & output = outputs.front();
        int rowCount = 0;
        int columnCount = 0;
        if(output.dims == 3) {
            rowCount = output.size[1];
            columnCount = output.size[2];
        } else if(output.dims == 2) {
            rowCount = output.rows;
            columnCount = output.cols;
        } else {
            return people;
        }

        cv::Mat detections;
        if(columnCount == 85) {
            detections = output.reshape(1, rowCount);
        } else if(rowCount == 85) {
            detections = output.reshape(1, rowCount).t();
        } else {
            std::cerr << "Unsupported YOLOv5 output shape: "
                << rowCount << " x " << columnCount << std::endl;
            return people;
        }

        // 将模型坐标按输入尺寸映射回摄像头原图坐标。
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        const float scaleX = static_cast<float>(frame.cols) / modelInputSize;
        const float scaleY = static_cast<float>(frame.rows) / modelInputSize;

        for(int index = 0; index < detections.rows; ++index) {
            const float * values = detections.ptr<float>(index);
            const float confidence = values[4] * values[5 + personClassId];
            if(confidence < confidenceThreshold) {
                continue;
            }

            const int centerX = cvRound(values[0] * scaleX);
            const int centerY = cvRound(values[1] * scaleY);
            const int width = cvRound(values[2] * scaleX);
            const int height = cvRound(values[3] * scaleY);
            const cv::Rect box(
                centerX - width / 2,
                centerY - height / 2,
                width,
                height);
            const cv::Rect clippedBox = box & cv::Rect(0, 0, frame.cols, frame.rows);
            if(clippedBox.width > 0 && clippedBox.height > 0) {
                boxes.push_back(clippedBox);
                confidences.push_back(confidence);
            }
        }

        // 使用非极大值抑制合并同一人物的重复检测框。
        std::vector<int> selectedIndexes;
        cv::dnn::NMSBoxes(
            boxes,
            confidences,
            confidenceThreshold,
            nmsThreshold,
            selectedIndexes);
        people.reserve(selectedIndexes.size());
        for(const int index : selectedIndexes) {
            people.push_back(boxes[index]);
        }
        return people;
    }
}



int main(int argc, char * argv[]) {
    // 命令行参数依次为 YOLO 模型、Haar 人脸模型和摄像头编号。
    const std::filesystem::path modelPath = argc > 1 ? argv[1] : "yolov5n.onnx";
    const std::filesystem::path faceModelPath =
        argc > 2 ? argv[2] : "haarcascade_frontalface_default.xml";

    ImGuiUI::Settings settings;
    ImGuiUI::Status status;
    if(!ImGuiUI::Initialize(GetModuleHandleW(nullptr))) {
        return 1;
    }

    // 初始化失败时保留界面，便于直接查看错误信息。
    const auto showStartupError = [&](const std::string & message) {
        status.cameraReady = false;
        status.message = message;
        std::cerr << message << std::endl;
        const cv::Mat emptyFrame;
        while(ImGuiUI::ProcessMessages()) {
            ImGuiUI::Render(settings, status, emptyFrame);
        }
        ImGuiUI::Shutdown();
        return 1;
    };

    int cameraIndex = 0;
    try {
        cameraIndex = argc > 3 ? std::stoi(argv[3]) : 0;
    } catch(const std::exception & error) {
        return showStartupError(std::string("Invalid camera index: ") + error.what());
    }

    if(!std::filesystem::exists(modelPath)) {
        return showStartupError("YOLO model file not found: " + modelPath.string());
    }
    if(!std::filesystem::exists(faceModelPath)) {
        return showStartupError("Haar face model file not found: " + faceModelPath.string());
    }

    // 限制 OpenCV CPU 线程数，避免 DNN 与预览渲染争抢所有核心。
    cv::setNumThreads(2);

    // 加载 YOLO 人体检测模型；加载失败时输出原因并退出。
    cv::dnn::Net personDetector;
    try {
        personDetector = cv::dnn::readNetFromONNX(modelPath.string());
    } catch(const cv::Exception & error) {
        return showStartupError(std::string("Failed to load YOLO model: ") + error.what());
    }
    if(personDetector.empty()) {
        return showStartupError("Failed to load YOLO model: " + modelPath.string());
    }
    personDetector.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    personDetector.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 加载 Haar 分类器，用于在人框区域内进一步检测人脸。
    cv::CascadeClassifier faceDetector;
    if(!FaceDetetction::LoadFaceDetector(faceDetector, faceModelPath.string())) {
        return showStartupError("Failed to load Haar face model: " + faceModelPath.string());
    }

    cv::VideoCapture camera(cameraIndex);
    if(!camera.isOpened()) {
        return showStartupError("Could not open camera " + std::to_string(cameraIndex));
    }

    // 降低采集分辨率并限制摄像头缓冲，减少每帧推理量和画面延迟。
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    camera.set(cv::CAP_PROP_BUFFERSIZE, 1);

    status.cameraReady = true;
    status.message = "检测已就绪";

    const std::filesystem::path outputDirectory("captures");
    auto lastCapture = std::chrono::steady_clock::time_point{};
    auto previousFrameTime = std::chrono::steady_clock::time_point{};
    bool hasCaptured = false;
    bool previousPerformanceMode = settings.performanceMode;
    int frameIndex = 0;
    std::vector<cv::Rect> cachedPeople;
    std::vector<std::vector<cv::Rect>> cachedFaces;
    cv::Mat frame;

    // 在 ImGui 窗口消息循环中采集画面、检测并刷新调试界面。
    while(ImGuiUI::ProcessMessages()) {
        if(!camera.read(frame)) {
            status.cameraReady = false;
            status.message = "摄像头读取失败";
            std::cerr << status.message << std::endl;
            break;
        }
        if(frame.empty()) {
            continue;
        }

        const auto frameStart = std::chrono::steady_clock::now();
        const bool performanceModeChanged = previousPerformanceMode != settings.performanceMode;
        if(performanceModeChanged) {
            const double targetWidth = settings.performanceMode ? 640.0 : 1280.0;
            const double targetHeight = settings.performanceMode ? 480.0 : 720.0;
            camera.set(cv::CAP_PROP_FRAME_WIDTH, targetWidth);
            camera.set(cv::CAP_PROP_FRAME_HEIGHT, targetHeight);
            previousPerformanceMode = settings.performanceMode;
        }

        status.frameWidth = frame.cols;
        status.frameHeight = frame.rows;
        cv::Mat displayFrame = frame.clone();
        bool faceDetected = false;

        if(!settings.detectionEnabled) {
            cachedPeople.clear();
            cachedFaces.clear();
            status.personCount = 0;
            status.faceCount = 0;
        } else {
            const bool shouldDetect = performanceModeChanged ||
                frameIndex % settings.detectionIntervalFrames == 0;
            if(shouldDetect) {
                cachedPeople = DetectPeople(
                    personDetector,
                    frame,
                    settings.confidenceThreshold,
                    settings.nmsThreshold);
                cachedFaces.clear();
                cachedFaces.reserve(cachedPeople.size());
                for(const cv::Rect & person : cachedPeople) {
                    cachedFaces.push_back(FaceDetetction::DetectFacesInPerson(
                        frame,
                        person,
                        faceDetector,
                        1.1,
                        settings.faceMinNeighbors,
                        cv::Size(30, 30)));
                }
            }

            status.personCount = static_cast<int>(cachedPeople.size());
            status.faceCount = 0;
            for(std::size_t index = 0; index < cachedPeople.size(); ++index) {
                if(settings.drawPeople) {
                    cv::rectangle(displayFrame, cachedPeople[index], cv::Scalar(255, 180, 0), 2, cv::LINE_AA);
                }
                const std::vector<cv::Rect> & faces = cachedFaces[index];
                status.faceCount += static_cast<int>(faces.size());
                if(!faces.empty()) {
                    faceDetected = true;
                    if(settings.drawFaces) {
                        FaceDetetction::DrawFaces(displayFrame, faces);
                    }
                }
            }
        }
        ++frameIndex;

        const auto now = std::chrono::steady_clock::now();
        const bool intervalElapsed = !hasCaptured ||
            now - lastCapture >= std::chrono::seconds(settings.captureIntervalSeconds);
        if(settings.captureEnabled &&
            settings.autoCapture &&
            faceDetected &&
            intervalElapsed) {
            const FaceCapture::CaptureResult result =
                FaceCapture::CaptureFaceEvent(settings.captureWithBoxes ? displayFrame : frame, outputDirectory);
            lastCapture = now;
            hasCaptured = true;
            status.captureMessage = result.succeeded()
                ? "摄像头画面和桌面截图均已保存"
                : "抓拍失败：摄像头 " + std::string(result.cameraSaved ? "成功" : "失败") +
                    "，桌面 " + (result.desktopSaved ? "成功" : "失败");
            std::cout << status.captureMessage << std::endl;
        }

        const auto frameEnd = std::chrono::steady_clock::now();
        status.processingMilliseconds = std::chrono::duration<float, std::milli>(
            frameEnd - frameStart).count();
        if(previousFrameTime != std::chrono::steady_clock::time_point{}) {
            const float frameSeconds = std::chrono::duration<float>(
                frameEnd - previousFrameTime).count();
            status.framesPerSecond = frameSeconds > 0.0f ? 1.0f / frameSeconds : 0.0f;
        }
        previousFrameTime = frameEnd;

        ImGuiUI::Render(settings, status, displayFrame);
    }

    camera.release();
    ImGuiUI::Shutdown();
    return 0;
}

