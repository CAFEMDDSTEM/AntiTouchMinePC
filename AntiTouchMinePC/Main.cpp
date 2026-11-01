
#include "GetSpoofPhoto.h"
#include "Identification.hpp"
#include "ImGuiUI.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <imgui.h>

#ifdef _WIN32
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
#define WM_TRAYICON (WM_USER + 100)
LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if(ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) {
        return true;
    }
    switch(msg) {
        case WM_TRAYICON: {
            switch(LOWORD(lParam)) {
				case WM_LBUTTONUP:
                    ShowWindow(hWnd, SW_SHOW); // 从托盘恢复主窗口
                    SetForegroundWindow(hWnd);
                    break;
                case WM_RBUTTONUP: {
                    POINT pt; GetCursorPos(&pt);
                    HMENU hMenu = CreatePopupMenu();
                    AppendMenu(hMenu, MF_STRING, 1001, L"退出");
                    SetForegroundWindow(hWnd); // 确保菜单能正确显示
                    TrackPopupMenu(hMenu, TPM_RIGHTBUTTON, pt.x, pt.y, 0, hWnd, NULL);
                    DestroyMenu(hMenu);
                    break;
                }  
            }
            return 0;
        }
		case WM_CLOSE:
			ShowWindow(hWnd, SW_HIDE); // 隐藏主窗口到托盘
			return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
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

    // 采集线程写入、推理线程和主线程读取的最新画面，深度固定为 1，新帧直接覆盖旧帧。
    struct FrameBuffer {
        std::mutex mutex;
        std::condition_variable updated;
        cv::Mat frame;
        std::uint64_t sequence = 0;
        bool running = true;
    };

    // 推理线程输出的检测结果，主线程每帧取一次快照。
    struct DetectionOutput {
        std::mutex mutex;
        std::vector<cv::Rect> people;
        std::vector<std::vector<cv::Rect>> faces;
        float processingMilliseconds = 0.0f;
        bool enabled = true;
    };

    // 推理线程使用的参数快照，避免直接读取正在被界面修改的 Settings。
    struct DetectionParams {
        bool enabled = true;
        float confidenceThreshold = 0.4f;
        float nmsThreshold = 0.45f;
        int faceMinNeighbors = 3;
        int detectionIntervalFrames = 2;
    };

    // 抓拍任务和抓拍结果，由抓拍线程异步处理。
    struct CaptureWorker {
        std::mutex mutex;
        std::condition_variable ready;
        cv::Mat frame;
        std::filesystem::path outputDirectory;
        std::string message;
        bool hasRequest = false;
        bool hasMessage = false;
        bool running = true;
    };
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

    // 推理线程独占 OpenCV 线程池，留一个核心给采集和界面线程。
    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    cv::setNumThreads(hardwareThreads > 2 ? static_cast<int>(hardwareThreads - 1) : 2);

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

    cv::VideoCapture camera;

    // MSMF 无法在推流过程中切换分辨率，必须释放后带着分辨率参数重新打开；
    // 同时 CAP_PROP_BUFFERSIZE 不能放进 open 的参数表，否则连分辨率参数一起被忽略。
    const auto openCamera = [&](int width, int height) {
        try {
            camera.release();
            if(!camera.open(cameraIndex, cv::CAP_ANY, {
                    cv::CAP_PROP_FRAME_WIDTH, width,
                    cv::CAP_PROP_FRAME_HEIGHT, height})) {
                return false;
            }
            camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
        } catch(const cv::Exception &) {
            return false;
        }
        return true;
    };

    if(!openCamera(640, 480)) {
        return showStartupError("Could not open camera " + std::to_string(cameraIndex));
    }

    status.cameraReady = true;
    status.message = "检测已就绪";

    const std::filesystem::path outputDirectory("captures");

    // 三线程流水线共享的状态：画面缓冲、检测结果、推理参数和抓拍任务。
    FrameBuffer frameBuffer;
    DetectionOutput detectionOutput;
    CaptureWorker captureWorker;
    std::mutex paramsMutex;
    DetectionParams detectionParams;

    std::atomic<bool> captureRunning(true);
    std::atomic<bool> cameraFailed(false);
    std::atomic<bool> cameraSettingsChanged(false);
    std::atomic<bool> resolutionFallback(false);
    std::atomic<int> requestedWidth(640);
    std::atomic<int> requestedHeight(480);

    // 采集线程独占摄像头，只负责把最新画面写入深度为 1 的缓冲。
    std::thread captureThread([&]() {
        try {
            int consecutiveFailures = 0;
            while(captureRunning.load()) {
                if(cameraSettingsChanged.exchange(false)) {
                    const int width = requestedWidth.load();
                    const int height = requestedHeight.load();
                    // 驱动不支持请求的分辨率时退回 640x480，而不是直接判定摄像头失效。
                    if(!openCamera(width, height)) {
                        resolutionFallback = true;
                        if(!openCamera(640, 480)) {
                            cameraFailed = true;
                            break;
                        }
                    }
                }

                // 每帧重新分配像素，避免覆盖推理线程和主线程仍在引用的画面。
                cv::Mat captured;
                bool readSucceeded = false;
                try {
                    readSucceeded = camera.read(captured) && !captured.empty();
                } catch(const cv::Exception &) {
                    readSucceeded = false;
                }

                if(!readSucceeded) {
                    // 数据流可能已被驱动破坏，重开一次再继续；连续失败才放弃。
                    if(++consecutiveFailures > 3 || !openCamera(requestedWidth.load(), requestedHeight.load())) {
                        cameraFailed = true;
                        break;
                    }
                    continue;
                }
                consecutiveFailures = 0;

                {
                    std::lock_guard<std::mutex> lock(frameBuffer.mutex);
                    frameBuffer.frame = captured;
                    ++frameBuffer.sequence;
                }
                frameBuffer.updated.notify_all();
            }
        } catch(const std::exception & error) {
            std::cerr << "采集线程异常：" << error.what() << std::endl;
            cameraFailed = true;
        }

        {
            std::lock_guard<std::mutex> lock(frameBuffer.mutex);
            frameBuffer.running = false;
        }
        frameBuffer.updated.notify_all();
    });

    // 推理线程等待新画面，按间隔运行 YOLO 和 Haar，把结果交给主线程。
    std::thread inferenceThread([&]() {
        std::uint64_t lastSequence = 0;
        std::uint64_t frameIndex = 0;
        std::vector<cv::Rect> cachedPeople;
        std::vector<std::vector<cv::Rect>> cachedFaces;

        while(true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(frameBuffer.mutex);
                frameBuffer.updated.wait(lock, [&]() {
                    return !frameBuffer.running || frameBuffer.sequence != lastSequence;
                });
                if(!frameBuffer.running) {
                    return;
                }
                lastSequence = frameBuffer.sequence;
                frame = frameBuffer.frame;
            }

            DetectionParams params;
            {
                std::lock_guard<std::mutex> lock(paramsMutex);
                params = detectionParams;
            }

            const auto detectStart = std::chrono::steady_clock::now();
            if(!params.enabled) {
                cachedPeople.clear();
                cachedFaces.clear();
            } else if(frameIndex % std::max(1, params.detectionIntervalFrames) == 0) {
                try {
                    cachedPeople = DetectPeople(
                        personDetector,
                        frame,
                        params.confidenceThreshold,
                        params.nmsThreshold);
                    cachedFaces.clear();
                    cachedFaces.reserve(cachedPeople.size());
                    for(const cv::Rect & person : cachedPeople) {
                        cachedFaces.push_back(FaceDetetction::DetectFacesInPerson(
                            frame,
                            person,
                            faceDetector,
                            1.1,
                            params.faceMinNeighbors,
                            cv::Size(30, 30)));
                    }
                } catch(const std::exception & error) {
                    // 单帧检测失败不能拖垮整个程序，清空结果后继续处理下一帧。
                    std::cerr << "检测异常：" << error.what() << std::endl;
                    cachedPeople.clear();
                    cachedFaces.clear();
                }
            }
            ++frameIndex;

            const float detectMilliseconds = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - detectStart).count();
            std::lock_guard<std::mutex> lock(detectionOutput.mutex);
            detectionOutput.people = cachedPeople;
            detectionOutput.faces = cachedFaces;
            detectionOutput.enabled = params.enabled;
            detectionOutput.processingMilliseconds = detectMilliseconds;
        }
    });

    // 抓拍线程负责截屏和写盘，抓拍时界面不再卡顿。
    std::thread captureWorkerThread([&]() {
        while(true) {
            cv::Mat requestFrame;
            std::filesystem::path requestDirectory;
            {
                std::unique_lock<std::mutex> lock(captureWorker.mutex);
                captureWorker.ready.wait(lock, [&]() {
                    return !captureWorker.running || captureWorker.hasRequest;
                });
                if(!captureWorker.running) {
                    return;
                }
                requestFrame = captureWorker.frame;
                requestDirectory = captureWorker.outputDirectory;
                captureWorker.frame.release();
                captureWorker.hasRequest = false;
            }

            std::string message;
            try {
                const FaceCapture::CaptureResult result =
                    FaceCapture::CaptureFaceEvent(requestFrame, requestDirectory);
                message = result.succeeded()
                    ? "摄像头画面和桌面截图均已保存"
                    : "抓拍失败：摄像头 " + std::string(result.cameraSaved ? "成功" : "失败") +
                        "，桌面 " + (result.desktopSaved ? "成功" : "失败");
            } catch(const std::exception & error) {
                message = std::string("抓拍异常：") + error.what();
            }

            std::lock_guard<std::mutex> lock(captureWorker.mutex);
            captureWorker.message = message;
            captureWorker.hasMessage = true;
        }
    });

    auto lastCapture = std::chrono::steady_clock::time_point{};
    auto previousFrameTime = std::chrono::steady_clock::time_point{};
    bool hasCaptured = false;
    bool previousPerformanceMode = settings.performanceMode;

    // 主线程只做界面渲染和画框，检测与抓拍都在后台线程完成。
    while(ImGuiUI::ProcessMessages()) {
        const auto loopStart = std::chrono::steady_clock::now();
        if(cameraFailed.load()) {
            status.cameraReady = false;
            status.message = "摄像头读取失败";
            std::cerr << status.message << std::endl;
            break;
        }

        // 把界面上的检测参数同步给推理线程。
        {
            std::lock_guard<std::mutex> lock(paramsMutex);
            detectionParams.enabled = settings.detectionEnabled;
            detectionParams.confidenceThreshold = settings.confidenceThreshold;
            detectionParams.nmsThreshold = settings.nmsThreshold;
            detectionParams.faceMinNeighbors = settings.faceMinNeighbors;
            detectionParams.detectionIntervalFrames = settings.detectionIntervalFrames;
        }

        if(previousPerformanceMode != settings.performanceMode) {
            requestedWidth = settings.performanceMode ? 640 : 1280;
            requestedHeight = settings.performanceMode ? 480 : 720;
            cameraSettingsChanged = true;
            previousPerformanceMode = settings.performanceMode;
        }

        if(resolutionFallback.exchange(false)) {
            status.message = "摄像头不支持所选分辨率，已回退到 640x480";
            std::cerr << status.message << std::endl;
        }

        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frameBuffer.mutex);
            frame = frameBuffer.frame;
        }
        status.frameWidth = frame.cols;
        status.frameHeight = frame.rows;

        std::vector<cv::Rect> people;
        std::vector<std::vector<cv::Rect>> faces;
        bool detectionActive = false;
        {
            std::lock_guard<std::mutex> lock(detectionOutput.mutex);
            people = detectionOutput.people;
            faces = detectionOutput.faces;
            detectionActive = detectionOutput.enabled;
            status.processingMilliseconds = detectionOutput.processingMilliseconds;
        }

        cv::Mat displayFrame = frame.empty() ? frame : frame.clone();
        bool faceDetected = false;

        status.personCount = detectionActive ? static_cast<int>(people.size()) : 0;
        status.faceCount = 0;
        if(detectionActive) {
            for(std::size_t index = 0; index < people.size(); ++index) {
                if(settings.drawPeople) {
                    cv::rectangle(displayFrame, people[index], cv::Scalar(255, 180, 0), 2, cv::LINE_AA);
                }
                if(index >= faces.size()) {
                    continue;
                }
                const std::vector<cv::Rect> & personFaces = faces[index];
                status.faceCount += static_cast<int>(personFaces.size());
                if(!personFaces.empty()) {
                    faceDetected = true;
                    if(settings.drawFaces) {
                        FaceDetetction::DrawFaces(displayFrame, personFaces);
                    }
                }
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const bool intervalElapsed = !hasCaptured ||
            now - lastCapture >= std::chrono::seconds(settings.captureIntervalSeconds);
        if(settings.captureEnabled && settings.autoCapture && faceDetected && intervalElapsed) {
            bool requestQueued = false;
            {
                std::lock_guard<std::mutex> lock(captureWorker.mutex);
                if(!captureWorker.hasRequest) {
                    captureWorker.frame = settings.captureWithBoxes ? displayFrame : frame;
                    captureWorker.outputDirectory = outputDirectory;
                    captureWorker.hasRequest = true;
                    requestQueued = true;
                }
            }
            if(requestQueued) {
                captureWorker.ready.notify_one();
                lastCapture = now;
                hasCaptured = true;
            }
        }

        {
            std::lock_guard<std::mutex> lock(captureWorker.mutex);
            if(captureWorker.hasMessage) {
                status.captureMessage = captureWorker.message;
                captureWorker.hasMessage = false;
                std::cout << status.captureMessage << std::endl;
            }
        }

        const auto renderEnd = std::chrono::steady_clock::now();
        if(previousFrameTime != std::chrono::steady_clock::time_point{}) {
            const float frameSeconds = std::chrono::duration<float>(
                renderEnd - previousFrameTime).count();
            status.framesPerSecond = frameSeconds > 0.0f ? 1.0f / frameSeconds : 0.0f;
        }
        previousFrameTime = renderEnd;

        ImGuiUI::Render(settings, status, displayFrame);

        // 窗口隐藏到托盘时 Present 不再限速，这里保证主循环不会空转占满 CPU。
        const float loopMilliseconds = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - loopStart).count();
        if(loopMilliseconds < 8.0f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(8.0f - loopMilliseconds) + 1));
        }
    }

    // 依次停止采集、推理和抓拍线程，再释放摄像头和界面资源。
    captureRunning = false;
    {
        std::lock_guard<std::mutex> lock(frameBuffer.mutex);
        frameBuffer.running = false;
    }
    frameBuffer.updated.notify_all();
    if(captureThread.joinable()) {
        captureThread.join();
    }
    if(inferenceThread.joinable()) {
        inferenceThread.join();
    }

    {
        std::lock_guard<std::mutex> lock(captureWorker.mutex);
        captureWorker.running = false;
    }
    captureWorker.ready.notify_all();
    if(captureWorkerThread.joinable()) {
        captureWorkerThread.join();
    }

    camera.release();
    ImGuiUI::Shutdown();
    return 0;
}

