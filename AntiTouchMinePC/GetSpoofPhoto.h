#pragma once
#ifndef GETSPOOFPHOTO_H
#define GETSPOOFPHOTO_H
#include <Windows.h>

#include <opencv2/core.hpp>

#include <filesystem>
namespace FaceCapture {
    // 描述一次人脸事件抓拍中，摄像头图片和桌面图片的保存结果。
    struct CaptureResult {
        // 摄像头画面是否保存成功。
        bool cameraSaved = false;

        // 桌面截图是否保存成功。
        bool desktopSaved = false;

        // 摄像头图片的完整输出路径。
        std::filesystem::path cameraPath;

        // 桌面图片的完整输出路径。
        std::filesystem::path desktopPath;

        // 只有摄像头画面和桌面截图都保存成功时才返回 true。
        bool succeeded() const {
            return cameraSaved && desktopSaved;
        }
    };

    bool SaveMatImage(const cv::Mat & image, const std::filesystem::path & outputPath);

    bool CaptureDesktopToMat(cv::Mat & desktopImage);

    CaptureResult CaptureFaceEvent(const cv::Mat & cameraFrame, const std::filesystem::path & outputDirectory = std::filesystem::path(L"captures"));
}
#endif // !GETSPOOFPHOTO_H
