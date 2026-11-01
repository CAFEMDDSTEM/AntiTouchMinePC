#include "GetSpoofPhoto.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cwchar>
#include <fstream>
#include <string>
#include <vector>

namespace FaceCapture {
	std::string NormalizeExtension(const std::filesystem::path & outputPath) {
        std::string extension = outputPath.extension().string();
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return extension;
	}

    std::wstring MakeTimestamp() {
        SYSTEMTIME localTime{};
        GetLocalTime(&localTime);

        wchar_t timestamp[32]{};
        std::swprintf(
            timestamp,
            sizeof(timestamp) / sizeof(timestamp[0]),
            L"%04u%02u%02u_%02u%02u%02u_%03u",
            localTime.wYear,
            localTime.wMonth,
            localTime.wDay,
            localTime.wHour,
            localTime.wMinute,
            localTime.wSecond,
            localTime.wMilliseconds);

        return timestamp;
    }

    bool SaveMatImage(const cv::Mat & image, const std::filesystem::path & outputPath) {
        // 当前编码流程只接收 8 位深度的图像。
        if(image.empty() || image.depth() != CV_8U) {
            return false;
        }

        // 只允许使用 OpenCV 当前支持的常用图片扩展名。
        const std::string extension = NormalizeExtension(outputPath);
        if(extension != ".jpg" &&
            extension != ".jpeg" &&
            extension != ".png" &&
            extension != ".bmp") {
            return false;
        }

        // JPEG 不支持四通道，因此先把 BGRA 转换成 BGR。
        cv::Mat imageForEncoding;

        if(image.channels() == 4) {
            cv::cvtColor(image, imageForEncoding, cv::COLOR_BGRA2BGR);
        } else if(image.channels() == 1 || image.channels() == 3) {
            imageForEncoding = image;
        } else {
            return false;
        }

        // 先在内存中完成图片编码，再使用文件流写入。
        // 这样可以正确支持包含中文字符的 Windows 输出路径。
        std::vector<unsigned char> encoded;
        if(!cv::imencode(extension, imageForEncoding, encoded)) {
            return false;
        }

        std::ofstream output(outputPath, std::ios::binary);
        if(!output) {
            return false;
        }

        // 以二进制方式写入完整的编码数据。
        output.write(
            reinterpret_cast<const char *>(encoded.data()),
            static_cast<std::streamsize>(encoded.size()));

        return output.good();
    }

    bool CaptureDesktopToMat(cv::Mat & desktopImage) {
        // 每次截图前释放上一次的结果，避免调用失败时留下旧图像。
        desktopImage.release();

        // 获取所有显示器组成虚拟屏幕的位置和尺寸。
        const int left = GetSystemMetrics(SM_XVIRTUALSCREEN);
        const int top = GetSystemMetrics(SM_YVIRTUALSCREEN);
        const int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        const int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);

        if(width <= 0 || height <= 0) {
            return false;
        }

        // 获取整个屏幕的设备上下文。
        HDC screenDc = GetDC(nullptr);
        if(screenDc == nullptr) {
            return false;
        }

        // 创建与屏幕兼容的内存设备上下文。
        HDC memoryDc = CreateCompatibleDC(screenDc);
        if(memoryDc == nullptr) {
            ReleaseDC(nullptr, screenDc);
            return false;
        }

        // 创建用于保存屏幕像素的兼容位图。
        HBITMAP bitmap =
            CreateCompatibleBitmap(screenDc, width, height);
        if(bitmap == nullptr) {
            DeleteDC(memoryDc);
            ReleaseDC(nullptr, screenDc);
            return false;
        }

        // 把新位图选进内存设备上下文，后续绘制会写入这个位图。
        HGDIOBJ previousBitmap = SelectObject(memoryDc, bitmap);
        if(previousBitmap == nullptr ||
            previousBitmap == HGDI_ERROR) {
            DeleteObject(bitmap);
            DeleteDC(memoryDc);
            ReleaseDC(nullptr, screenDc);
            return false;
        }

        // 将虚拟桌面像素复制到内存位图。
        // CAPTUREBLT 用于尽量包含具有分层效果的窗口。
        const BOOL copied = BitBlt(
            memoryDc,
            0,
            0,
            width,
            height,
            screenDc,
            left,
            top,
            SRCCOPY | CAPTUREBLT);

        // 获取像素数据前先恢复原来的位图。
        SelectObject(memoryDc, previousBitmap);
        bool succeeded = false;

        if(copied) {
            // 将位图高度设为负数，要求 GetDIBits 返回自上而下的像素。
            BITMAPINFO bitmapInfo{};
            bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bitmapInfo.bmiHeader.biWidth = width;
            bitmapInfo.bmiHeader.biHeight = -height;
            bitmapInfo.bmiHeader.biPlanes = 1;
            bitmapInfo.bmiHeader.biBitCount = 32;
            bitmapInfo.bmiHeader.biCompression = BI_RGB;
            bitmapInfo.bmiHeader.biSizeImage =
                static_cast<DWORD>(width) *
                static_cast<DWORD>(height) *
                4;

            std::vector<unsigned char> pixels(
                static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height) *
                4);

            // 将 GDI 位图转换为 32 位 BGRA 像素数据。
            const int copiedLines = GetDIBits(
                screenDc,
                bitmap,
                0,
                static_cast<UINT>(height),
                pixels.data(),
                &bitmapInfo,
                DIB_RGB_COLORS);

            if(copiedLines == height) {
                // GDI 返回的数据包含无用的 Alpha 通道，因此转换为三通道 BGR。
                const cv::Mat bgra(
                    height,
                    width,
                    CV_8UC4,
                    pixels.data());

                cv::cvtColor(bgra, desktopImage, cv::COLOR_BGRA2BGR);
                succeeded = !desktopImage.empty();
            }
        }

        DeleteObject(bitmap);
        DeleteDC(memoryDc);
        ReleaseDC(nullptr, screenDc);
        return succeeded;
    }

    CaptureResult CaptureFaceEvent(const cv::Mat & cameraFrame, const std::filesystem::path & outputDirectory) {
        CaptureResult result;

        // 创建抓拍输出目录；目录已经存在时不会报错。
        std::error_code directoryError;
        std::filesystem::create_directories(
            outputDirectory,
            directoryError);

        if(directoryError) {
            return result;
        }

        // 同一次抓拍使用相同时间戳，方便把两张图片对应起来。
        const std::wstring timestamp = MakeTimestamp();

        // 摄像头图片和桌面图片分别使用独立后缀。
        result.cameraPath =
            outputDirectory /
            std::filesystem::path(timestamp + L"_camera.png");
        result.desktopPath =
            outputDirectory / 
            std::filesystem::path(timestamp + L"_desktop.png");

        // 直接保存当前摄像头 cv::Mat，不需要重新打开摄像头。
        result.cameraSaved =
            SaveMatImage(cameraFrame, result.cameraPath);

        // 先通过 GDI 把桌面截图读入 cv::Mat，再保存为图片文件。
        cv::Mat desktopImage;
        if(CaptureDesktopToMat(desktopImage)) {
            result.desktopSaved =
                SaveMatImage(desktopImage, result.desktopPath);
        }

        return result;
    }
}