#pragma once
#ifndef IMGUIUI_HPP
#define IMGUIUI_HPP

#include <Windows.h>
#include <opencv2/core.hpp>

#include <string>

namespace ImGuiUI {
	struct Settings {
		bool detectionEnabled = true;
		bool drawPeople = true;
		bool drawFaces = true;
		bool captureEnabled = false;
		bool captureWithBoxes = true;
		bool autoCapture = true;
		bool performanceMode = true;
		float confidenceThreshold = 0.4f;
		float nmsThreshold = 0.45f;
		int faceMinNeighbors = 3;
		int captureIntervalSeconds = 5;
		int detectionIntervalFrames = 2;
	};

	struct Status {
		bool cameraReady = false;
		int personCount = 0;
		int faceCount = 0;
		int frameWidth = 0;
		int frameHeight = 0;
		float framesPerSecond = 0.0f;
		float processingMilliseconds = 0.0f;
		std::string message = "正在初始化";
		std::string captureMessage = "尚未抓拍";
	};

	bool Initialize(HINSTANCE instance);
	bool ProcessMessages();
	void Render(Settings & settings, const Status & status, const cv::Mat & cameraFrame);
	void Shutdown();
}

#endif // !IMGUIUI_HPP
