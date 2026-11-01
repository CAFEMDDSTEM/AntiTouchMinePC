#include "ImGuiUI.hpp"

#include <d3d11.h>
#include <dxgi.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <iostream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
	HWND window,
	UINT message,
	WPARAM wParam,
	LPARAM lParam);

namespace {
	ID3D11Device * device = nullptr;
	ID3D11DeviceContext * deviceContext = nullptr;
	IDXGISwapChain * swapChain = nullptr;
	ID3D11RenderTargetView * renderTarget = nullptr;
	ID3D11Texture2D * previewTexture = nullptr;
	ID3D11ShaderResourceView * previewView = nullptr;
	UINT resizeWidth = 0;
	UINT resizeHeight = 0;
	HWND mainWindow = nullptr;
	bool windowOpen = true;

	void CleanupRenderTarget() {
		if(renderTarget != nullptr) {
			renderTarget->Release();
			renderTarget = nullptr;
		}
	}

	void CreateRenderTarget() {
		ID3D11Texture2D * backBuffer = nullptr;
		if(SUCCEEDED(swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) {
			device->CreateRenderTargetView(backBuffer, nullptr, &renderTarget);
			backBuffer->Release();
		}
	}

	void CleanupPreviewTexture() {
		if(previewView != nullptr) {
			previewView->Release();
			previewView = nullptr;
		}
		if(previewTexture != nullptr) {
			previewTexture->Release();
			previewTexture = nullptr;
		}
	}

	bool CreateDeviceD3D(HWND window) {
		DXGI_SWAP_CHAIN_DESC swapChainDescription{};
		swapChainDescription.BufferCount = 2;
		swapChainDescription.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDescription.BufferDesc.RefreshRate.Numerator = 60;
		swapChainDescription.BufferDesc.RefreshRate.Denominator = 1;
		swapChainDescription.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
		swapChainDescription.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDescription.OutputWindow = window;
		swapChainDescription.SampleDesc.Count = 1;
		swapChainDescription.Windowed = TRUE;
		swapChainDescription.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

		const D3D_FEATURE_LEVEL featureLevels[] = {
			D3D_FEATURE_LEVEL_11_0,
			D3D_FEATURE_LEVEL_10_0
		};
		D3D_FEATURE_LEVEL selectedFeatureLevel{};
		const HRESULT result = D3D11CreateDeviceAndSwapChain(
			nullptr,
			D3D_DRIVER_TYPE_HARDWARE,
			nullptr,
			0,
			featureLevels,
			ARRAYSIZE(featureLevels),
			D3D11_SDK_VERSION,
			&swapChainDescription,
			&swapChain,
			&device,
			&selectedFeatureLevel,
			&deviceContext);
		if(FAILED(result)) {
			return false;
		}

		CreateRenderTarget();
		return renderTarget != nullptr;
	}

	void CleanupDeviceD3D() {
		CleanupPreviewTexture();
		CleanupRenderTarget();
		if(swapChain != nullptr) {
			swapChain->Release();
			swapChain = nullptr;
		}
		if(deviceContext != nullptr) {
			deviceContext->Release();
			deviceContext = nullptr;
		}
		if(device != nullptr) {
			device->Release();
			device = nullptr;
		}
	}

	bool UpdatePreviewTexture(const cv::Mat & frame) {
		if(frame.empty()) {
			return false;
		}

		cv::Mat bgraFrame;
		if(frame.channels() == 3) {
			cv::cvtColor(frame, bgraFrame, cv::COLOR_BGR2BGRA);
		} else if(frame.channels() == 4) {
			bgraFrame = frame;
		} else {
			return false;
		}

		if(previewTexture != nullptr) {
			D3D11_TEXTURE2D_DESC currentDescription{};
			previewTexture->GetDesc(&currentDescription);
			if(currentDescription.Width != static_cast<UINT>(bgraFrame.cols) ||
				currentDescription.Height != static_cast<UINT>(bgraFrame.rows)) {
				CleanupPreviewTexture();
			}
		}

		if(previewTexture == nullptr) {
			D3D11_TEXTURE2D_DESC textureDescription{};
			textureDescription.Width = static_cast<UINT>(bgraFrame.cols);
			textureDescription.Height = static_cast<UINT>(bgraFrame.rows);
			textureDescription.MipLevels = 1;
			textureDescription.ArraySize = 1;
			textureDescription.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
			textureDescription.SampleDesc.Count = 1;
			textureDescription.Usage = D3D11_USAGE_DEFAULT;
			textureDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			if(FAILED(device->CreateTexture2D(&textureDescription, nullptr, &previewTexture)) ||
				FAILED(device->CreateShaderResourceView(previewTexture, nullptr, &previewView))) {
				CleanupPreviewTexture();
				return false;
			}
		}

		deviceContext->UpdateSubresource(
			previewTexture,
			0,
			nullptr,
			bgraFrame.data,
			static_cast<UINT>(bgraFrame.step),
			0);
		return true;
	}

	LRESULT WINAPI WindowProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
		if(ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam)) {
			return 1;
		}

		switch(message) {
		case WM_SIZE:
			if(device != nullptr && wParam != SIZE_MINIMIZED) {
				resizeWidth = LOWORD(lParam);
				resizeHeight = HIWORD(lParam);
			}
			return 0;
		case WM_SYSCOMMAND:
			if((wParam & 0xFFF0) == SC_KEYMENU) {
				return 0;
			}
			break;
		case WM_DESTROY:
			windowOpen = false;
			PostQuitMessage(0);
			return 0;
		default:
			break;
		}
		return DefWindowProcW(window, message, wParam, lParam);
	}
}

namespace ImGuiUI {
	bool Initialize(HINSTANCE instance) {
		ImGui_ImplWin32_EnableDpiAwareness();
		const wchar_t * className = L"AntiTouchMinePCImGuiWindow";
		WNDCLASSEXW windowClass{};
		windowClass.cbSize = sizeof(windowClass);
		windowClass.style = CS_CLASSDC;
		windowClass.lpfnWndProc = WindowProcedure;
		windowClass.hInstance = instance;
		windowClass.lpszClassName = className;
		if(RegisterClassExW(&windowClass) == 0) {
			std::cerr << "Failed to register the ImGui window class." << std::endl;
			return false;
		}

		mainWindow = CreateWindowW(
			className,
			L"AntiTouchMinePC - 调试与设置",
			WS_OVERLAPPEDWINDOW,
			100,
			100,
			1180,
			820,
			nullptr,
			nullptr,
			instance,
			nullptr);
		if(mainWindow == nullptr) {
			std::cerr << "Failed to create the ImGui window." << std::endl;
			UnregisterClassW(className, instance);
			return false;
		}

		if(!CreateDeviceD3D(mainWindow)) {
			std::cerr << "Failed to initialize Direct3D 11." << std::endl;
			CleanupDeviceD3D();
			DestroyWindow(mainWindow);
			mainWindow = nullptr;
			UnregisterClassW(className, instance);
			return false;
		}

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO & io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		ImGui::StyleColorsDark();
		ImGui::GetStyle().WindowRounding = 6.0f;

		if(!ImGui_ImplWin32_Init(mainWindow)) {
			std::cerr << "Failed to initialize the ImGui Win32 backend." << std::endl;
			ImGui::DestroyContext();
			CleanupDeviceD3D();
			DestroyWindow(mainWindow);
			mainWindow = nullptr;
			UnregisterClassW(className, instance);
			return false;
		}
		if(!ImGui_ImplDX11_Init(device, deviceContext)) {
			std::cerr << "Failed to initialize the ImGui Direct3D backend." << std::endl;
			ImGui_ImplWin32_Shutdown();
			ImGui::DestroyContext();
			CleanupDeviceD3D();
			DestroyWindow(mainWindow);
			mainWindow = nullptr;
			UnregisterClassW(className, instance);
			return false;
		}

		io.FontDefault = io.Fonts->AddFontFromFileTTF(
			"C:\\Windows\\Fonts\\msyh.ttc",
			18.0f,
			nullptr,
			io.Fonts->GetGlyphRangesChineseFull());
		if(io.FontDefault == nullptr) {
			std::cerr << "Chinese font not found; the UI will use the default font." << std::endl;
		}

		ShowWindow(mainWindow, SW_SHOWDEFAULT);
		UpdateWindow(mainWindow);
		windowOpen = true;
		return true;
	}

	bool ProcessMessages() {
		MSG message{};
		while(PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&message);
			DispatchMessageW(&message);
			if(message.message == WM_QUIT) {
				windowOpen = false;
			}
		}
		return windowOpen;
	}

	void Render(Settings & settings, const Status & status, const cv::Mat & cameraFrame) {
		if(resizeWidth != 0 && resizeHeight != 0) {
			CleanupRenderTarget();
			const HRESULT resizeResult = swapChain->ResizeBuffers(
				0,
				resizeWidth,
				resizeHeight,
				DXGI_FORMAT_UNKNOWN,
				0);
			resizeWidth = 0;
			resizeHeight = 0;
			if(FAILED(resizeResult)) {
				std::cerr << "Failed to resize the Direct3D swap chain." << std::endl;
				windowOpen = false;
				return;
			}
			CreateRenderTarget();
			if(renderTarget == nullptr) {
				std::cerr << "Failed to recreate the Direct3D render target." << std::endl;
				windowOpen = false;
				return;
			}
		}

		const bool hasPreview = UpdatePreviewTexture(cameraFrame);
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		ImGui::SetNextWindowSize(ImVec2(900.0f, 650.0f), ImGuiCond_FirstUseEver);
		ImGui::Begin("AntiTouchMinePC - 调试与设置");
		ImGui::Text("运行状态：%s", status.message.c_str());
		ImGui::Text("摄像头：%s    分辨率：%d x %d",
			status.cameraReady ? "已连接" : "未连接",
			status.frameWidth,
			status.frameHeight);
		ImGui::Text("界面 FPS：%.1f    推理耗时：%.1f ms",
			status.framesPerSecond,
			status.processingMilliseconds);
		ImGui::Text("检测到人物：%d    人脸：%d", status.personCount, status.faceCount);
		ImGui::Text("最近抓拍：%s",
			settings.captureEnabled ? status.captureMessage.c_str() : "抓拍已关闭");
		ImGui::Separator();

		ImGui::Checkbox("启用检测", &settings.detectionEnabled);
		ImGui::SameLine();
		ImGui::Checkbox("性能模式（640x480）", &settings.performanceMode);
		ImGui::SameLine();
		ImGui::Checkbox("绘制人物框", &settings.drawPeople);
		ImGui::SameLine();
		ImGui::Checkbox("绘制人脸框", &settings.drawFaces);
		ImGui::Checkbox("启用抓拍（摄像头和桌面）", &settings.captureEnabled);
		ImGui::BeginDisabled(!settings.captureEnabled);
		ImGui::Checkbox("抓拍图片绘制检测框", &settings.captureWithBoxes);
		ImGui::Checkbox("检测到人脸时自动抓拍", &settings.autoCapture);
		ImGui::SliderInt("自动抓拍间隔（秒）", &settings.captureIntervalSeconds, 1, 60);
		ImGui::EndDisabled();
		ImGui::SliderFloat("人物置信度", &settings.confidenceThreshold, 0.05f, 0.95f, "%.2f");
		ImGui::SliderFloat("NMS 阈值", &settings.nmsThreshold, 0.05f, 0.95f, "%.2f");
		ImGui::SliderInt("Haar 人脸检测邻居数", &settings.faceMinNeighbors, 1, 10);
		ImGui::SliderInt("每 N 帧检测一次", &settings.detectionIntervalFrames, 1, 5);
		ImGui::Separator();

		const ImVec2 availableSize = ImGui::GetContentRegionAvail();
		ImGui::BeginChild("CameraPreview", availableSize, ImGuiChildFlags_Borders);
		if(hasPreview && previewView != nullptr) {
			const float imageWidth = ImGui::GetContentRegionAvail().x;
			const float imageHeight = ImGui::GetContentRegionAvail().y;
			const float aspectRatio = static_cast<float>(cameraFrame.cols) / cameraFrame.rows;
			float displayWidth = imageWidth;
			float displayHeight = displayWidth / aspectRatio;
			if(displayHeight > imageHeight) {
				displayHeight = imageHeight;
				displayWidth = displayHeight * aspectRatio;
			}
			ImGui::Image(
				static_cast<ImTextureID>(reinterpret_cast<intptr_t>(previewView)),
				ImVec2(displayWidth, displayHeight));
		} else {
			ImGui::TextUnformatted("等待摄像头画面...");
		}
		ImGui::EndChild();
		ImGui::End();

		ImGui::Render();
		const float clearColor[4] = { 0.08f, 0.09f, 0.11f, 1.0f };
		deviceContext->OMSetRenderTargets(1, &renderTarget, nullptr);
		deviceContext->ClearRenderTargetView(renderTarget, clearColor);
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
		const HRESULT presentResult = swapChain->Present(1, 0);
		if(FAILED(presentResult)) {
			std::cerr << "Failed to present the ImGui frame." << std::endl;
			windowOpen = false;
		}
	}

	void Shutdown() {
		CleanupPreviewTexture();
		ImGui_ImplDX11_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
		CleanupDeviceD3D();
		if(mainWindow != nullptr) {
			DestroyWindow(mainWindow);
			mainWindow = nullptr;
		}
		UnregisterClassW(L"AntiTouchMinePCImGuiWindow", GetModuleHandleW(nullptr));
	}
}
