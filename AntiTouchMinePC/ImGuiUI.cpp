#include "ImGuiUI.hpp"

#include <d3d11.h>
#include <dxgi.h>
#include <shellapi.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "shell32.lib")

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
	bool trayIconAdded = false;
	bool closeToTrayEnabled = true;
	constexpr UINT trayIconMessage = WM_APP + 100;
	constexpr UINT restoreMenuCommand = 1001;
	constexpr UINT exitMenuCommand = 1002;

	struct PhotoEntry {
		std::filesystem::path path;
		std::string timestamp;
		cv::Mat thumbnailPixels;
		ID3D11ShaderResourceView * thumbnail = nullptr;
		UINT thumbnailWidth = 0;
		UINT thumbnailHeight = 0;
	};

	struct PhotoThumbnailData {
		std::filesystem::path path;
		std::string timestamp;
		cv::Mat pixels;
		UINT width = 0;
		UINT height = 0;
	};

	struct CachedPhoto {
		PhotoThumbnailData photo;
		std::uintmax_t fileSize = 0;
		std::filesystem::file_time_type modifiedTime{};
	};

	struct PhotoLoader {
		std::mutex mutex;
		std::condition_variable ready;
		std::condition_variable resultConsumed;
		std::thread thread;
		bool running = false;
		bool requested = false;
		bool busy = false;
		bool resultReady = false;
		std::string error;
		std::vector<PhotoThumbnailData> result;
	};

	std::vector<PhotoEntry> photoEntries;
	PhotoLoader photoLoader;
	std::chrono::steady_clock::time_point lastPhotoRefresh{};
	bool photoWindowOpen = false;
	std::string photoListMessage;

	std::string FormatPhotoTimestamp(const std::filesystem::path & path) {
		std::string name = path.stem().string();
		const std::size_t suffix = name.rfind('_');
		if(suffix == std::string::npos) {
			return name;
		}
		name.resize(suffix);
		if(name.size() != 19 || name[8] != '_' || name[15] != '_') {
			return name;
		}
		return name.substr(0, 4) + "-" + name.substr(4, 2) + "-" + name.substr(6, 2) +
			" " + name.substr(9, 2) + ":" + name.substr(11, 2) + ":" + name.substr(13, 2) +
			"." + name.substr(16, 3);
	}

	PhotoThumbnailData LoadPhotoThumbnailData(const std::filesystem::path & path) {
		PhotoThumbnailData result;
		result.path = path;
		result.timestamp = FormatPhotoTimestamp(path);
		try {
			const cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
			if(image.empty()) {
				std::cerr << "Failed to load photo thumbnail: " << path.string() << std::endl;
				return result;
			}

			const double scale = std::min({ 144.0 / image.cols, 96.0 / image.rows, 1.0 });
			result.width = static_cast<UINT>(std::max(1.0, std::round(image.cols * scale)));
			result.height = static_cast<UINT>(std::max(1.0, std::round(image.rows * scale)));
			cv::Mat resized;
			cv::resize(image, resized, cv::Size(
				static_cast<int>(result.width),
				static_cast<int>(result.height)));
			if(resized.channels() == 4) {
				result.pixels = resized;
			} else if(resized.channels() == 3) {
				cv::cvtColor(resized, result.pixels, cv::COLOR_BGR2BGRA);
			} else if(resized.channels() == 1) {
				cv::cvtColor(resized, result.pixels, cv::COLOR_GRAY2BGRA);
			} else {
				std::cerr << "Unsupported photo channel count: " << path.string() << std::endl;
			}
		} catch(const cv::Exception & error) {
			std::cerr << "Failed to load photo thumbnail " << path.string() << ": "
				<< error.what() << std::endl;
		}
		return result;
	}

	ID3D11ShaderResourceView * CreatePhotoThumbnailTexture(
		const cv::Mat & pixels,
		const std::filesystem::path & path) {
		D3D11_TEXTURE2D_DESC description{};
		description.Width = static_cast<UINT>(pixels.cols);
		description.Height = static_cast<UINT>(pixels.rows);
		description.MipLevels = 1;
		description.ArraySize = 1;
		description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		description.SampleDesc.Count = 1;
		description.Usage = D3D11_USAGE_DEFAULT;
		description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA initialData{};
		initialData.pSysMem = pixels.data;
		initialData.SysMemPitch = static_cast<UINT>(pixels.step);
		ID3D11Texture2D * texture = nullptr;
		if(FAILED(device->CreateTexture2D(&description, &initialData, &texture))) {
			std::cerr << "Failed to create photo thumbnail texture: " << path.string() << std::endl;
			return nullptr;
		}

		ID3D11ShaderResourceView * view = nullptr;
		if(FAILED(device->CreateShaderResourceView(texture, nullptr, &view))) {
			std::cerr << "Failed to create photo thumbnail view: " << path.string() << std::endl;
			texture->Release();
			return nullptr;
		}
		texture->Release();
		return view;
	}

	void ReleasePhotoEntry(PhotoEntry & entry) {
		if(entry.thumbnail != nullptr) {
			entry.thumbnail->Release();
			entry.thumbnail = nullptr;
		}
	}

	void PhotoLoaderThread() {
		std::unordered_map<std::string, CachedPhoto> cache;
		while(true) {
			{
				std::unique_lock<std::mutex> lock(photoLoader.mutex);
				photoLoader.ready.wait(lock, []() {
					return !photoLoader.running || photoLoader.requested;
				});
				if(!photoLoader.running) {
					return;
				}
				photoLoader.requested = false;
				photoLoader.busy = true;
			}

			std::vector<PhotoThumbnailData> refreshed;
			std::string errorMessage;
			try {
				const std::filesystem::path directory("captures");
				std::error_code error;
				std::filesystem::directory_iterator iterator(directory, error);
				std::vector<std::filesystem::path> paths;
				if(error == std::errc::no_such_file_or_directory) {
					std::error_code absoluteError;
					const std::filesystem::path absoluteDirectory =
						std::filesystem::absolute(directory, absoluteError);
					errorMessage = "照片目录不存在：" +
						(absoluteError ? directory.string() : absoluteDirectory.string());
				} else if(error) {
					errorMessage = "读取照片目录失败：" + error.message();
				} else {
					const std::filesystem::directory_iterator end;
					while(iterator != end) {
						std::error_code entryError;
						if(iterator->is_regular_file(entryError) &&
							iterator->path().extension() == ".png") {
							paths.push_back(iterator->path());
						} else if(entryError) {
							errorMessage = "读取照片信息失败：" + entryError.message();
						}
						iterator.increment(error);
						if(error) {
							errorMessage = "读取照片目录失败：" + error.message();
							break;
						}
					}
				}

				std::sort(paths.begin(), paths.end(), [](const auto & left, const auto & right) {
					return left.filename().native() > right.filename().native();
				});
				std::unordered_set<std::string> currentPaths;
				refreshed.reserve(paths.size());
				for(const std::filesystem::path & path : paths) {
					const std::string key = path.string();
					currentPaths.insert(key);
					std::error_code metadataError;
					const std::uintmax_t fileSize = std::filesystem::file_size(path, metadataError);
					if(metadataError) {
						errorMessage = "读取照片大小失败：" + metadataError.message();
						continue;
					}
					const auto modifiedTime = std::filesystem::last_write_time(path, metadataError);
					if(metadataError) {
						errorMessage = "读取照片时间失败：" + metadataError.message();
						continue;
					}

					auto cached = cache.find(key);
					if(cached == cache.end() || cached->second.fileSize != fileSize ||
						cached->second.modifiedTime != modifiedTime) {
						CachedPhoto entry;
						entry.photo.path = path;
						entry.photo.timestamp = FormatPhotoTimestamp(path);
						entry.fileSize = fileSize;
						entry.modifiedTime = modifiedTime;
						cached = cache.insert_or_assign(key, std::move(entry)).first;
					}
					refreshed.push_back(cached->second.photo);
				}
				for(auto cached = cache.begin(); cached != cache.end();) {
					if(currentPaths.find(cached->first) == currentPaths.end()) {
						cached = cache.erase(cached);
					} else {
						++cached;
					}
				}
			} catch(const std::exception & error) {
				errorMessage = std::string("加载照片列表失败：") + error.what();
			}

			{
				std::unique_lock<std::mutex> lock(photoLoader.mutex);
				photoLoader.result = refreshed;
				photoLoader.error = errorMessage;
				photoLoader.resultReady = true;
				photoLoader.resultConsumed.wait(lock, []() {
					return !photoLoader.running || !photoLoader.resultReady;
				});
				if(!photoLoader.running) {
					return;
				}
			}

			for(PhotoThumbnailData & photo : refreshed) {
				if(!photo.pixels.empty()) {
					continue;
				}
				auto cached = cache.find(photo.path.string());
				if(cached != cache.end()) {
					cached->second.photo = LoadPhotoThumbnailData(photo.path);
					photo = cached->second.photo;
				}
			}

			{
				std::lock_guard<std::mutex> lock(photoLoader.mutex);
				photoLoader.result = std::move(refreshed);
				photoLoader.error = std::move(errorMessage);
				photoLoader.resultReady = true;
				photoLoader.busy = false;
			}
		}
	}

	void RefreshPhotoEntries() {
		std::vector<PhotoThumbnailData> loadedPhotos;
		std::string errorMessage;
		bool hasNewResult = false;
		{
			std::lock_guard<std::mutex> lock(photoLoader.mutex);
			if(photoLoader.resultReady) {
				loadedPhotos = std::move(photoLoader.result);
				errorMessage = std::move(photoLoader.error);
				photoLoader.resultReady = false;
				hasNewResult = true;
			}
		}
		if(hasNewResult) {
			photoLoader.resultConsumed.notify_one();
		}

		if(hasNewResult) {
			std::unordered_map<std::string, std::size_t> existingEntries;
			existingEntries.reserve(photoEntries.size());
			for(std::size_t index = 0; index < photoEntries.size(); ++index) {
				existingEntries.emplace(photoEntries[index].path.string(), index);
			}

			std::vector<PhotoEntry> refreshed;
			refreshed.reserve(loadedPhotos.size());
			for(PhotoThumbnailData & photo : loadedPhotos) {
				PhotoEntry entry;
				entry.path = std::move(photo.path);
				entry.timestamp = std::move(photo.timestamp);
				entry.thumbnailPixels = std::move(photo.pixels);
				entry.thumbnailWidth = photo.width;
				entry.thumbnailHeight = photo.height;
				const auto existing = existingEntries.find(entry.path.string());
				if(existing != existingEntries.end()) {
					PhotoEntry & previous = photoEntries[existing->second];
					entry.thumbnail = previous.thumbnail;
					previous.thumbnail = nullptr;
					if(entry.thumbnail != nullptr) {
						entry.thumbnailPixels.release();
					}
				}
				refreshed.push_back(std::move(entry));
			}
			for(PhotoEntry & entry : photoEntries) {
				ReleasePhotoEntry(entry);
			}
			photoEntries = std::move(refreshed);
			photoListMessage = std::move(errorMessage);
		}

		const auto now = std::chrono::steady_clock::now();
		if(lastPhotoRefresh == std::chrono::steady_clock::time_point{} ||
			now - lastPhotoRefresh >= std::chrono::seconds(1)) {
			lastPhotoRefresh = now;
			std::lock_guard<std::mutex> lock(photoLoader.mutex);
			if(photoLoader.running && !photoLoader.busy && !photoLoader.requested) {
				photoLoader.requested = true;
				photoLoader.ready.notify_one();
			}
		}
	}

	bool IsPhotoLoaderActive() {
		std::lock_guard<std::mutex> lock(photoLoader.mutex);
		return photoLoader.busy || photoLoader.requested || photoLoader.resultReady;
	}

	bool AddTrayIcon() {
		if(trayIconAdded) {
			return true;
		}

		NOTIFYICONDATAW iconData{};
		iconData.cbSize = sizeof(iconData);
		iconData.hWnd = mainWindow;
		iconData.uID = 1;
		iconData.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
		iconData.uCallbackMessage = trayIconMessage;
		iconData.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
		lstrcpynW(iconData.szTip, L"AntiTouchMinePC", ARRAYSIZE(iconData.szTip));
		if(!Shell_NotifyIconW(NIM_ADD, &iconData)) {
			std::cerr << "Failed to add the notification area icon. Error: "
				<< GetLastError() << std::endl;
			return false;
		}
		trayIconAdded = true;
		return true;
	}

	void RemoveTrayIcon() {
		if(!trayIconAdded) {
			return;
		}
		NOTIFYICONDATAW iconData{};
		iconData.cbSize = sizeof(iconData);
		iconData.hWnd = mainWindow;
		iconData.uID = 1;
		if(!Shell_NotifyIconW(NIM_DELETE, &iconData)) {
			std::cerr << "Failed to remove the notification area icon. Error: "
				<< GetLastError() << std::endl;
		}
		trayIconAdded = false;
	}

	void RestoreMainWindow() {
		RemoveTrayIcon();
		ShowWindow(mainWindow, SW_SHOW);
		SetForegroundWindow(mainWindow);
	}

	void ShowTrayMenu() {
		HMENU menu = CreatePopupMenu();
		if(menu == nullptr) {
			std::cerr << "Failed to create the notification area menu. Error: "
				<< GetLastError() << std::endl;
			return;
		}
		if(!AppendMenuW(menu, MF_STRING, restoreMenuCommand, L"打开主窗口") ||
			!AppendMenuW(menu, MF_SEPARATOR, 0, nullptr) ||
			!AppendMenuW(menu, MF_STRING, exitMenuCommand, L"退出")) {
			std::cerr << "Failed to populate the notification area menu. Error: "
				<< GetLastError() << std::endl;
			DestroyMenu(menu);
			return;
		}

		POINT cursor{};
		if(!GetCursorPos(&cursor)) {
			std::cerr << "Failed to get the cursor position. Error: "
				<< GetLastError() << std::endl;
			DestroyMenu(menu);
			return;
		}
		SetForegroundWindow(mainWindow);
		const UINT command = TrackPopupMenu(
			menu,
			TPM_RIGHTBUTTON | TPM_RETURNCMD,
			cursor.x,
			cursor.y,
			0,
			mainWindow,
			nullptr);
		PostMessageW(mainWindow, WM_NULL, 0, 0);
		DestroyMenu(menu);

		if(command == restoreMenuCommand) {
			RestoreMainWindow();
		} else if(command == exitMenuCommand) {
			RemoveTrayIcon();
			windowOpen = false;
			PostQuitMessage(0);
		}
	}

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
		switch(message) {
		case trayIconMessage:
			switch(LOWORD(lParam)) {
			case WM_LBUTTONUP:
			case WM_LBUTTONDBLCLK:
				RestoreMainWindow();
				return 0;
			case WM_RBUTTONUP:
				ShowTrayMenu();
				return 0;
			}
			break;
		case WM_CLOSE:
			if(!closeToTrayEnabled) {
				DestroyWindow(window);
			} else if(AddTrayIcon()) {
				ShowWindow(window, SW_HIDE);
			}
			return 0;
		case WM_DESTROY:
			RemoveTrayIcon();
			windowOpen = false;
			mainWindow = nullptr;
			PostQuitMessage(0);
			return 0;
		default:
			break;
		}

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

		photoLoader.running = true;
		photoLoader.thread = std::thread(PhotoLoaderThread);
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
		if(ImGui::Button("打开照片列表")) {
			photoWindowOpen = true;
			lastPhotoRefresh = std::chrono::steady_clock::time_point{};
		}
		ImGui::Separator();

		if(ImGui::CollapsingHeader("检测设置", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("启用检测", &settings.detectionEnabled);
			ImGui::SliderFloat("人物置信度", &settings.confidenceThreshold, 0.05f, 0.95f, "%.2f");
			ImGui::SliderFloat("NMS 阈值", &settings.nmsThreshold, 0.05f, 0.95f, "%.2f");
			ImGui::SliderInt("Haar 人脸检测邻居数", &settings.faceMinNeighbors, 1, 10);
			ImGui::SliderInt("每 N 帧检测一次", &settings.detectionIntervalFrames, 1, 5);
		}

		if(ImGui::CollapsingHeader("界面与性能", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("性能模式（640x480）", &settings.performanceMode);
			ImGui::Checkbox("绘制人物框", &settings.drawPeople);
			ImGui::Checkbox("绘制人脸框", &settings.drawFaces);
			ImGui::Checkbox("关闭时最小化到托盘", &settings.closeToTray);
			closeToTrayEnabled = settings.closeToTray;
		}

		if(ImGui::CollapsingHeader("抓拍设置", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("启用抓拍（摄像头和桌面）", &settings.captureEnabled);
			ImGui::BeginDisabled(!settings.captureEnabled);
			ImGui::Checkbox("抓拍图片绘制检测框", &settings.captureWithBoxes);
			ImGui::Checkbox("检测到人脸时自动抓拍", &settings.autoCapture);
			ImGui::SliderInt("自动抓拍间隔（秒）", &settings.captureIntervalSeconds, 1, 60);
			ImGui::EndDisabled();
		}

		if(ImGui::CollapsingHeader("摄像头预览", ImGuiTreeNodeFlags_DefaultOpen)) {
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
		}
		ImGui::End();

		if(photoWindowOpen) {
			RefreshPhotoEntries();
			ImGui::SetNextWindowSize(ImVec2(720.0f, 520.0f), ImGuiCond_FirstUseEver);
			if(ImGui::Begin("照片列表", &photoWindowOpen)) {
				if(!photoListMessage.empty()) {
					ImGui::TextWrapped("%s", photoListMessage.c_str());
				}
				if(photoEntries.empty() && IsPhotoLoaderActive()) {
					ImGui::TextUnformatted("正在读取照片...");
				} else if(photoEntries.empty()) {
					ImGui::TextUnformatted("暂时没有抓拍照片。");
				} else if(ImGui::BeginTable(
					"CapturedPhotosTable",
					3,
					ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
						ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
					ImVec2(0.0f, 0.0f))) {
					ImGui::TableSetupColumn("照片", ImGuiTableColumnFlags_WidthFixed, 180.0f);
					ImGui::TableSetupColumn("时间", ImGuiTableColumnFlags_WidthStretch);
					ImGui::TableSetupColumn("操作", ImGuiTableColumnFlags_WidthFixed, 130.0f);
					ImGui::TableSetupScrollFreeze(0, 1);
					ImGui::TableHeadersRow();

					std::filesystem::path photoToDelete;
					ImGuiListClipper clipper;
					clipper.Begin(static_cast<int>(photoEntries.size()), 104.0f);
					while(clipper.Step()) {
						for(int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
						const std::size_t index = static_cast<std::size_t>(row);
						PhotoEntry & entry = photoEntries[index];
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						if(entry.thumbnail == nullptr && !entry.thumbnailPixels.empty() &&
							ImGui::IsRectVisible(ImVec2(
								static_cast<float>(entry.thumbnailWidth),
								static_cast<float>(entry.thumbnailHeight)))) {
							entry.thumbnail = CreatePhotoThumbnailTexture(entry.thumbnailPixels, entry.path);
							if(entry.thumbnail != nullptr) {
								entry.thumbnailPixels.release();
							}
						}
						if(entry.thumbnail != nullptr) {
							ImGui::Image(
								static_cast<ImTextureID>(reinterpret_cast<intptr_t>(entry.thumbnail)),
								ImVec2(
									static_cast<float>(entry.thumbnailWidth),
									static_cast<float>(entry.thumbnailHeight)));
						} else {
							ImGui::Dummy(ImVec2(144.0f, 96.0f));
							ImGui::TextUnformatted(entry.path.filename().string().c_str());
						}
						ImGui::TableSetColumnIndex(1);
						ImGui::TextUnformatted(entry.timestamp.c_str());
						ImGui::TableSetColumnIndex(2);
						ImGui::PushID(static_cast<int>(index));
						if(ImGui::Button("打开")) {
							const INT_PTR result = reinterpret_cast<INT_PTR>(ShellExecuteW(
								mainWindow,
								L"open",
								entry.path.c_str(),
								nullptr,
								nullptr,
								SW_SHOWNORMAL));
							if(result <= 32) {
								photoListMessage = "打开照片失败，ShellExecute 错误码：" +
									std::to_string(result);
								std::cerr << photoListMessage << std::endl;
							} else {
								photoListMessage.clear();
							}
						}
						ImGui::SameLine();
						if(ImGui::Button("删除")) {
							photoToDelete = entry.path;
						}
						ImGui::PopID();
						}
					}
					ImGui::EndTable();

					if(!photoToDelete.empty()) {
						std::error_code deleteError;
						if(std::filesystem::remove(photoToDelete, deleteError)) {
							photoListMessage.clear();
							lastPhotoRefresh = std::chrono::steady_clock::time_point{};
						} else {
							photoListMessage = "删除照片失败：" +
								(deleteError ? deleteError.message() : std::string("文件不存在"));
							std::cerr << photoListMessage << std::endl;
							lastPhotoRefresh = std::chrono::steady_clock::time_point{};
						}
					}
				}
				ImGui::End();
			}
		}

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
		{
			std::lock_guard<std::mutex> lock(photoLoader.mutex);
			photoLoader.running = false;
		}
		photoLoader.ready.notify_one();
		photoLoader.resultConsumed.notify_one();
		if(photoLoader.thread.joinable()) {
			photoLoader.thread.join();
		}
		CleanupPreviewTexture();
		for(PhotoEntry & entry : photoEntries) {
			ReleasePhotoEntry(entry);
		}
		photoEntries.clear();
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
