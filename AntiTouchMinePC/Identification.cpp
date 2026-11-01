#include "Identification.hpp"

namespace FaceDetetction {
	bool LoadFaceDetector(cv::CascadeClassifier & detector, const std::string & modelPath) {
        // 先重置分类器，避免重复使用旧的模型状态。
        detector = cv::CascadeClassifier();
        if(modelPath.empty()) {
            return false;
        }
        // 加载 Haar XML 模型文件。
        if(!detector.load(modelPath)) {
            return false;
        }
        // 模型对象为空时不能继续执行检测。
        return !detector.empty();
	}

    std::vector<cv::Rect> DetectFacesInPerson(const cv::Mat & frame, const cv::Rect & personBox, cv::CascadeClassifier & detector,
    double scaleFactor, int minNeighbors, cv::Size minFaceSize) {
        std::vector<cv::Rect> result;
        // 画面为空或分类器尚未加载时直接返回空结果。
        if(frame.empty() || detector.empty()) {
            return result;
        }
        // 计算完整画面范围，并裁掉人物框中超出画面的部分。
        const cv::Rect frameRect(0, 0, frame.cols, frame.rows);
        const cv::Rect safePersonBox = personBox & frameRect;
        // 人物框与画面没有交集时，不需要执行人脸检测。
        if(safePersonBox.width <= 0 || safePersonBox.height <= 0) {
            return result;
        }
        // 只裁剪人物区域，减少需要扫描的像素数量。
        const cv::Mat roi = frame(safePersonBox);
        // Haar 分类器需要灰度图像，同时兼容灰度、BGR 和 BGRA 输入。
        cv::Mat gray;
        if(roi.channels() == 1) {
            gray = roi.clone();
        } else if(roi.channels() == 3) {
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        } else if(roi.channels() == 4) {
            cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
        } else {
            return result;
        }
        // 均衡化灰度直方图，减轻光照不均匀对人脸检测的影响。
        cv::equalizeHist(gray, gray);
        // localFaces 中的坐标相对于人物区域，而不是相对于完整画面。
        std::vector<cv::Rect> localFaces;
        detector.detectMultiScale(gray, localFaces, scaleFactor, minNeighbors, 0, minFaceSize);
        result.reserve(localFaces.size());
        // 加上人物框左上角坐标，转换成完整画面中的坐标。
        for(const cv::Rect & face : localFaces) {
            result.emplace_back(safePersonBox.x + face.x, safePersonBox.y + face.y, face.width, face.height);
        }

        return result;
    }

    void DrawFaces(cv::Mat & frame, const std::vector<cv::Rect> & faces, const cv::Scalar & color, int thickness) {
        // 遍历所有检测结果并绘制抗锯齿矩形。
        for(const cv::Rect & face : faces) {
            cv::rectangle(frame, face, color, thickness, cv::LINE_AA);
        }
    }
}