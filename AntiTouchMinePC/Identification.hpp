#pragma once
#ifndef IDENTIFICATION_HPP
#define IDENTIFICATION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

namespace FaceDetetction {
    bool LoadFaceDetector(cv::CascadeClassifier & detector, const std::string & modelPath);

    std::vector<cv::Rect> DetectFacesInPerson(
    const cv::Mat & frame,
    const cv::Rect & personBox,
    cv::CascadeClassifier & detector,
    double scaleFactor = 1.1,
    int minNeighbors = 3,
    cv::Size minFaceSize = cv::Size(30, 30)
    );

    void DrawFaces(cv::Mat & frame, const std::vector<cv::Rect> & faces, const cv::Scalar & color = cv::Scalar(0, 255, 0), int thickness = 2);
}

#endif // !IDENTIFICATION_HPP


