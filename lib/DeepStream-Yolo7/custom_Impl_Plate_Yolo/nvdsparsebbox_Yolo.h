/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */


#ifndef __UTILS_H__
#define __UTILS_H__

#include <map>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <math.h>
#include <memory>
#include <vector>

#include "NvInfer.h"
#include "nvdsinfer_custom_impl.h"
#include "NvInferPlugin.h"

//#include "plateInfo.h"

#define NUM_KEYPOINTS 5
#define NUM_OUTPUTS 3


static std::string trim(std::string s);
static float clamp(const float val, const float minVal, const float maxVal);
static bool fileExists(const std::string fileName, bool verbose = true);
static std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType);
static std::string dimsToString(const nvinfer1::Dims d);
static int getNumChannels(nvinfer1::ITensor* t);
static void printLayerInfo(
    std::string layerIndex, std::string layerName, std::string layerInput,  std::string layerOutput, std::string weightPtr);


static inline float sigmoid(float x){
    return static_cast<float>(1.f / (1.f + exp(-x)));
}


extern "C" bool NvDsInferParseYoloPlate(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);


static bool NvDsInferParseCustomYoloPlate(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList,
        const uint &numClasses);

static void decode_v1(std::vector<NvDsInferLayerInfo> out_data, std::vector<NvDsInferInstanceMaskInfo>& prebox, float threshold=0.5);
static void nms(std::vector<NvDsInferInstanceMaskInfo> &input_boxes, float NMS_THRESH);
// compare function
static bool cmp(NvDsInferInstanceMaskInfo b1, NvDsInferInstanceMaskInfo b2) {
    return b1.detectionConfidence > b2.detectionConfidence;
}


static int ANCHORS[18] = {4,5,  6,8,  10,12,  15,19,  23,30,  39,52,  72,97,  123,164,  209,297};
static int OUTPUT_DIMS[9] = {63,80,80, 63,40,40, 63,20,20};
static int STRIDE[3] = {8, 16, 32};


CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPlate)

#endif



