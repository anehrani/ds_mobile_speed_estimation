#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

extern "C" bool NvDsInferParseYolo7NMS(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);


static bool NvDsInferParseCustomYolo7(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList,
        const uint &numClasses);



// modified for number plate info
static bool NvDsInferParseCustomYolo7(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList,
        const uint &numClasses)
{
    if (outputLayersInfo.empty())
    {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

/*
    if (numClasses != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured: " << detectionParams.numClassesConfigured
                 << ", detected by network: " << numClasses << std::endl;
    }
*/

    int* num_dets = (int*) outputLayersInfo[0].buffer;
    float* det_boxes = (float*) outputLayersInfo[1].buffer;
    float* det_scores = (float*) outputLayersInfo[2].buffer;
    int* det_classes = (int*) outputLayersInfo[3].buffer;

    objectList.reserve( num_dets[0] );

    for (int i=0;i< num_dets[0]; i++){
        NvDsInferParseObjectInfo single_object;
        single_object.left = det_boxes[i*4 + 0];
        single_object.top = det_boxes[i*4 + 1];
        single_object.width = det_boxes[i*4 + 2] - det_boxes[i*4 + 0];
        single_object.height = det_boxes[i*4 + 3] - det_boxes[i*4 + 1];
        //
        single_object.detectionConfidence = det_scores[i];
        single_object.classId = det_classes[i];

        objectList.emplace_back( single_object ); // make filling objects faster as we have number of detections info at hand
    }

    return true;
}


extern "C" bool NvDsInferParseYolo7NMS( std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
    //objectList.clear();
    return NvDsInferParseCustomYolo7 (
            outputLayersInfo, networkInfo, detectionParams, objectList, 1);
}





CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo7NMS)