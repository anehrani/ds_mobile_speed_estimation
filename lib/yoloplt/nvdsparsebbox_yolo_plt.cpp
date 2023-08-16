#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>


#define NUM_KEYPOINTS 5
#define NUM_OUTPUTS 3
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
//#define MODEL_INPUT_WIDTH 192
//#define MODEL_INPUT_HEIGHT 192

//#define  rescalex MUXER_OUTPUT_WIDTH/MODEL_INPUT_WIDTH
//#define  rescaley MUXER_OUTPUT_HEIGHT/MODEL_INPUT_HEIGHT


static float clamp(const float val, const float minVal, const float maxVal);

static inline float sigmoid(float x){
    return static_cast<float>(1.f / (1.f + exp(-x)));
}


extern "C" bool NvDsInferParseYoloPlate(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);


static bool NvDsInferParseCustomYoloPlate(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList,
        const uint &numClasses);

static void decode_v1(std::vector<NvDsInferLayerInfo> out_data, std::vector<NvDsInferInstanceMaskInfo>& prebox, float threshold=0.5);
static void nms(std::vector<NvDsInferInstanceMaskInfo> &input_boxes, float NMS_THRESH);
// compare function
static bool cmp(NvDsInferInstanceMaskInfo b1, NvDsInferInstanceMaskInfo b2) {
    return b1.detectionConfidence > b2.detectionConfidence;
}


static int ANCHORS[18] = {4,5,  6,8,  10,12,  15,19,  23,30,  39,52,  72,97,  123,164,  209,297};
static int OUTPUT_DIMS[9] =  {63,40,40, 63,20,20, 63,10,10}; // 320: {40,20,10} //  256: {63,32,32, 63,16,16, 63,8,8}  //192: {63,24,24, 63,12,12, 63,6,6}, 640: {63,80,80, 63,40,40, 63,20,20}; 480: {63,60,60, 63,30,30, 63,15,15}
static int STRIDE[3] = {8, 16, 32};


// modified for number plate info
static bool NvDsInferParseCustomYoloPlate(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList,
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

    // decode net outputs
    decode_v1( outputLayersInfo, objectList );
    // non-maximum suppression
    std::sort(objectList.begin(), objectList.end(), cmp);
    nms(objectList, .15); // set nms threshold = 0.5
    return true;
}


extern "C" bool NvDsInferParseYoloPlate( std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
    //objectList.clear();
    return NvDsInferParseCustomYoloPlate (
            outputLayersInfo, networkInfo, detectionParams, objectList, 1);
}


static void decode_v1(std::vector<NvDsInferLayerInfo> out_data, std::vector<NvDsInferInstanceMaskInfo>& preMpoints, float threshold) {
    int channels = NUM_KEYPOINTS * 3 + 6;
    int spacial_size;// = fea_w*fea_h;
    // decoding outputs at the same time
    for (int oi=0; oi< NUM_OUTPUTS; oi++){
        spacial_size = OUTPUT_DIMS[3*oi + 1] * OUTPUT_DIMS[3*oi + 2];
        float* layer_out = (float*) out_data[oi].buffer;
        // this is for 3 anchored model
        for(int c = 0; c < 3; c++)
        {
            float anchor_w = float(ANCHORS[6*oi + c * 2 + 0]);
            float anchor_h = float(ANCHORS[6*oi + c * 2 + 1]);
            float *ptr_x = layer_out + spacial_size * (c * channels + 0);
            float *ptr_y = layer_out + spacial_size * (c * channels + 1);
            float *ptr_w = layer_out + spacial_size * (c * channels + 2);
            float *ptr_h = layer_out + spacial_size * (c * channels + 3);
            float *ptr_s = layer_out + spacial_size * (c * channels + 4);
            float *ptr_c = layer_out + spacial_size * (c * channels + 5);
            // iterating over features
            for(int i = 0; i < OUTPUT_DIMS[3*oi + 2]; i++) {
                for (int j = 0; j < OUTPUT_DIMS[3*oi + 1]; j++) {
                    int index = i * OUTPUT_DIMS[3*oi + 1] + j;
                    float confidence = sigmoid(ptr_s[index]) * sigmoid(ptr_c[index]);
                    if(confidence > threshold)
                    {
                        NvDsInferInstanceMaskInfo tmp_object;
                        float dx = sigmoid(ptr_x[index]);
                        float dy = sigmoid(ptr_y[index]);
                        float dw = sigmoid(ptr_w[index]);
                        float dh = sigmoid(ptr_h[index]);
                        float pb_cx = (dx * 2.f - 0.5f + j) * STRIDE[oi];
                        float pb_cy = (dy * 2.f - 0.5f + i) * STRIDE[oi];
                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;


                        tmp_object.detectionConfidence = confidence;
                        // todo: for the sake of tracking, I need to take a larger area around Number Plate to cheat :)
                        tmp_object.left =clamp( (pb_cx - pb_w * 0.5f), 0, MUXER_OUTPUT_WIDTH - 2); // - 1 * pb_w give larger area for better tracking
                        tmp_object.top = clamp((pb_cy - pb_h * 0.5f) , 0, MUXER_OUTPUT_HEIGHT - 1);//  - 2 * pb_w
                        tmp_object.width = clamp (pb_w, 10,  MUXER_OUTPUT_WIDTH );
                        tmp_object.height = clamp( pb_h, 5, MUXER_OUTPUT_HEIGHT);
                        tmp_object.classId = 0;
                        tmp_object.mask_size = 3 * NUM_KEYPOINTS*sizeof (float);
                        tmp_object.mask_width = 3;
                        tmp_object.mask_height = NUM_KEYPOINTS;
                        tmp_object.mask = new float[3 * NUM_KEYPOINTS];
                        for(int l = 0; l < NUM_KEYPOINTS; l++)
                        {
                            *(tmp_object.mask + 3*l) = (layer_out[(spacial_size * (c * channels + l * 3 + 6)) + index] * 2 - 0.5 + j) * STRIDE[oi];
                            *(tmp_object.mask + 3*l + 1) = (layer_out[(spacial_size * (c * channels + l * 3 + 7)) + index] * 2 - 0.5 + i) * STRIDE[oi];
                            *(tmp_object.mask + 3*l + 2) = sigmoid(layer_out[spacial_size * (c * channels + l * 3 + 8) + index]); // this is confidence of the poin
                            //std::cout<< "x: " << tmp_object.mask[3*l] << "  y: "<< tmp_object.mask[3*l+1] << " score: "<< tmp_object.mask[3*l+2] <<std::endl;
                        }
                        preMpoints.push_back(tmp_object);
                    } // end of if

                } // end of for j
            } // end of for i

        } // end of for c

    } // end of for oi number of network outputs
}

//
void nms(std::vector<NvDsInferInstanceMaskInfo>& input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());

    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).width + 1)
                   * (input_boxes.at(i).height + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].left, input_boxes[j].left);
            float yy1 = std::max(input_boxes[i].top, input_boxes[j].top);
            float xx2 = std::min(input_boxes[i].left + input_boxes[i].width, input_boxes[j].left + input_boxes[j].width);
            float yy2 = std::min(input_boxes[i].top + input_boxes[i].height, input_boxes[j].top + input_boxes[j].height);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}



float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPlate)