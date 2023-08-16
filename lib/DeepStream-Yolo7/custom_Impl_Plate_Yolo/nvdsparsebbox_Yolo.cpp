/*
 *
 */
#include <iostream>

#include <cmath>
#include <experimental/filesystem>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cuda_runtime_api.h>
#include "nvdsinfer_custom_impl.h"



#include "nvdsparsebbox_Yolo.h"




// modified for number plate info
static bool NvDsInferParseCustomYoloPlate(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList,
        const uint &numClasses)
{
    if (outputLayersInfo.empty())
    {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

    if (numClasses != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured: " << detectionParams.numClassesConfigured
                  << ", detected by network: " << numClasses << std::endl;
    }


    std::vector<NvDsInferInstanceMaskInfo> netOutputs;
    decode_v1( outputLayersInfo, netOutputs );

    // non-maximum suppression
    std::sort(netOutputs.begin(), netOutputs.end(), cmp);
    nms(netOutputs, .5); // set nms threshold = 0.5




    //NvDsInferObjectDetectionInfo tmp_obj;

    for (int i=0; i< netOutputs.size(); i++ ){
        std::cout<< " object   " << i << "  confidence : " << netOutputs.at(i).detectionConfidence << std::endl;
        std::cout<< netOutputs.at(i).left << " , " << netOutputs.at(i).top << " , " <<  netOutputs.at(i).width << " , " << netOutputs.at(i).height << std::endl;

        objectList.push_back( { netOutputs.at(i).classId, clamp(netOutputs.at(i).left, 0, networkInfo.width),
                                clamp(netOutputs.at(i).top, 0, networkInfo.height),
                                clamp(netOutputs.at(i).width, 0, networkInfo.width),
                                clamp( netOutputs.at(i).height, 0, networkInfo.height),
                                netOutputs.at(i).detectionConfidence} );
    }

    // filling out the detected points from different channel here


    return true;
}


extern "C" bool NvDsInferParseYoloPlate(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{

    objectList.clear();

    NvDsInferParseCustomYoloPlate (
            outputLayersInfo, networkInfo, detectionParams, objectList, 1);



    std::cout<< " num detections: " << objectList.size() << std::endl;
    /*
    //plateInfo::frame_meta tmp_frame_meta;


    for (int i=0; i< objectList.size(); i++) {


        std::cout<< " object   " << i << " confidence : " << objectList.at(i).detectionConfidence << std::endl;
        std::cout<< objectList.at(i).left << " , " << objectList.at(i).top << " , " <<  objectList.at(i).width << " , " << objectList.at(i).height << std::endl;

//        tmp_frame_meta.top = objectList.at(i).top;
//        tmp_frame_meta.left = objectList.at(i).left;
//        tmp_frame_meta.width = objectList.at(i).width;
//        tmp_frame_meta.height = objectList.at(i).height;
        //plateInfo::objectBuffer::getInstance()->putList( tmp_frame_meta );
    }
    */

    //std::cout<< "-----------------------  \n";


    return true;
}


static void decode_v1(std::vector<NvDsInferLayerInfo> out_data, std::vector<NvDsInferInstanceMaskInfo> & preMpoints, float threshold) {
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
                        NvDsInferInstanceMaskInfo temp_mask;
                        float dx = sigmoid(ptr_x[index]);
                        float dy = sigmoid(ptr_y[index]);
                        float dw = sigmoid(ptr_w[index]);
                        float dh = sigmoid(ptr_h[index]);
                        float pb_cx = (dx * 2.f - 0.5f + j) * STRIDE[oi];
                        float pb_cy = (dy * 2.f - 0.5f + i) * STRIDE[oi];
                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;
                        temp_mask.detectionConfidence = confidence;
                        temp_mask.top = pb_cx - pb_w * 0.5f;
                        temp_mask.left = pb_cy - pb_h * 0.5f;
                        temp_mask.width = pb_w;
                        temp_mask.height = pb_h;
                        temp_mask.mask_size = sizeof(float) * 3 * NUM_KEYPOINTS;
                        temp_mask.mask_height = 1;
                        temp_mask.mask_width = 3*NUM_KEYPOINTS;
                        temp_mask.mask = new float[3*NUM_KEYPOINTS];
                        for(int l = 0; l < NUM_KEYPOINTS; l ++)
                        {
                            temp_mask.mask[3*l + 0] = (layer_out[(spacial_size * (c * channels + l * 3 + 6)) + index] * 2 - 0.5 + j) * STRIDE[oi];
                            temp_mask.mask[3*l + 1] = (layer_out[(spacial_size * (c * channels + l * 3 + 7)) + index] * 2 - 0.5 + i) * STRIDE[oi];
                            temp_mask.mask[3*l + 2] = sigmoid(layer_out[spacial_size * (c * channels + l * 3 + 8) + index]); // this is confidence of the poin
                        }
                        preMpoints.push_back(temp_mask);
                    } // end of if

                } // end of for j
            } // end of for i

        } // end of for c

    } // end of for oi number of network outputs
}

//
void nms(std::vector<NvDsInferInstanceMaskInfo> &input_boxes, float NMS_THRESH)
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




static void leftTrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "\nFile does not exist: " << fileName << std::endl;
        return false;
    }
    return true;
}

std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType)
{
    assert(fileExists(weightsFilePath));
    std::cout << "\nLoading pre-trained weights" << std::endl;

    std::vector<float> weights;

    if (weightsFilePath.find(".weights") != std::string::npos) {
        std::ifstream file(weightsFilePath, std::ios_base::binary);
        assert(file.good());
        std::string line;

        if (networkType.find("yolov2") != std::string::npos && networkType.find("yolov2-tiny") == std::string::npos)
        {
            // Remove 4 int32 bytes of data from the stream belonging to the header
            file.ignore(4 * 4);
        }
        else
        {
            // Remove 5 int32 bytes of data from the stream belonging to the header
            file.ignore(4 * 5);
        }

        char floatWeight[4];
        while (!file.eof())
        {
            file.read(floatWeight, 4);
            assert(file.gcount() == 4);
            weights.push_back(*reinterpret_cast<float*>(floatWeight));
            if (file.peek() == std::istream::traits_type::eof()) break;
        }
    }

    else if (weightsFilePath.find(".wts") != std::string::npos) {
        std::ifstream file(weightsFilePath);
        assert(file.good());
        int32_t count;
        file >> count;
        assert(count > 0 && "\nInvalid .wts file.");

        uint32_t floatWeight;
        std::string name;
        uint32_t size;

        while (count--) {
            file >> name >> std::dec >> size;
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                file >> std::hex >> floatWeight;
                weights.push_back(*reinterpret_cast<float *>(&floatWeight));
            };
        }
    }

    else {
        std::cerr << "\nFile " << weightsFilePath << " is not supported" << std::endl;
        std::abort();
    }

    std::cout << "Loading weights of " << networkType << " complete"
            << std::endl;
    std::cout << "Total weights read: " << weights.size() << std::endl;
    return weights;
}

std::string dimsToString(const nvinfer1::Dims d)
{
    std::stringstream s;
    assert(d.nbDims >= 1);
    s << "[";
    for (int i = 0; i < d.nbDims - 1; ++i)
        s << d.d[i] << ", ";
    s << d.d[d.nbDims - 1] << "]";

    return s.str();
}

int getNumChannels(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[0];
}

void printLayerInfo(
    std::string layerIndex, std::string layerName, std::string layerInput, std::string layerOutput, std::string weightPtr)
{
    std::cout << std::setw(8) << std::left << layerIndex << std::setw(30) << std::left << layerName;
    std::cout << std::setw(20) << std::left << layerInput << std::setw(20) << std::left << layerOutput;
    std::cout << weightPtr << std::endl;
}


//CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPlate)
//CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParesYoloPlate)



/*
 * NvDsInferObjectDetectionInfo
 * NvDsInferInstanceMaskInfo
 *
 * */