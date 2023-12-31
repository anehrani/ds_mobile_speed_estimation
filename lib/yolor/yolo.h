/*


 */

#ifndef _YOLO_H_
#define _YOLO_H_

#include "layers/convolutional_layer.h"
#include "layers/batchnorm_layer.h"
#include "layers/implicit_layer.h"
#include "layers/channels_layer.h"
#include "layers/shortcut_layer.h"
#include "layers/route_layer.h"
#include "layers/upsample_layer.h"
#include "layers/pooling_layer.h"
#include "layers/reorg_layer.h"
#include "layers/reduce_layer.h"
#include "layers/shuffle_layer.h"
#include "layers/softmax_layer.h"
#include "layers/cls_layer.h"
#include "layers/reg_layer.h"

#include "nvdsinfer_custom_impl.h"

struct NetworkInfo
{
    std::string inputBlobName;
    std::string networkType;
    std::string configFilePath;
    std::string wtsFilePath;
    std::string int8CalibPath;
    std::string deviceType;
    uint numDetectedClasses;
    int clusterMode;
    std::string networkMode;
};

struct TensorInfo
{
    std::string blobName;
    uint gridSizeX {0};
    uint gridSizeY {0};
    uint numBBoxes {0};
    float scaleXY;
    std::vector<float> anchors;
    std::vector<int> mask;
};

class Yolo : public IModelParser {
public:
    Yolo(const NetworkInfo& networkInfo);

    ~Yolo() override;

    bool hasFullDimsSupported() const override { return false; }

    const char* getModelName() const override {
        return m_ConfigFilePath.empty() ? m_NetworkType.c_str() : m_ConfigFilePath.c_str();
    }

    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition& network) override;

    nvinfer1::ICudaEngine *createEngine (nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config);

protected:
    const std::string m_InputBlobName;
    const std::string m_NetworkType;
    const std::string m_ConfigFilePath;
    const std::string m_WtsFilePath;
    const std::string m_Int8CalibPath;
    const std::string m_DeviceType;
    const uint m_NumDetectedClasses;
    const int m_ClusterMode;
    const std::string m_NetworkMode;

    uint m_InputH;
    uint m_InputW;
    uint m_InputC;
    uint64_t m_InputSize;
    uint m_NumClasses;
    uint m_LetterBox;
    uint m_NewCoords;
    uint m_YoloCount;
    float m_IouThreshold;
    float m_ScoreThreshold;
    uint m_TopK;

    std::vector<TensorInfo> m_YoloTensors;
    std::vector<std::map<std::string, std::string>> m_ConfigBlocks;
    std::vector<std::map<std::string, std::string>> m_ConfigNMSBlocks;
    std::vector<nvinfer1::Weights> m_TrtWeights;

private:
    NvDsInferStatus buildYoloNetwork(std::vector<float>& weights, nvinfer1::INetworkDefinition& network);

    std::vector<std::map<std::string, std::string>> parseConfigFile(const std::string cfgFilePath);

    void parseConfigBlocks();

    void parseConfigNMSBlocks();

    void destroyNetworkUtils();
};

#endif // _YOLO_H_
