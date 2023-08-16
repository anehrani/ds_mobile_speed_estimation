//
// Created by ek on 23.01.2023.
//

#ifndef YOLOV7NUMPLATE_SPEEDESTIMATIONPLG_H
#define YOLOV7NUMPLATE_SPEEDESTIMATIONPLG_H


#include <map>
#include <math.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <memory>
#include <numeric>

#include "nvdsmeta.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#define NUM_KEYPOINTS 5
#define MODEL_INPUT_WIDTH 320
#define MODEL_INPUT_HEIGHT 320

namespace speedEstimation{

    constexpr auto MAX_INACTIVE_INTERVAL = 5;
    const int MAX_TRACK_AGE = 25;
    const int MIN_TRACK_AGE = 8;
    const int SPEED_CALCULATION_PERIOD = 1;
    const int METADATA_UPDATE_PERIOD = 1;

    const int FPS = 15;

    const float depth_estimation_threshold = .4; // tolerance for different depth estimation of same car

    // camera specific parameters---------------------------------------
    const float LENS_FOCAL_LENGTH = 25; //6; // in mm
    const float LICENCE_PLATE_SIZE[2] = { 520, 120 }; // license plate width-height in millimeter
    /*
     * NOTE: frames were resized into 1920 x 1080; pixels size should adjusted according to this resize
     * target pixel size = original pixel size x target size/ original size
     * */
    const float CAMERA_PIXEL_SIZE = 2 * 0.00274;  //0.00145 * 2; // size of sensor pixel in mm

    const float transformation_uv_cc[9] = {
            0.16666667, 0.00000000e+00, -160,
            0.00000000e+00,  0.16666667 ,  -90,
            0.00000000e+00, 0.00000000e+00, 1.00000000e+00
    }; // NOTE: the inverse of the intrinsic matrix K^-1

    typedef struct point_ {
        float x;
        float y;
        float z;
        float score;
    } objectPoint;

    typedef struct {
        float left;
        float top;
        float width;
        float height;
        float detectionConfidence;

        float right() { return left + width; }
        float bottom() { return top + height; }
        float center_x() { return left + width / 2.; }
        float center_y() { return top + height / 2.; }
    } BBox;

    typedef struct {
        int frame_num;
        float depth;
        objectPoint landmark[NUM_KEYPOINTS];
        BBox NumberPlate; // this for tracking
        BBox Vehicle;
        objectPoint center_3d;

    } trackData;


    typedef struct _track_obj_block {
        float avg_speed = -1;
        std::vector<trackData> track_positions;
        // areas that object is in
        int age;
        int cuTrackSize;
        bool track_status = 1;
        int source_id;
        int inactive_depth = 0;

    } track_obj_block;

    static std::unordered_map<int , track_obj_block> tracking_metadata;

    inline std::unique_ptr<std::unordered_map<int , track_obj_block>> tracking_metadata_ptr = \
            std::make_unique<std::unordered_map<int , track_obj_block>>(tracking_metadata);

    int ProcessTrackMetaData( NvDsObjectMeta* , int64 frameNumber);

    int  calculate_speed_d( track_obj_block& track_metadata_t );

    objectPoint transform_uv2cc( objectPoint landmark );

    /* calculating depth */
    float calculate_depth( trackData & );

    bool calculate_displacement( trackData & );

    void update_track_metadata( int );

    void process_data(std::vector<float> & input, float &);


}


float Euclidian_distance(float &x1, float &y1, float &x2, float &y2);










namespace speedCalculation{






}




#endif //YOLOV7NUMPLATE_SPEEDESTIMATIONPLG_H
