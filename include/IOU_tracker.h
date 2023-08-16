//
// Created by ek on 24.01.2023.
//

#ifndef YOLOV7NUMPLATE_IOU_TRACKER_H
#define YOLOV7NUMPLATE_IOU_TRACKER_H


#include <map>
#include <math.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <memory>
#include "plateInfo.h"



/******************************************************************************
* STRUCTS
******************************************************************************/

// rough initial setting

namespace trackingAg {

    // init parameters
    //const int NUM_KEYPOINTS = 5;
    const float sigma_l = 0;		// low detection threshold
    const float sigma_h = 0.2;		// high detection threshold
    const float sigma_iou = 0.00000000001;	// IOU threshold
    const float t_min = 2;		// minimum track length in frames



    constexpr auto MAX_INACTIVE_INTERVAL = 5;
    const int MAX_TRACK_AGE = 20;
    const int MIN_TRACK_AGE = 10;
    const int FPS = 20;

    const float depth_estimation_threshold = .4; // tolerance for different depth estimation of same car

    // camera specific parameters---------------------------------------
    const float CAMERA_FOCAL_LENGTH = 6; // in mm
    const float LICENCE_PLATE_SIZE[2] = { 350, 115 }; // license plate width-height in millimeter
    const float CAMERA_SENSOR_SIZE = 0.00145; // size of sensor pixel in mm

    typedef struct point_ {
        float x;
        float y;
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
    } trackData;


    typedef struct _track_obj_block {
        float avg_speed = 0;
        std::vector<trackData> track_positions;
        // areas that object is in
        int age;
        int cuTrackSize;
        bool track_status = 1;
        int source_id;
        int inactive_depth = 0;

    } track_obj_block;


    static std::unordered_map<int , track_obj_block> tracking_metadata;


    // Return the IoU between two boxes
    inline float intersectionOverUnion(BBox box1, BBox box2);

    // Returns the index of the bounding box with the highest IoU
    inline int highestIOU(trackData box, std::vector<trackData> boxes , float & highest);

    // Starts IOUT tracker
    bool track_iou( std::vector<trackingAg::trackData> detections );
    // Give an ID to the result tracks from "track_iou"
    // Method useful the way IOU is implemented in Python
    //void enumerate_tracks();
    float calculate_speed_d( track_obj_block& track_metadata_tr );

    /* calculating depth */
    float calculate_depth( trackData & );


    void update_track_metadata( std::unordered_map<int64_t , track_obj_block> * track_metadata_tr , int );
    //void runEstimation(NvDsFrameMeta *frameMeta  );





}



#endif //YOLOV7NUMPLATE_IOU_TRACKER_H
