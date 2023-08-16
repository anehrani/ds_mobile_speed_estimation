//
// Created by ek on 16.01.2023.
//

#ifndef YOLOV7NUMPLATE_PLATEINFO_H
#define YOLOV7NUMPLATE_PLATEINFO_H

#include <vector>
#include <mutex>
#include <atomic>
#include <thread>

#define NUM_KEYPOINTS 5


typedef struct {
    float x;
    float y;
    float score;
} PlatePoint;


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
    PlatePoint landmark[NUM_KEYPOINTS];
    BBox NumberPlate;
    BBox Vehicle;
} NvDsPlatePointsMetaData;



namespace plateInfo {

    class objectBuffer{
        /*
         *
         */
    public:
        static objectBuffer *getInstance();

        ~objectBuffer();
        void destroyInstance();

        void putList(NvDsPlatePointsMetaData & );
        void getList( std::vector<NvDsPlatePointsMetaData> & );
        void clearList();


    private:
        std::vector<NvDsPlatePointsMetaData> frame_meta_list;
        static std::mutex m_instanceMutex;

    };

};
#endif //YOLOV7NUMPLATE_PLATEINFO_H
