#include <iostream>
#include "include/ds_yolo.h"

//
//./Yolov7numPlate /home/ek/EkinStash/testSpeedData/BritishSchool/test_01.h264 ../results/out_01.mp4

int main(int argc, char** argv) {

    yolo_deepstream(argc, argv);

    std::cout << "Finished!" << std::endl;
    return 0;
}
