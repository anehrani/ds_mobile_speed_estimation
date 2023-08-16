//
// Created by ek on 23.01.2023.
//

#include "speedEstimationplg.h"



int speedEstimation::ProcessTrackMetaData(NvDsObjectMeta* newObjFlow, int64 frameNumber){


    /* speed calculation goes here */
    std::cout << "frame_number: "  << frameNumber  <<  std::endl;

//    if (frameNumber == 134)
//        std::cout<< " fr num 134 \n";

    /* if already entered the area*/
    trackData tmp_trackData;
    track_obj_block tmp_trk_block;

    tmp_trackData.Vehicle.left = newObjFlow->parent->rect_params.left;
    tmp_trackData.Vehicle.top = newObjFlow->parent->rect_params.top;
    tmp_trackData.Vehicle.width = newObjFlow->parent->rect_params.width;
    tmp_trackData.Vehicle.height = newObjFlow->parent->rect_params.height;
    tmp_trackData.Vehicle.detectionConfidence = newObjFlow->parent->confidence;

    tmp_trackData.NumberPlate.left = newObjFlow->rect_params.left;
    tmp_trackData.NumberPlate.top = newObjFlow->rect_params.top;
    tmp_trackData.NumberPlate.width = newObjFlow->rect_params.width;
    tmp_trackData.NumberPlate.height = newObjFlow->rect_params.height;
    tmp_trackData.NumberPlate.detectionConfidence = newObjFlow->confidence;

    float whRation = std::max(tmp_trackData.Vehicle.width, tmp_trackData.Vehicle.height);
    for (int i=0;i<NUM_KEYPOINTS; i++){
        tmp_trackData.landmark[i].x = tmp_trackData.Vehicle.left + newObjFlow->mask_params.data[3*i] * whRation / MODEL_INPUT_WIDTH;
        tmp_trackData.landmark[i].y = tmp_trackData.Vehicle.top + newObjFlow->mask_params.data[3*i+1]* whRation / MODEL_INPUT_HEIGHT;
        tmp_trackData.landmark[i].score = newObjFlow->mask_params.data[3*i+2];
    }
    tmp_trackData.frame_num = frameNumber;

    // todo: improve this part for high efficiecy
    tmp_trackData.depth = calculate_depth( tmp_trackData );
    if (tmp_trackData.depth < 0)
        return 1;


    auto exists = tracking_metadata_ptr->find(newObjFlow->parent->object_id);
    if (exists !=  tracking_metadata_ptr->end() ) {
        /* first controlling the track status */
        exists->second.age += 1;
        exists->second.cuTrackSize += 1;
        exists->second.track_status = 1; /* tells wether to continue to track */
        exists->second.inactive_depth = 0;
        // calculate the 3D center
        tmp_trackData.center_3d = transform_uv2cc( tmp_trackData.landmark[2] );

        exists->second.track_positions.push_back(tmp_trackData);
        // update criterias for speed estimation eligibility
        if ( exists->second.cuTrackSize > MIN_TRACK_AGE && exists->second.cuTrackSize % SPEED_CALCULATION_PERIOD ==0 ) {
            calculate_speed_d(exists->second);
            std::cout<< " <> speed updated <> " << newObjFlow->parent->object_id  << " track size: " \
            << exists->second.cuTrackSize << "  speed est: " << exists->second.avg_speed << std::endl;
        }

    }
        /* check if vehicle is inside the ROI */
    else {
        // the Id is entered for first time
        /* Filling out other necessary info */

        tmp_trk_block.age = 1;
        tmp_trk_block.cuTrackSize = 1;
        tmp_trk_block.track_status = 1; /* tells wether to continue to track */
        tmp_trk_block.inactive_depth = 0;
        // calculate the 3D center
        tmp_trackData.center_3d = transform_uv2cc( tmp_trackData.landmark[2] );

        tmp_trk_block.track_positions.push_back( tmp_trackData );

        tracking_metadata_ptr->insert(std::make_pair(newObjFlow->parent->object_id, tmp_trk_block));

    } // end of check exist

    // Updating the tracking matadata
    std::cout<< "size of metadata in speedEst: " << tracking_metadata_ptr->size() << std::endl;

    // updates should handle with care to increase the algorithm efficiency
    if (frameNumber % METADATA_UPDATE_PERIOD == 0 ) // make it very efficient, no need to update metadata every single frame
        speedEstimation::update_track_metadata( frameNumber );


    return 0;
}


float speedEstimation::calculate_depth( speedEstimation::trackData &objectPoint ){

    float depth=0;
    /*
    float* x1 = &objectPoint.landmark[0].x;
    float* y1 = &objectPoint.landmark[0].y;
    float* x2 = &objectPoint.landmark[1].x;
    float* y2 = &objectPoint.landmark[1].y;
    float* x3 = &objectPoint.landmark[4].x;
    float* y3 = &objectPoint.landmark[4].y;
    float* x4 = &objectPoint.landmark[3].x;
    float* y4 = &objectPoint.landmark[3].y;
    //
    float xuh = *x1 + (i/4.0) * (*x2 - *x1);
    float yuh = *y1 + (i/4.0) * (*y2 - *y1);
    float xdh = *x4 + (i/4.0) * (*x3 - *x4);
    float ydh = *y4 + (i/4.0) * (*y3 - *y4);
    //
    float xuw = *x1 + (i/4.0) * (*x4 - *x1);
    float yuw = *y1 + (i/4.0) * (*y4 - *y1);
    float xdw = *x2 + (i/4.0) * (*x3 - *x2);
    float ydw = *y2 + (i/4.0) * (*y3 - *y2);
    */
    // five points based averaging to reduce noise (at least this time)
    float avg_height = 0;
    float avg_width = 0;

    for (int i=0; i< 5; i++){
        float xuh = objectPoint.landmark[0].x + (i/4.0) * (objectPoint.landmark[1].x - objectPoint.landmark[0].x);
        float yuh = objectPoint.landmark[0].y + (i/4.0) * (objectPoint.landmark[1].y - objectPoint.landmark[0].y);
        float xdh = objectPoint.landmark[3].x + (i/4.0) * (objectPoint.landmark[4].x - objectPoint.landmark[3].x);
        float ydh = objectPoint.landmark[3].y + (i/4.0) * (objectPoint.landmark[4].y - objectPoint.landmark[3].y);
        //
        float xuw = objectPoint.landmark[0].x + (i/4.0) * (objectPoint.landmark[3].x - objectPoint.landmark[0].x);
        float yuw = objectPoint.landmark[0].y + (i/4.0) * (objectPoint.landmark[3].y - objectPoint.landmark[0].y);
        float xdw = objectPoint.landmark[1].x + (i/4.0) * (objectPoint.landmark[4].x - objectPoint.landmark[1].x);
        float ydw = objectPoint.landmark[1].y + (i/4.0) * (objectPoint.landmark[4].y - objectPoint.landmark[1].y);

        avg_width  += Euclidian_distance(xuw, yuw, xdw, ydw);;
        avg_height += Euclidian_distance(xuh, yuh, xdh, ydh);;
    }
    avg_height /=5.;
    avg_width  /=5.;
    // checking number plate width/height ratio
    depth =  LENS_FOCAL_LENGTH * LICENCE_PLATE_SIZE[1]/ ( avg_height * CAMERA_PIXEL_SIZE );
    float deviation = std::abs(avg_width / avg_height - LICENCE_PLATE_SIZE[0]/LICENCE_PLATE_SIZE[1]);

    std::cout<< "ratio deviation: " << deviation << std::endl;
    // too much noise appeared in calculation
    if (deviation > 1)
        return -1;


    if ( deviation < depth_estimation_threshold ){

        // update deoth based on width either
        depth  = 0.5 * (depth + LENS_FOCAL_LENGTH * LICENCE_PLATE_SIZE[0]/ ( avg_width * CAMERA_PIXEL_SIZE) );
    }

    return depth/1000;
}


speedEstimation::objectPoint speedEstimation::transform_uv2cc( objectPoint landmark ){
    objectPoint xyz;
    xyz.x = transformation_uv_cc[0] * landmark.x + transformation_uv_cc[1] * landmark.y + transformation_uv_cc[2];
    xyz.y = transformation_uv_cc[3] * landmark.x + transformation_uv_cc[4] * landmark.y + transformation_uv_cc[5];
    xyz.z = transformation_uv_cc[6] * landmark.x + transformation_uv_cc[7] * landmark.y + transformation_uv_cc[8];
    //float den = std::max(std::abs(xyz.x), std::abs(xyz.y));

    float den = std::sqrt(xyz.x*xyz.x + xyz.y*xyz.y + xyz.z*xyz.z);

    xyz.x /= den;
    xyz.y /= den;
    xyz.z /= den;


    //std::cout<< "x: " << xyz.x << " y: "<< xyz.y << " z: " << xyz.z << std::endl;
    return xyz;
}


int speedEstimation::calculate_speed_d( track_obj_block & track_data){
    /*
     *
     * */
    float deltaD = 0, deltaT=0;
    std::vector<float> speedEstimationData;
    float speedMean = 0;

    // speed based on the cc
    /*
     * There is not significant difference between the depth only case and vector difference case according to this experimet
     * */
    /*
    for ( int i=0; i< track_data.track_positions.size()/2; i+=2) {
        for (int j = track_data.track_positions.size()/2; j < track_data.track_positions.size(); j+=2) {
            float dx =  track_data.track_positions.at(j).depth * track_data.track_positions.at(j).center_3d.x - track_data.track_positions.at(i).depth *  track_data.track_positions.at(i).center_3d.x;
            float dy =  track_data.track_positions.at(j).depth * track_data.track_positions.at(j).center_3d.y - track_data.track_positions.at(i).depth *  track_data.track_positions.at(i).center_3d.y;
            float dz =  track_data.track_positions.at(j).depth * track_data.track_positions.at(j).center_3d.z - track_data.track_positions.at(i).depth *  track_data.track_positions.at(i).center_3d.z;
            // NOTE: angle between two consecitive positions must be close to zero and less that 10 deg in most cases
            float inner = track_data.track_positions.at(j).center_3d.x * track_data.track_positions.at(i).center_3d.x + \
                            track_data.track_positions.at(j).center_3d.y * track_data.track_positions.at(i).center_3d.y + \
                            track_data.track_positions.at(j).center_3d.z * track_data.track_positions.at(i).center_3d.z;
            std::cout << "inner product of consecutive positions: " << " (i, " << i << " j " << j << "): " << inner << std::endl;

            deltaD = std::sqrt( dx*dx + dy*dy + dz*dz );
            deltaT = std::abs( track_data.track_positions.at(j).frame_num - track_data.track_positions.at(i).frame_num);
            if (deltaT == 0 || deltaD == 0)
                continue;
            speedEstimationData.push_back( deltaD/ deltaT );
            speedMean += deltaD/ deltaT;
        }
    }

    */

    for ( int i=0; i< track_data.track_positions.size()/2; i+=1) {
        for (int j = track_data.track_positions.size()/2; j < track_data.track_positions.size(); j+=1) {
            deltaD = std::abs( track_data.track_positions.at(j).depth - track_data.track_positions.at(i).depth );
            deltaT = std::abs( track_data.track_positions.at(j).frame_num - track_data.track_positions.at(i).frame_num);
            if (deltaT == 0 || deltaD == 0)
                continue;
            speedEstimationData.push_back( deltaD/ deltaT );
            speedMean += deltaD/ deltaT;
        }
    }


    speedMean /= speedEstimationData.size();
    process_data( speedEstimationData, speedMean );

    // update only if there is a valid speed
    if (speedMean >= 0)
        if (track_data.avg_speed>=0)
            track_data.avg_speed =  ( 0.75 * track_data.avg_speed + 0.25 * std::floor( 3.6 * FPS * speedMean) ) ;
        else
            track_data.avg_speed = std::floor( 3.6 * FPS * speedMean );

    // all fine so far
    return 0;
}

void speedEstimation::update_track_metadata( int current_frame_number ){

    std::cout<< " ****** update metadata  ****** \n";
    std::vector<int > inactive_tracks;
    // iterate over the metadata ---
    // currently only check out the inactive depth
    for (auto iter = tracking_metadata_ptr->begin(); iter != tracking_metadata_ptr->end(); iter++  ) {
        //
        iter->second.inactive_depth = current_frame_number - iter->second.track_positions.back().frame_num;
        std::cout << " inactive length:-->>>   " << iter->second.inactive_depth << std::endl;

        if (iter->second.inactive_depth >= MAX_INACTIVE_INTERVAL) {
            iter->second.track_status = 0;
            inactive_tracks.emplace_back(iter->first);
        }
        // correcting number of stored frames !
        if (iter->second.cuTrackSize > MAX_TRACK_AGE ){
            std::cout << " extra frame removed   " << std::endl;
            iter->second.track_positions.erase( iter->second.track_positions.begin(), iter->second.track_positions.begin()\
            + (iter->second.cuTrackSize - MAX_TRACK_AGE  ));
            iter->second.cuTrackSize -= (iter->second.cuTrackSize - MAX_TRACK_AGE);
        }

    }
    //
    //std::cout<< " ****** remove from metadata  ****** \n";
    for (auto iter = inactive_tracks.begin(); iter != inactive_tracks.end(); iter++){
        tracking_metadata_ptr->erase(*iter);
        std::cout << " inactive object removed   " << std::endl;
    }
}


void speedEstimation::process_data(std::vector<float> & inputData, float & data_mean){
    float prev_mean = data_mean;
    std::sort(inputData.begin(), inputData.end());
    bool remove_min= true, remove_max=true;
    int counter = 0;
    int init_size = inputData.size();
    std::cout<< "data size: " << inputData.size() << std::endl;
    while ( remove_min || remove_max ){
        if ( remove_max ) {
            inputData.erase(inputData.begin() + inputData.size() - 1 );
            data_mean = std::accumulate(inputData.begin(), inputData.end(), 0.0) / inputData.size();
            std::cout<< "mean change with max: " << std::abs(prev_mean-data_mean)/data_mean  << std::endl;
            if (std::abs(prev_mean-data_mean)/prev_mean < .05){
                remove_max = false;
            }
            counter++ ;
            prev_mean = data_mean;
        }

        if ( remove_min ) {
            inputData.erase(inputData.begin());
            data_mean = std::accumulate(inputData.begin(), inputData.end(), 0.0) / inputData.size();
            std::cout<< "mean change with min: " << std::abs(prev_mean-data_mean)/data_mean  << std::endl;
            if (std::abs(prev_mean-data_mean)/prev_mean < .05){
                remove_min = false;
            }
            counter++ ;
            prev_mean = data_mean;

        }

        if (counter > 0.5 * init_size ){
            data_mean = 0;
            break;
        }

    }

    std::cout<< "removed: " << counter <<  std::endl;

}

float Euclidian_distance(float &x1, float &y1, float &x2, float &y2){
    return std::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

bool cmp(float a, float b){
    return a > b;
}

// this function failed -> no way to find such points!
bool speedEstimation::calculate_displacement( speedEstimation::trackData &objectPoint ){
    float x1 = objectPoint.landmark[0].x;
    float y1 = objectPoint.landmark[0].y;
    float x2 = objectPoint.landmark[1].x;
    float y2 = objectPoint.landmark[1].y;
    float x3 = objectPoint.landmark[4].x;
    float y3 = objectPoint.landmark[4].y;
    float x4 = objectPoint.landmark[3].x;
    float y4 = objectPoint.landmark[3].y;

    // checking out the vertical distance from other points too
    std::vector<float> vdist;
    float avg_dist = 0;
    for (int i=0; i< 5; i++){
        float xu = x1 + (i/4.0) * (x2 -x1);
        float yu = y1 + (i/4.0) * (y2 -y1);
        float xd = x4 + (i/4.0) * (x3 -x4);
        float yd = y4 + (i/4.0) * (y3 -y4);

        float ldist = Euclidian_distance(xu, yu, xd, yd);
        avg_dist += ldist;
        vdist.push_back( ldist  );
    }
    avg_dist /=5;

    float depth = LENS_FOCAL_LENGTH * LICENCE_PLATE_SIZE[1]/ ( avg_dist * CAMERA_PIXEL_SIZE );

    // angles
    float ea1 = ((x2-x1)*(x4-x1) + (y2-y1)*(y4-y1));
    float ea2 = ((x1-x2)*(x3-x2) + (y1-y2)*(y3-y2));
    float ea3 = ((x2-x3)*(x4-x2) + (y2-y3)*(y4-y3));
    float ea4 = ((x1-x4)*(x3-x4) + (y1-y4)*(y3-y4));
    /*
    // checking encircle
    float cx = objectPoint.landmark[2].x;//(x1 + x2 + x3 + x4)/4.;
    float cy = objectPoint.landmark[2].y;//(y1 + y2 + y3 + y4)/4.;
    float r1 = .5 * ( std::sqrt( (x1-cx)*(x1-cx) + (cy-y1)*(cy-y1) ) + std::sqrt( (x3-cx)*(x3-cx) + (cy-y3)*(cy-y3) ));
    float r2 = .5 * ( std::sqrt( (x2-cx)*(x2-cx) + (cy-y2)*(cy-y2) ) + std::sqrt( (x4-cx)*(x4-cx) + (cy-y4)*(cy-y4) ));
    float r = 0.5 *(r1 + r2);
    float ep1 = (x1 - cx)*(x1 - cx) + (y1 - cy)*(y1 - cy);
    float ep2 = (x2 - cx)*(x2 - cx) + (y2 - cy)*(y2 - cy);
    float ep3 = (x3 - cx)*(x3 - cx) + (y3 - cy)*(y3 - cy);
    float ep4 = (x4 - cx)*(x4 - cx) + (y4 - cy)*(y4 - cy);
    std::cout << " Error for p1: "<< ep1 - r*r<< std::endl;
    std::cout << " Error for p2: "<< ep2 - r*r<< std::endl;
    std::cout << " Error for p3: "<< ep3 - r*r<< std::endl;
    std::cout << " Error for p4: "<< ep4 - r*r<< std::endl;
    */
    std::cout << " Error for p1: "<< ea1  << std::endl;
    std::cout << " Error for p2: "<< ea2 << std::endl;
    std::cout << " Error for p3: "<< ea3 << std::endl;
    std::cout << " Error for p4: "<< ea4 << std::endl;


    return 1;
}

