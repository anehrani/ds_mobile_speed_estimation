//
// Created by ek on 24.01.2023.
//

#pragma once
#include <algorithm>
#include <iostream>
#include "IOU_tracker.h"






inline float trackingAg::intersectionOverUnion(BBox box1, BBox box2)
{
    float minx1 = box1.left;
    float maxx1 = box1.left + box1.width;
    float miny1 = box1.top;
    float maxy1 = box1.top+ box1.height;

    float minx2 = box2.left;
    float maxx2 = box2.left + box2.width;
    float miny2 = box2.top;
    float maxy2 = box2.top + box2.height;

    if (minx1 > maxx2 || maxx1 < minx2 || miny1 > maxy2 || maxy1 < miny2)
        return 0.0f;
    else
    {
        float dx = std::min(maxx2, maxx1) - std::max(minx2, minx1);
        float dy = std::min(maxy2, maxy1) - std::max(miny2, miny1);
        float area1 = (maxx1 - minx1)*(maxy1 - miny1);
        float area2 = (maxx2 - minx2)*(maxy2 - miny2);
        float inter = dx*dy; // Intersection
        float uni = area1 + area2 - inter; // Union
        float IoU = inter / uni;
        return IoU;
    }
//	return 0.0f;
}

inline int trackingAg::highestIOU(trackData box, std::vector<trackData> boxes, float & highest)
{
    float iou = 0;
    int index = -1;
    for (int i = 0; i < boxes.size(); i++)
    {
        iou = intersectionOverUnion(box.NumberPlate, boxes[i].NumberPlate);
        if ( iou >= highest)
        {
            highest = iou;
            index = i;
        }
    }
    return index;
}

bool trackingAg::track_iou(   std::vector<trackingAg::trackData> detections)
{
    //std::cout << "track_iou function" << std::endl;

    int newId = -1;
    int index;		// Index of the box with the highest IOU
    bool updated;	// Whether if a track was updated or not
    int numFrames = detections.size();
    std::vector<int> inactive_tracks;


    //std::cout << "map size: " << tracking_metadata.size() << "   Num of frames: > " << numFrames << std::endl;

    /// Update active tracks
    // trying to find intersection
    if (!tracking_metadata.empty() )
        for (auto tracki = tracking_metadata.begin(); tracki != tracking_metadata.end(); tracki++  ) {
            updated = false;        // Get the index of the detection with the highest IOU
            float highestIou = 0;
            // controlling the availability



            index = highestIOU(tracki->second.track_positions.back(), detections, highestIou);
            // Check is above the IOU threshold
            // std::cout << " --- IOU Best match = " << intersectionOverUnion(track.boxes.back(), frameBoxes[index]) << std::endl;
            if (index != -1 && highestIou >= sigma_iou) {
                // detection at index i belongs to this track
                tracki->second.track_positions.push_back(detections[index]);
                tracki->second.inactive_depth = 0;
                tracki->second.track_status = 1;
                tracki->second.track_positions.back().depth = calculate_depth( tracki->second.track_positions.back() );

                if (tracki->second.track_positions.size() > MIN_TRACK_AGE )
                    tracki->second.avg_speed = calculate_speed_d( tracki->second );

                //todo: if number of keepping tracks exceeds, remove initial ones?!

                updated = true;

            }
            else{
                tracki->second.inactive_depth += 1;

            }


            tracki->second.age +=1;

            // collecting inactive trakcs
            if (tracki->second.inactive_depth > MAX_INACTIVE_INTERVAL)
                inactive_tracks.push_back(tracki->first);


            // remove the index from detection
            if (updated)
                detections.erase(detections.begin() + index);



    } // End for active tracks

    // another search for remaining detections
    // todo: in case of no intersection, looking for closest number plate --> use assignment problem
    /* this can happen due to frame skipping, high speed etc */

    /// Create new tracks
    for (auto box : detections)
    {
        // trying to create new id
        for (int i=0; i< 2*tracking_metadata.size(); i++)
            if ( !(tracking_metadata.find(i) != tracking_metadata.end()) ) {
                newId = i;
                break;
            }

        if (newId == -1) newId = tracking_metadata.size();

        // adding new track
        track_obj_block newTrack;
        newTrack.track_status=1;
        newTrack.track_positions.push_back( box );
        newTrack.age = 1;
        newTrack.inactive_depth = 0;
        //
        // check if the car is not stright to camera
        newTrack.track_positions.back().depth =  calculate_depth( newTrack.track_positions.back() );

        tracking_metadata.insert(std::make_pair( newId, newTrack ) );

    }

    // find and remove inactive tracks
    for (auto iter = inactive_tracks.begin(); iter != inactive_tracks.end(); iter++){
        tracking_metadata.erase(*iter);
    }

    return true;

}


float trackingAg::calculate_depth( trackingAg::trackData &objectPoint ){

    float depth=0;

    float width_top = std::sqrt((objectPoint.landmark[1].x - objectPoint.landmark[0].x)*(objectPoint.landmark[1].x - objectPoint.landmark[0].x)  +  (objectPoint.landmark[1].y - objectPoint.landmark[0].y)*(objectPoint.landmark[1].y - objectPoint.landmark[0].y));
    float width_down = std::sqrt((objectPoint.landmark[3].x - objectPoint.landmark[4].x)*(objectPoint.landmark[3].x - objectPoint.landmark[4].x)  +  (objectPoint.landmark[3].y - objectPoint.landmark[4].y)*(objectPoint.landmark[3].y - objectPoint.landmark[4].y));
    float height_left = std::sqrt((objectPoint.landmark[3].x - objectPoint.landmark[0].x)*(objectPoint.landmark[3].x - objectPoint.landmark[0].x)  +  (objectPoint.landmark[3].y - objectPoint.landmark[0].y)*(objectPoint.landmark[3].y - objectPoint.landmark[0].y));
    float height_right = std::sqrt((objectPoint.landmark[1].x - objectPoint.landmark[4].x)*(objectPoint.landmark[1].x - objectPoint.landmark[4].x)  +  (objectPoint.landmark[1].y - objectPoint.landmark[4].y)*(objectPoint.landmark[1].y - objectPoint.landmark[4].y));


    float depthW = CAMERA_FOCAL_LENGTH * width_top/ (LICENCE_PLATE_SIZE[1] * CAMERA_SENSOR_SIZE );
    depthW += CAMERA_FOCAL_LENGTH * width_down/ (LICENCE_PLATE_SIZE[1] * CAMERA_SENSOR_SIZE );

    float depthH = CAMERA_FOCAL_LENGTH * height_left/ (LICENCE_PLATE_SIZE[1] * CAMERA_SENSOR_SIZE );
    depthH += CAMERA_FOCAL_LENGTH * height_right/ (LICENCE_PLATE_SIZE[1] * CAMERA_SENSOR_SIZE );


    float rate_of_orientation = 0;
    rate_of_orientation += sqrt( std::pow( objectPoint.NumberPlate.left - objectPoint.landmark[0].x ,2) + std::pow( objectPoint.NumberPlate.top  - objectPoint.landmark[0].y,2));
    rate_of_orientation += sqrt( std::pow( objectPoint.NumberPlate.left + objectPoint.NumberPlate.width - objectPoint.landmark[1].x ,2) + std::pow( objectPoint.NumberPlate.top - objectPoint.landmark[1].y,2));
    rate_of_orientation += sqrt( std::pow( objectPoint.NumberPlate.left + objectPoint.NumberPlate.width - objectPoint.landmark[4].x ,2) + std::pow( objectPoint.NumberPlate.top + objectPoint.NumberPlate.height - objectPoint.landmark[4].y,2));
    rate_of_orientation += sqrt( std::pow( objectPoint.NumberPlate.left - objectPoint.landmark[3].x ,2) + std::pow( objectPoint.NumberPlate.top  + objectPoint.NumberPlate.height  - objectPoint.landmark[3].y,2));

    if (rate_of_orientation > depth_estimation_threshold)
        depth = 0.5*depthH;
    else
        depth = .25 * ( depthH + depthW );


    return depth;
}


float trackingAg::calculate_speed_d( trackingAg::track_obj_block & track_data){
    /*
     *
     * */
    float deltaD = 0, deltaT=0;

    for ( int i=0; i< track_data.track_positions.size()/2; i+=2) {
        for (int j = track_data.track_positions.size()/2; j < track_data.track_positions.size(); j+=2) {
            deltaD += std::abs( track_data.track_positions.at(j).depth - track_data.track_positions.at(i).depth );
            deltaT += std::abs( track_data.track_positions.at(j).frame_num - track_data.track_positions.at(i).frame_num);
        }
    }

    float avg_speed = deltaD / deltaT;

    std::cout<< " dist: " << deltaD << " T: " << deltaT << "  speed: "<< avg_speed << std::endl;

    return avg_speed;


}