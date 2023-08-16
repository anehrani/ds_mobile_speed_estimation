//
// Created by ek on 16.01.2023.
//

#include "plateInfo.h"

#include <iostream>

plateInfo::objectBuffer *single_instance = nullptr;

std::mutex plateInfo::objectBuffer::m_instanceMutex;

plateInfo::objectBuffer * plateInfo::objectBuffer::getInstance() {

    const std::lock_guard<std::mutex> lock (m_instanceMutex);
    if (single_instance == nullptr)
        single_instance = new plateInfo::objectBuffer();
    return single_instance;
}

plateInfo::objectBuffer::~objectBuffer() {
    destroyInstance();
}

void plateInfo::objectBuffer::destroyInstance() {
    const std::lock_guard<std::mutex> lock(m_instanceMutex);
    if (single_instance)
    {
        delete single_instance;
        single_instance = nullptr;
    }
}

void plateInfo::objectBuffer::putList(NvDsPlatePointsMetaData & newFrameInfo ) {

    frame_meta_list.push_back( newFrameInfo );
    std::cout<< " obj list is updated ---- \n";
}

void plateInfo::objectBuffer::getList( std::vector<NvDsPlatePointsMetaData> & listInfo ) {

    listInfo = frame_meta_list;
    std::cout<< " list info is taken ---- \n";
}

void plateInfo::objectBuffer::clearList( ) {
    frame_meta_list.clear();
    std::cout<< " obj list is cleared ---- \n";
}

