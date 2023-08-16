//
// Created by ek on 28.07.2022.
//

#ifndef TINYYOLOV2_DS_YOLO_H
#define TINYYOLOV2_DS_YOLO_H

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"

//#include "nvds_analytics_meta.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
// --------------------------------
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"

const int enable_batch_process=1;


static const std::string ll_lib_file="/opt/nvidia/deepstream/deepstream-6.1/lib/libnvds_nvmultiobjecttracker.so";
// ll-config-file required to set different tracker types
static const std::string ll_config_file="../cfg/config_tracker_NvDCF_accuracy.yml";
//ll-config-file=config_tracker_DeepSORT.yml


static const int tracker_width=640;
static const int tracker_height=384;


static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data);
static gboolean bus_call (GstBus * bus, GstMessage * msg, gpointer data);
int yolo_deepstream (int argc, char *argv[]);

#endif //DS_YOLO_H
