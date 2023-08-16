//
// Created by ek on 28.07.2022.
//

#include "ds_yolo.h"
#include <iostream>
//#include "plateInfo.h"
#include "speedEstimationplg.h"
//#include "IOU_tracker.h"
#include "fast.h"

#define MAX_DISPLAY_LEN 64
#define PGIE_CLASS_ID_VEHICLE 1
#define PGIE_CLASS_ID_PERSON 14
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define MUXER_BATCH_TIMEOUT_USEC 40000


static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{

#if 1

    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    /* display stuff */
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    char file_name[128];
    GstMapInfo in_map_info;
    if (!gst_buffer_map (buf, &in_map_info, GST_MAP_READ)) {
        g_print ("Error: Failed to map gst buffer\n");
        gst_buffer_unmap (buf, &in_map_info);
        return GST_PAD_PROBE_OK;
    }
    NvBufSurface *surface = (NvBufSurface *)in_map_info.data;
    //

    cv::Mat rgba_mat;


    // -----------------------------------------------------------------
    //                         Vizualization
    // -----------------------------------------------------------------
    NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ);
    /* Cache the mapped data for CPU access */
    NvBufSurfaceSyncForCpu(surface, 0, 0); //will do nothing for unified memory type on dGPU

    // ------------ iteration over meta data ------------------------
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

        std::cout<< "----------------------------------------------------------------------: " <<frame_meta->frame_num <<std::endl;
        // this part for visualization ------ -
        guint height = surface->surfaceList[frame_meta->batch_id].height;
        guint width = surface->surfaceList[frame_meta->batch_id].width;
        cv::Mat nv12_mat = cv::Mat(height * 3 / 2, width, CV_8UC1,
                                   surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
                                   surface->surfaceList[frame_meta->batch_id].pitch);
        cv::cvtColor(nv12_mat, rgba_mat, cv::COLOR_YUV2BGR_NV12);
        // -------------------
        //
        int vis = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (!obj_meta)
                continue;
            // check the detector id
            if (obj_meta->unique_component_id == 1){
//                float left = obj_meta->rect_params.left;
//                float top = obj_meta->rect_params.top;
//                float right = left + obj_meta->rect_params.width;
//                float bottom = top + obj_meta->rect_params.height;

//                cv::rectangle(rgba_mat, cv::Point(left, top),
//                              cv::Point(right, bottom),
//                              cv::Scalar(0, 155, 150), 1);
//
//
//                cv::putText(rgba_mat, obj_meta->obj_label, cv::Point(left, top - 5),
//                            1, 2, cv::Scalar(0, 0, 200), 1);
            }

            else if (obj_meta->unique_component_id == 2){
                vis = 2;
                speedEstimation::ProcessTrackMetaData( obj_meta, frame_meta->frame_num );

            }

        }// object metalist


        if (vis==2) {
            std::cout << "size of metadata: " << speedEstimation::tracking_metadata_ptr->size() << std::endl;
            for (auto iter = speedEstimation::tracking_metadata_ptr->begin();
                 iter != speedEstimation::tracking_metadata_ptr->end(); ++iter) {
                /* plotting detected number plate for visual comparison*/
                if (iter->second.inactive_depth > 0)
                    continue;
                //
                float left = iter->second.track_positions.back().NumberPlate.left;
                float top = iter->second.track_positions.back().NumberPlate.top;
                float right = left + iter->second.track_positions.back().NumberPlate.width;
                float bottom = top + iter->second.track_positions.back().NumberPlate.height;
                float confidence = iter->second.track_positions.back().NumberPlate.detectionConfidence;
                float depth = iter->second.track_positions.back().depth; // change to meter
                float speed = iter->second.avg_speed; // change to meter

                //
                for (int j = 0; j < 5; j++) {
                    float px = iter->second.track_positions.back().landmark[j].x;// * rescalex;
                    float py = iter->second.track_positions.back().landmark[j].y;// * rescaley;
                    //cv::Rect Roi_1(iter->second.track_positions.back().landmark[j].x-5, iter->second.track_positions.back().landmark[j].y-5,\
                10, 10);
                    //cv::Mat croppedImg = rgba_mat(Roi_1);
                    //cv::cvtColor(croppedImg, croppedImg, cv::COLOR_BGR2GRAY);
                    cv::circle(rgba_mat, cv::Point(px, py), 1, cv::Scalar(0, 5, 250), 1);

                }

                cv::rectangle(rgba_mat, cv::Point(left, top),
                              cv::Point(right, bottom),
                              cv::Scalar(0, 155, 150), 1);

                cv::rectangle(rgba_mat, cv::Point(iter->second.track_positions.back().Vehicle.left, iter->second.track_positions.back().Vehicle.top),
                              cv::Point(iter->second.track_positions.back().Vehicle.left + iter->second.track_positions.back().Vehicle.width,
                                        iter->second.track_positions.back().Vehicle.top + iter->second.track_positions.back().Vehicle.height),
                              cv::Scalar(155, 1, 150), 1);


                cv::putText(rgba_mat, std::to_string(iter->first) + ": " + std::to_string(depth) + " : " + std::to_string(speed), cv::Point(left, top - 5),
                            1, 2, cv::Scalar(0, 200, 2), 2);

                sprintf(file_name, "../results/speed_depth_stream%2d_%03d.jpg", frame_meta->source_id,
                        frame_meta->frame_num);
                imwrite(file_name, rgba_mat);

            }
        }


#if 0
         display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
         NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
         display_meta->num_labels = 1;
         txt_params->display_text = static_cast<char*>( g_malloc0 (MAX_DISPLAY_LEN));
         offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
         offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);
//
         /* Now set the offsets where the string should appear */
         txt_params->x_offset = 10;
         txt_params->y_offset = 12;
//
//        /* Font , font-color and font-size */
         txt_params->font_params.font_name = "Serif";
         txt_params->font_params.font_size = 10;
         txt_params->font_params.font_color.red = 1.0;
         txt_params->font_params.font_color.green = 1.0;
         txt_params->font_params.font_color.blue = 1.0;
         txt_params->font_params.font_color.alpha = 1.0;

         /* Text background color */
         txt_params->set_bg_clr = 1;
         txt_params->text_bg_clr.red = 0.0;
         txt_params->text_bg_clr.green = 0.0;
         txt_params->text_bg_clr.blue = 0.0;
         txt_params->text_bg_clr.alpha = 1.0;

         nvds_add_display_meta_to_frame(frame_meta, display_meta);
#endif
    }
#endif

    return GST_PAD_PROBE_OK;
}

static gboolean bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS:
            g_print ("End of stream\n");
            g_main_loop_quit (loop);
            break;
        case GST_MESSAGE_ERROR:{
            gchar *debug;
            GError *error;
            gst_message_parse_error (msg, &error, &debug);
            g_printerr ("ERROR from element %s: %s\n",
                        GST_OBJECT_NAME (msg->src), error->message);
            if (debug)
                g_printerr ("Error details: %s\n", debug);
            g_free (debug);
            g_error_free (error);
            g_main_loop_quit (loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int yolo_deepstream (int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *nvdsanalytics=NULL, *vehicle_tracker=NULL,
            *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *sgie = NULL,  *nvvidconv = NULL,
            *queue=NULL, *nvvidconv2=NULL, *capsfilter=NULL, *encoder = NULL, *codeparser = NULL, *container = NULL,
            *nvosd = NULL;
    // all queus
    GstElement *queue0=NULL, *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
            *queue5 = NULL, *queue6 = NULL, *queue7 = NULL, *queue8=NULL;

    GstElement *fpsSink = nullptr;
    GstCaps *caps = NULL;
    GstElement *transform = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    /* Check input arguments */
    if (argc != 3) {
        g_printerr ("Usage: %s <H264 filename> and output file name <mp4>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("dstest1-pipeline");

    /* Source element for reading from the file */
    source = gst_element_factory_make ("filesrc", "file-source");

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    h264parser = gst_element_factory_make ("h265parse", "h265-parser");


    /* creating queue */
    queue = gst_element_factory_make( "queue", "queue");

    /* converter */
    nvvidconv2 = gst_element_factory_make("nvvideoconvert", "convertor2");

    /* */
    capsfilter = gst_element_factory_make("capsfilter", "capsfilter");

    /*  */
    encoder = gst_element_factory_make("avenc_mpeg4", "encoder");

    /* */
    codeparser = gst_element_factory_make("mpeg4videoparse", "mpeg4-parser");

    /* */
    container = gst_element_factory_make("qtmux", "qtmux");

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder"); //nvv4l2decoder_h265  
    // decoder = gst_element_factory_make("nvv4l2decoder_h265", "nvv4l2-decoder");
    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("streammux not be created. Exiting.\n");
        return -1;
    }

    /* introducing tracker */
    vehicle_tracker = gst_element_factory_make("nvtracker", "vehicle_tracking");

    if (vehicle_tracker== nullptr){
        g_print("tracker is not defined!\n ");
        return -1;
    }
    /* Use nvinfer to run inferencing on decoder's output,
     *
     * behaviour of inferencing is set through config file */
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
    sgie = gst_element_factory_make ("nvinfer", "secondary-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    /* Finally render the osd output */
    if(prop.integrated) {
        transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
    }
//    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
//    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    sink = gst_element_factory_make ("filesink", "filesink"); // filesink
    
    fpsSink = gst_element_factory_make ("fpsdisplaysink", "fps-display");
    if (nullptr == fpsSink) {
        g_print("problem with fpsSink!\n ");
        return -1;
    }

    /* analytics */
    nvdsanalytics = gst_element_factory_make("nvdsanalytics", "nvdsanalytics");
    if (nullptr==nvdsanalytics){
        g_print(" nvdsanalytics is null \n");
        return -1;
    }



    if (!source || !h264parser || !decoder || !pgie
        || !nvvidconv || !nvosd || !sink) {
        g_printerr ("One element among [pgie, source, h264parser, decoder, nvosd, sink] could not be created. Exiting.\n");
        return -1;
    }

    if(!transform && prop.integrated) {
        g_printerr ("One tegra element could not be created. Exiting.\n");
        return -1;
    }

    /* we set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", argv[1], NULL);

    g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    //g_object_set (G_OBJECT (pgie),
    //              "config-file-path", "../cfg/config_infer_primary_yoloV7.txt", NULL);
    g_object_set (G_OBJECT (pgie), "config-file-path",
                  "../cfg/config_infer_primary_yoloV7.txt", "unique-id",
                  1, NULL);
    g_object_set (G_OBJECT (sgie), "config-file-path",
                  "../cfg/config_infer_secondary_yoloV7.txt", "unique-id",
                  2, "process-mode", 2, NULL);

    // tracker settings
    //
    g_object_set (G_OBJECT (vehicle_tracker), CONFIG_GROUP_TRACKER_WIDTH, tracker_width, nullptr);
    g_object_set (G_OBJECT (vehicle_tracker), CONFIG_GROUP_TRACKER_HEIGHT, tracker_height, nullptr);
    g_object_set (G_OBJECT (vehicle_tracker), CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, ll_config_file.c_str(), nullptr);
    g_object_set (G_OBJECT (vehicle_tracker), CONFIG_GROUP_TRACKER_LL_LIB_FILE, ll_lib_file.c_str(), nullptr);
    g_object_set (G_OBJECT (vehicle_tracker), CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS,
                  enable_batch_process, nullptr);


    g_object_set (G_OBJECT (sink), "location", argv[2], NULL);
    g_object_set (G_OBJECT (sink), "sync", 1, NULL);
    g_object_set (G_OBJECT (sink), "async", 0, NULL);
    caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "I420", NULL);
    g_object_set (G_OBJECT (capsfilter), "caps", caps, NULL);
    g_object_set (G_OBJECT (encoder), "bitrate", 2000000, NULL);

    /* sinking */
    g_object_set (G_OBJECT (fpsSink), "text-overlay", FALSE, "video-sink", sink, "sync", FALSE, NULL);
    g_object_set (G_OBJECT (nvdsanalytics), "config-file", "../cfg/config_nvdsanalytics.txt", nullptr);

    queue0 = gst_element_factory_make ("queue", "queue0");
    queue1 = gst_element_factory_make ("queue", "queue1");
    queue2 = gst_element_factory_make ("queue", "queue2");
    queue3 = gst_element_factory_make ("queue", "queue3");
    queue4 = gst_element_factory_make ("queue", "queue4");
    queue5 = gst_element_factory_make ("queue", "queue5");
    queue6 = gst_element_factory_make ("queue", "queue6");
    queue7 = gst_element_factory_make ("queue", "queue7");
    queue8 = gst_element_factory_make ("queue", "queue8");

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    if(prop.integrated) {
        gst_bin_add_many (GST_BIN (pipeline),
                          source, h264parser, decoder, streammux, pgie, vehicle_tracker, sgie, nvdsanalytics,
                          queue0, queue1, queue2, queue3, queue4, queue5, queue6, queue7, queue8,
                          nvvidconv, nvosd, transform, sink, NULL);
    }
    else {
        // this is for pc
        gst_bin_add_many (GST_BIN (pipeline),
                          source, h264parser, decoder, streammux, queue1, pgie, queue2, vehicle_tracker,
                          queue0, sgie, queue3, nvdsanalytics, queue4, nvvidconv, queue5, nvosd, queue6,
                          nvvidconv2, queue7, capsfilter, encoder,
                          codeparser, container, queue8, fpsSink, NULL);
    }

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr ("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    if (!srcpad) {
        g_printerr ("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    /* we link the elements together */
    /* file-source -> h264-parser -> nvh264-decoder ->
     * nvinfer -> nvvidconv -> nvosd -> video-renderer */

    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

    if(prop.integrated) {
        if (!gst_element_link_many (streammux, queue1, pgie, queue2, vehicle_tracker, queue0, sgie, queue3, nvdsanalytics,
                                   queue4, nvvidconv, queue5, nvosd, queue6, transform, fpsSink, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
    }
    else {
        // this is for pc
        if (!gst_element_link_many (streammux, queue1, pgie, queue2, vehicle_tracker, queue0, sgie, queue3, nvdsanalytics,
                                    queue4, nvvidconv, queue5, nvosd, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }

//        if (!gst_element_link_many (streammux, pgie, nvdsanalytics,
//                                    nvvidconv, nvosd, queue, nvvidconv2, capsfilter, encoder,
//                                    codeparser, container, fpsSink, NULL)) {
//            g_printerr ("Elements could not be linked: 2. Exiting.\n");
//            return -1;
//        }

        if(!gst_element_link_many(nvosd, queue6, nvvidconv2, queue7, capsfilter, encoder, codeparser, container, queue8, fpsSink, NULL) ){
            g_printerr ("Elements could not be linked: 3. Exiting.\n");
            return -1;
        }


    }

    osd_sink_pad = gst_element_get_static_pad (nvdsanalytics, pad_name_src);
    if (!osd_sink_pad)
        g_print ("Unable to get src pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                           osd_sink_pad_buffer_probe, reinterpret_cast<gpointer>(fpsSink), NULL);
    gst_object_unref (osd_sink_pad);


    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    /*
    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                           osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref (osd_sink_pad);
    */
    /* Set the pipeline to "playing" state */
    g_print ("Now playing");
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}