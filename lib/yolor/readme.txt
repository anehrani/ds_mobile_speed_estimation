for deestream 6.1 run:

CUDA_VER=11.6 make -C .


for older versions: 

DeepStream 6.0.1 / 6.0 on x86 platform
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
DeepStream 6.1 on Jetson platform
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
DeepStream 6.0.1 / 6.0 on Jetson platform
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo




INT8 calibration
1. Install OpenCV
sudo apt-get install libopencv-dev
2. Compile/recompile the nvdsinfer_custom_impl_Yolo lib with OpenCV support
DeepStream 6.1 on x86 platform

CUDA_VER=11.6 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
DeepStream 6.0.1 / 6.0 on x86 platform

CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
DeepStream 6.1 on Jetson platform

CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
DeepStream 6.0.1 / 6.0 on Jetson platform

CUDA_VER=10.2 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
3. For COCO dataset, download the val2017, extract, and move to DeepStream-Yolo folder
Select 1000 random images from COCO dataset to run calibration

mkdir calibration
for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do \
    cp ${jpg} calibration/; \
done
Create the calibration.txt file with all selected images

realpath calibration/*jpg > calibration.txt
Set environment variables

export INT8_CALIB_IMG_PATH=calibration.txt
export INT8_CALIB_BATCH_SIZE=1
Edit the config_infer file

...
model-engine-file=model_b1_gpu0_fp32.engine
#int8-calib-file=calib.table
...
network-mode=0
...
To

...
model-engine-file=model_b1_gpu0_int8.engine
int8-calib-file=calib.table
...
network-mode=1
...
Run

deepstream-app -c deepstream_app_config.txt

