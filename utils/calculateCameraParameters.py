
import numpy as np
import math


cam_info = {
    "H": 1920,
    "V": 1080,
    "diag":6.4,
    "FOV": 60, 
    "F" : 6
}

if __name__ == '__main__':

    theta  = math.atan(cam_info["V"]/cam_info["H"])
    width_mm  = cam_info["diag"] * math.cos(theta)
    height_mm = cam_info["diag"] * math.sin(theta)

    pixel_size_w = width_mm/cam_info["H"]
    pixel_size_h = height_mm/cam_info["V"]

    # Focal_length = (cam_info["diag"] / 2) / math.tan(cam_info["FOV"] * 3.14 / 360 / 2),

    print("pixel_size_w mm", pixel_size_w)
    print("pixel_size_h mm", pixel_size_h)

    K = np.array([[cam_info["F"], 0, 0.5* cam_info["H"]],
                  [0, cam_info["F"], 0.5* cam_info["V"]],
                  [0, 0, 1]])
    Kinv = np.linalg.inv(K)

    print("Kinv", Kinv)








    print(" Finished! ")
