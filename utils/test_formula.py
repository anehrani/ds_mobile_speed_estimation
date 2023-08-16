#
#
#
import numpy as np
import cv2 as cv
from glob import glob
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def calc_depth_from_width( width ):
    return  CAM_FOCAL * LICENCE_PLATE_SIZE[0]/ (width * CAMERA_SENSOR_SIZE )

def calc_depth_from_height( height ):
    return  CAM_FOCAL * LICENCE_PLATE_SIZE[1]/ ( height * CAMERA_SENSOR_SIZE )




# Define the FMKF class
class FMKF:
    def __init__(self, alpha, q, r):
        self.alpha = alpha  # forgetting factor
        self.q = q  # process noise variance
        self.r = r  # measurement noise variance
        self.x_hat = None  # estimated state
        self.p = None  # estimated covariance

    def update(self, z):
        if self.x_hat is None:
            # Initialize state estimate and covariance
            self.x_hat = np.array([[z]])
            self.p = np.array([[self.r]])
        else:
            # Predict
            x_hat_minus = self.x_hat
            p_minus = self.p + self.q

            # Update
            k = p_minus / (p_minus + self.r)
            self.x_hat = (1 - self.alpha) * x_hat_minus + self.alpha * (z - k * x_hat_minus)
            self.p = (1 - self.alpha) * p_minus

        return self.x_hat[0][0]


def process_data( input, mean ):

    prev_mean = mean

    remove_min = True
    remove_max = True

    input.sort()

    while remove_max or remove_min:

        if remove_max:
            input.pop(-1)

            mean = np.mean(input)
            mean_change_rate = np.abs(prev_mean - mean)/mean
            print("mean variation ", prev_mean - mean)
            print("mean stability ", mean_change_rate)
            if mean_change_rate < 0.1:
                remove_max = False
            prev_mean = mean

        if remove_min:
            input.pop(0)
            mean = np.mean(input)
            mean_change_rate = np.abs(prev_mean - mean)/mean
            print("mean variation ", prev_mean - mean)
            print("mean stability ",  mean_change_rate)
            if mean_change_rate < 0.1:
                remove_min = False
            prev_mean = mean

    return input


if __name__ == '__main__':

    images_path = "/home/ek/EkinStash/testSpeedData/testFrames/*.png"
    imgs_path = glob(images_path)



    CAM_FOCAL = 6
    LICENCE_PLATE_SIZE = [ 520, 120 ] # license plate width-height in millimeter
    CAMERA_SENSOR_SIZE = 0.00145; # size of sensor pixel in mm

    """    

    for img_path in imgs_path:
        img = cv.imread( img_path )
        cv.imshow("img", img); cv.waitKey(0)
        plt.imshow(img)
    """
    # cv.namedWindow("img", cv.WINDOW_FREERATIO)
    # img = cv.imread( imgs_path[0] )
    # cv.imshow("img", img); cv.waitKey(0)
    """
    Pu = [2391, 1377, 2511, 1377, 2511, 1405, 2389, 1405]
    Pd = [2378, 1469, 2504, 1475, 2502, 1502, 2375, 1497]
    P3 = [2198, 1252, 2292, 1253, 2292, 1272, 2197, 1273]
    # for point U
    width_top_u = np.sqrt((Pu[0] - Pu[2])**2 + (Pu[1] - Pu[3])**2 )
    width_down_u = np.sqrt((Pu[4] - Pu[6])**2 + (Pu[5] - Pu[7])**2 )
    height_left_u = np.sqrt((Pu[0] - Pu[6])**2 + (Pu[1] - Pu[7])**2 )
    height_right_u = np.sqrt((Pu[2] - Pu[4])**2 + (Pu[3] - Pu[5])**2 )
    #
    width_top_d = np.sqrt((Pd[0] - Pd[2])**2 + (Pd[1] - Pd[3])**2 )
    width_down_d = np.sqrt((Pd[4] - Pd[6])**2 + (Pd[5] - Pd[7])**2 )
    height_left_d = np.sqrt((Pd[0] - Pd[6])**2 + (Pd[1] - Pd[7])**2 )
    height_right_d = np.sqrt((Pd[2] - Pd[4])**2 + (Pd[3] - Pd[5])**2 )
    #
    width_top_3 = np.sqrt((P3[0] - P3[2])**2 + (P3[1] - P3[3])**2 )
    width_down_3 = np.sqrt((P3[4] - P3[6])**2 + (P3[5] - P3[7])**2 )
    height_left_3 = np.sqrt((P3[0] - P3[6])**2 + (P3[1] - P3[7])**2 )
    height_right_3 = np.sqrt((P3[2] - P3[4])**2 + (P3[3] - P3[5])**2 )

    depth = CAM_FOCAL * LICENCE_PLATE_SIZE[0]/ (height_left_3 * CAMERA_SENSOR_SIZE );

    xy1 = (0, 1)
    xy2 = (2, 1)
    xy3 = (2, 0)
    xy4 = (0, 0)

    r1 = .5 * np.sqrt((xy1[0] - xy3[0])**2 + (xy1[1] - xy3[1])**2)
    r2 = .5 * np.sqrt((xy2[0] - xy4[0])**2 + (xy2[1] - xy4[1])**2)
    r = .5* (r1+r2)

    cx = (xy1[0] + xy2[0] + xy3[0] + xy4[0])/4.
    cy = (xy1[1] + xy2[1] + xy3[1] + xy4[1])/4.

    ep1 = (xy1[0] - cx)**2 + (xy1[1] - cy)**2 - r*r
    ep2 = (xy2[0] - cx)**2 + (xy2[1] - cy)**2- r*r
    ep3 = (xy3[0] - cx)**2 + (xy3[1] - cy)**2- r*r
    ep4 = (xy4[0] - cx)**2 + (xy4[1] - cy)**2- r*r

    a1 = ((xy2[0] - xy1[0])*(xy4[0] - xy1[0]) + (xy2[1] - xy1[1])*(xy4[1] - xy1[1]))
    a2 = ((xy1[0] - xy2[0])*(xy3[0] - xy2[0]) + (xy1[1] - xy2[1])*(xy3[1] - xy2[1]))
    a3 = ((xy2[0] - xy3[0])*(xy4[0] - xy3[0]) + (xy2[1] - xy3[1])*(xy4[1] - xy3[1]))
    a4 = ((xy1[0] - xy4[0])*(xy3[0] - xy4[0]) + (xy1[1] - xy4[1])*(xy3[1] - xy4[1]))
    """

    # Define Kalman filter function

    d = [ 0.347748, 0.310669, 0.246737, 0.0255947, 0.361387, 0.54668, 0.924778, 0.699011, 0.15801, 0.347975, 0.290863, 0.349194, 0.404701, 0.0186081, 0.711103, 0.404026, 0.11311, 0.0556278, 0.180656, 9.06107, 9.88458, 9.71835, 10.3217, 19.4193, 19.2769, 0.227089,  0.310669, 0.246737, 0.0255947, 0.361387]
    # Define the initial state of the system
    # x = np.array( d )  # position
    v = 0  # velocity
    z = d[0]  # observation (measured position)
    dt = 1  # time step
    a = 0.5  # acceleration variance
    q = 0.1  # measurement variance


    dmean = np.mean(d)
    dvar = np.var(d)

    lb = dmean - .6*dmean
    lu = dmean + .6*dmean

    processed = process_data( d, dmean )

# Generate some simulated data
    true_positions = len(d) * [ 0.310669 ]
    observations = d #[p + np.random.normal(0, np.sqrt(0.1)) for p in true_positions]

    # Initialize FMKF object
    f = FMKF(1.99, 91, 50)


    estimated_positions = []
    for z in observations:
        x = f.update(z)
        estimated_positions.append(x)



    plt.plot(np.arange(len(true_positions)), estimated_positions, '.', "r")
    plt.plot(np.arange(len(true_positions)), true_positions, "-", "g")
    plt.plot(np.arange(len(true_positions)), observations, "^", "r")
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.legend(['Estimated position', 'Measured position'])
    plt.show()














    print("Finished! ")



