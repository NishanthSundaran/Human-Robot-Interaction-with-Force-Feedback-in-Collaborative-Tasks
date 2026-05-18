#include <pcl/pcl_macros.h>

////////////////// Hesai Data structure support //////////////////
// https://github.com/HesaiTechnology/HesaiLidar_General_ROS/blob/master/src/HesaiLidar_General_SDK/src/PandarGeneralRaw/include/pandarGeneral/point_types.h
/** Hesai Point struct */
struct EIGEN_ALIGN16 PointRobosense {
    PCL_ADD_POINT4D;
    float intensity;
    //uint16_t ring = 0;
    //double timestamp = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
/** Register as pointcloud struct for using in pcl::createMapping method */
POINT_CLOUD_REGISTER_POINT_STRUCT(PointRobosense,
                        (float, x, x)
                        (float, y, y)
                        (float, z, z)
                        (float, intensity, intensity)
                        //(std::uint16_t, ring, ring)
                        //(double, timestamp, timestamp)      
)