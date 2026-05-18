#include <pcl/pcl_macros.h>

////////////////// Hesai Data structure support //////////////////
// https://github.com/HesaiTechnology/HesaiLidar_General_ROS/blob/master/src/HesaiLidar_General_SDK/src/PandarGeneralRaw/include/pandarGeneral/point_types.h
/** Hesai Point struct */
struct EIGEN_ALIGN16 PointHesai {
        PCL_ADD_POINT4D;
        float intensity;
        unsigned int time;
        uint16_t ring;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
/** Register as pointcloud struct for using in pcl::createMapping method */
POINT_CLOUD_REGISTER_POINT_STRUCT(PointHesai,
                        (float, x, x)
                        (float, y, y)
                        (float, z, z)
                        (float, intensity, intensity)
                        (unsigned int, time, time)
                        (std::uint16_t, ring, ring)
)