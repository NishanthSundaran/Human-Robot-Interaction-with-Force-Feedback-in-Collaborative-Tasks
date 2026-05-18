#include <pcl/pcl_macros.h>

/** Hesai Pandar Point struct */
struct EIGEN_ALIGN16 PointPandar {
        PCL_ADD_POINT4D;
        float intensity;
        double timestamp;
        uint16_t ring;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
/** Register as pointcloud struct for using in pcl::createMapping method */
POINT_CLOUD_REGISTER_POINT_STRUCT(PointPandar,
                        (float, x, x)
                        (float, y, y)
                        (float, z, z)
                        (float, intensity, intensity)
                        (double, timestamp, timestamp)
                        (std::uint16_t, ring, ring)
)
