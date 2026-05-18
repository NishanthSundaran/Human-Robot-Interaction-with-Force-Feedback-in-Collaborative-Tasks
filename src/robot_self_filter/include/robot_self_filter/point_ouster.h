#include <pcl/pcl_macros.h>

////////////////// OS0-128 Data structure support //////////////////
/** Ouster Point struct */
struct EIGEN_ALIGN16 PointOuster {
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
/** Register as pointcloud struct for using in pcl::createMapping method */
POINT_CLOUD_REGISTER_POINT_STRUCT(PointOuster,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (std::uint32_t, t, t) // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                  (std::uint16_t, reflectivity, reflectivity)
                                  (std::uint8_t, ring, ring)
                                  (std::uint16_t, ambient, ambient)
                                  (std::uint32_t, range, range)
)