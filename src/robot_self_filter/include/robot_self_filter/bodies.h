// ============================ bodies.h (Add Cylinder getters) ============================
#ifndef GEOMETRIC_SHAPES_POINT_INCLUSION_
#define GEOMETRIC_SHAPES_POINT_INCLUSION_

#include "robot_self_filter/shapes.h"
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <Eigen/Core>
#include <vector>

namespace robot_self_filter
{
namespace bodies
{

struct BoundingSphere
{
  tf2::Vector3 center;
  double radius;
};

// Abstract base
class Body
{
public:
  Body()
  {
    m_pose.setIdentity();
    m_type = shapes::UNKNOWN_SHAPE;
  }
  virtual ~Body() = default;

  shapes::ShapeType getType() const { return m_type; }
  void setPose(const tf2::Transform &pose)
  {
    m_pose = pose;
    updateInternalData();
  }
  const tf2::Transform &getPose() const { return m_pose; }
  void setDimensions(const shapes::Shape *shape)
  {
    useDimensions(shape);
    updateInternalData();
  }

  // For any shape
  virtual bool containsPoint(const tf2::Vector3 &p, bool verbose = false) const = 0;
  virtual bool intersectsRay(const tf2::Vector3 &origin,
                             const tf2::Vector3 &dir,
                             std::vector<tf2::Vector3> *intersections = nullptr,
                             unsigned int count = 0) const = 0;
  virtual double computeVolume() const = 0;
  virtual void computeBoundingSphere(BoundingSphere &sphere) const = 0;

protected:
  virtual void useDimensions(const shapes::Shape *shape) = 0;
  virtual void updateInternalData() = 0;

  shapes::ShapeType m_type;
  tf2::Transform    m_pose;
};

// Sphere
class Sphere : public Body
{
public:
  Sphere() : Body()
  {
    m_type     = shapes::SPHERE;
    m_scale    = 1.0;
    m_padding  = 0.0;
    m_radius   = 0.0;
    m_radiusU  = 0.0;
    m_radius2  = 0.0;
  }
  Sphere(const shapes::Shape *shape) : Sphere() { setDimensions(shape); }

  void setScale(double s)
  {
    m_scale = s;
    updateInternalData();
  }
  void setPadding(double p)
  {
    m_padding = p;
    updateInternalData();
  }
  double getScaledRadius() const { return m_radiusU; }

  bool containsPoint(const tf2::Vector3 &p, bool verbose = false) const override;
  bool intersectsRay(const tf2::Vector3 &origin,
                     const tf2::Vector3 &dir,
                     std::vector<tf2::Vector3> *intersections = nullptr,
                     unsigned int count = 0) const override;
  double computeVolume() const override;
  void computeBoundingSphere(BoundingSphere &sphere) const override;

protected:
  void useDimensions(const shapes::Shape *shape) override;
  void updateInternalData() override;

  double m_scale;
  double m_padding;
  double m_radius;
  double m_radiusU;
  double m_radius2;
  tf2::Vector3 m_center;
};

// Cylinder
class Cylinder : public Body
{
public:
  Cylinder() : Body()
  {
    m_type          = shapes::CYLINDER;
    m_scale_rad     = 1.0;
    m_scale_vert    = 1.0;
    m_padding_rad   = 0.0;
    m_padding_vert  = 0.0;
    m_length        = 0.0;
    m_length2       = 0.0;
    m_radius        = 0.0;
    m_radiusU       = 0.0;
    m_radius2       = 0.0;
    m_radiusB       = 0.0;
    m_radiusBSqr    = 0.0;
    m_d1            = 0.0;
    m_d2            = 0.0;
  }
  Cylinder(const shapes::Shape *shape) : Cylinder() { setDimensions(shape); }

  void setScale(double scale_rad, double scale_vert)
  {
    m_scale_rad  = scale_rad;
    m_scale_vert = scale_vert;
    updateInternalData();
  }
  void setPadding(double pad_rad, double pad_vert)
  {
    m_padding_rad  = pad_rad;
    m_padding_vert = pad_vert;
    updateInternalData();
  }

  // **Add these getters so we can use them in the marker code**
  double getScaledRadius() const { return m_radiusU; }
  double getScaledHalfLength() const { return m_length2; }

  bool containsPoint(const tf2::Vector3 &p, bool verbose = false) const override;
  bool intersectsRay(const tf2::Vector3 &origin,
                     const tf2::Vector3 &dir,
                     std::vector<tf2::Vector3> *intersections = nullptr,
                     unsigned int count = 0) const override;
  double computeVolume() const override;
  void computeBoundingSphere(BoundingSphere &sphere) const override;

protected:
  void useDimensions(const shapes::Shape *shape) override;
  void updateInternalData() override;

  double m_scale_rad, m_scale_vert;
  double m_padding_rad, m_padding_vert;

  double m_length, m_length2, m_radius, m_radiusU, m_radius2;
  double m_radiusB, m_radiusBSqr;
  double m_d1, m_d2;

  tf2::Vector3 m_center;
  tf2::Vector3 m_normalH;
  tf2::Vector3 m_normalB1;
  tf2::Vector3 m_normalB2;
};

// Box
class Box : public Body
{
public:
  Box() : Body()
  {
    m_type = shapes::BOX;
    for (int i = 0; i < 3; i++)
    {
      m_scale[i]   = 1.0;
      m_padding[i] = 0.0;
    }
    m_length  = m_width  = m_height  = 0.0;
    m_length2 = m_width2 = m_height2 = 0.0;
    m_radius2 = 0.0;
    m_radiusB = 0.0;
  }
  Box(const shapes::Shape *shape) : Box() { setDimensions(shape); }

  void setScale(double sx, double sy, double sz)
  {
    m_scale[0] = sx; m_scale[1] = sy; m_scale[2] = sz;
    updateInternalData();
  }
  void setPadding(double px, double py, double pz)
  {
    m_padding[0] = px; m_padding[1] = py; m_padding[2] = pz;
    updateInternalData();
  }

  double getScaledHalfLength() const { return m_length2; }
  double getScaledHalfWidth()  const { return m_width2; }
  double getScaledHalfHeight() const { return m_height2; }

  bool containsPoint(const tf2::Vector3 &p, bool verbose = false) const override;
  bool intersectsRay(const tf2::Vector3 &origin,
                     const tf2::Vector3 &dir,
                     std::vector<tf2::Vector3> *intersections = nullptr,
                     unsigned int count = 0) const override;
  double computeVolume() const override;
  void computeBoundingSphere(BoundingSphere &sphere) const override;

protected:
  void useDimensions(const shapes::Shape *shape) override;
  void updateInternalData() override;

  double m_scale[3];   // x,y,z
  double m_padding[3]; // x,y,z

  double m_length, m_width, m_height;
  double m_length2, m_width2, m_height2;
  double m_radius2, m_radiusB;

  tf2::Vector3 m_center;
  tf2::Vector3 m_normalL;
  tf2::Vector3 m_normalW;
  tf2::Vector3 m_normalH;

  tf2::Vector3 m_corner1;
  tf2::Vector3 m_corner2;
};

// Mesh
class ConvexMesh : public Body
{
public:
  ConvexMesh();
  ConvexMesh(const shapes::Shape *shape);
  ~ConvexMesh() override = default;

  void setScale(double s) { m_scale = s; updateInternalData(); }
  void setPadding(double p) { m_padding = p; updateInternalData(); }

  bool containsPoint(const tf2::Vector3 &p, bool verbose = false) const override;
  bool intersectsRay(const tf2::Vector3 &origin,
                     const tf2::Vector3 &dir,
                     std::vector<tf2::Vector3> *intersections = nullptr,
                     unsigned int count = 0) const override;
  double computeVolume() const override;
  void computeBoundingSphere(BoundingSphere &sphere) const override;

  const std::vector<unsigned int> &getTriangles() const { return m_triangles; }
  const std::vector<tf2::Vector3> &getScaledVertices() const { return m_scaledVertices; }

protected:
  void useDimensions(const shapes::Shape *shape) override;
  void updateInternalData() override;

  bool isPointInsidePlanes(const tf2::Vector3 &point) const;
  unsigned int countVerticesBehindPlane(const Eigen::Vector4d &planeNormal) const;

  std::vector<Eigen::Vector4d> m_planes;
  std::vector<tf2::Vector3>    m_vertices;
  std::vector<tf2::Vector3>    m_scaledVertices;
  std::vector<unsigned int>    m_triangles;
  tf2::Transform               m_iPose;

  tf2::Vector3 m_center;
  tf2::Vector3 m_meshCenter;
  double       m_meshRadiusB;

  tf2::Vector3 m_boxOffset;
  Box          m_boundingBox;
  double       m_radiusB;
  double       m_radiusBSqr;

  double m_scale   = 1.0;
  double m_padding = 0.0;
};

Body* createBodyFromShape(const shapes::Shape *shape);
void mergeBoundingSpheres(const std::vector<BoundingSphere> &spheres, BoundingSphere &mergedSphere);

} // namespace bodies
} // namespace robot_self_filter

#endif
