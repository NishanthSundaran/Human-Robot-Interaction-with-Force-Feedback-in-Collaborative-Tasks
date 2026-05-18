/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2008, Willow Garage
*  All rights reserved.
* 
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
* 
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/** \author Ioan Sucan */

// Standard C/C++
#include <cstdio>
#include <random>
#include <chrono>
#include <thread>

// ROS2 Core
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

// TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// Robot Self Filter (assumes a ROS2 port exists)
#include "robot_self_filter/self_mask.h"

class TestSelfFilter : public rclcpp::Node
{
public:
  TestSelfFilter()
  : Node("test_self_filter"),
    buffer_(this->get_clock()),
    tf_listener_(buffer_),
    id_(1)
  {
    // Create publisher for visualization markers
    vm_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "visualization_marker", rclcpp::QoS(10).keep_last(10240));

    // Setup link info for self-filter
    std::vector<robot_self_filter::LinkInfo> links;
    robot_self_filter::LinkInfo li;
    li.name = "base_link";
    li.padding = 0.05;
    li.scale = 1.0;
    links.push_back(li);

    // Create the SelfMask filter (pcl::PointXYZ). You may need to adapt
    // depending on how your ROS2 self_filter library is structured.
    sf_ = std::make_unique<robot_self_filter::SelfMask<pcl::PointXYZ>>(buffer_, links);
    RCLCPP_INFO(this->get_logger(), "TestSelfFilter node has been initialized.");
  }

  ~TestSelfFilter() override = default;

  // Callback to handle intersection points
  void gotIntersection(double x, double y, double z)
  {
    sendPoint(x, y, z);
  }

  // Publish a sphere marker at (x,y,z)
  void sendPoint(double x, double y, double z)
  {
    visualization_msgs::msg::Marker mk;
    mk.header.stamp = this->now();
    mk.header.frame_id = "base_link";
    mk.ns = "test_self_filter";
    mk.id = id_++;
    mk.type = visualization_msgs::msg::Marker::SPHERE;
    mk.action = visualization_msgs::msg::Marker::ADD;

    mk.pose.position.x = x;
    mk.pose.position.y = y;
    mk.pose.position.z = z;
    mk.pose.orientation.w = 1.0;

    mk.scale.x = mk.scale.y = mk.scale.z = 0.01;

    mk.color.a = 1.0;
    mk.color.r = 1.0;
    mk.color.g = 0.04;
    mk.color.b = 0.04;

    // Marker lifetime
    mk.lifetime = rclcpp::Duration(std::chrono::seconds(10));

    vm_pub_->publish(mk);
  }

  // Main routine
  void run()
  {
    // Generate point cloud
    pcl::PointCloud<pcl::PointXYZ> in;
    in.header.frame_id = "base_link";
    // Instead of ROS1 stamp, store as a uint64_t in PCL's timestamp if needed
    in.header.stamp = static_cast<uint64_t>(this->now().nanoseconds());
    
    const unsigned int N = 500000;
    in.points.resize(N);

    // Use C++11 random engine to replace drand48
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.5, 1.5);

    for (unsigned int i = 0; i < N; ++i)
    {
      in.points[i].x = dist(gen);
      in.points[i].y = dist(gen);
      in.points[i].z = dist(gen);
    }

    // Allow some time for TF caches or other setup
    for (unsigned int i = 0; i < 1000; ++i)
    {
      rclcpp::spin_some(shared_from_this());
      rclcpp::sleep_for(std::chrono::milliseconds(1));
    }

    auto start_time = std::chrono::steady_clock::now();

    // Mask the cloud
    std::vector<int> mask;
    // The callback parameter: bind gotIntersection => pass x,y,z
    sf_->maskIntersection(
      in, "laser_tilt_mount_link", 0.01, mask,
      [this](double x, double y, double z)
      {
        this->gotIntersection(x, y, z);
      });

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    double points_per_sec = static_cast<double>(N) / elapsed_sec;
    RCLCPP_INFO(this->get_logger(), "%.2f points per second", points_per_sec);

    // Count how many points got marked as INSIDE
    int inside_count = 0;
    for (int m : mask)
    {
      if (m == robot_self_filter::INSIDE)
      {
        inside_count++;
      }
    }
    RCLCPP_INFO(this->get_logger(), "Points inside: %d", inside_count);

    // Spin to allow seeing the markers
    rclcpp::spin(shared_from_this());
  }

private:
  // TF
  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Self filter
  std::unique_ptr<robot_self_filter::SelfMask<pcl::PointXYZ>> sf_;

  // Publisher
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vm_pub_;

  // Tracking marker ID
  int id_;
};

// Main entry point
int main(int argc, char** argv)
{
  // Initialize ROS2
  rclcpp::init(argc, argv);

  // Create our node
  auto node = std::make_shared<TestSelfFilter>();

  // Let things warm up
  rclcpp::sleep_for(std::chrono::seconds(1));

  // Run the main routine
  node->run();

  // On exit
  rclcpp::shutdown();
  return 0;
}
