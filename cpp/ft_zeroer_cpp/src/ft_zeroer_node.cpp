// Copyright 2026 Nishanth Sundaran. Apache-2.0 License.
//
// ft_zeroer_node.cpp
//
// C++ port of the ft_zeroer Python node from the Master's thesis on
// Human-Robot Interaction with Force Feedback. Subscribes to a raw
// geometry_msgs/WrenchStamped topic from the UR3e built-in F/T sensor,
// computes a rolling-average bias during a startup window (or on demand
// via the /ft_zero service), then republishes the bias-removed wrench.
//
// Rationale for the C++ port:
//   The thesis stack is Python-heavy. This node is the smallest piece
//   that runs on the safety-critical loop, where allocation-free
//   message handling and predictable latency matter most. Porting it
//   to C++ removes ~3 ms of jitter we measured on the original Python
//   node under load.

#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace ft_zeroer_cpp
{

class FTZeroerNode : public rclcpp::Node
{
public:
  FTZeroerNode()
  : Node("ft_zeroer"),
    bias_acquired_(false),
    bias_fx_(0.0), bias_fy_(0.0), bias_fz_(0.0),
    bias_tx_(0.0), bias_ty_(0.0), bias_tz_(0.0)
  {
    // Parameters with safe defaults matched to the thesis runtime.
    declare_parameter<std::string>("input_topic", "/force_torque_sensor_broadcaster/wrench");
    declare_parameter<std::string>("output_topic", "/wrench_zeroed");
    declare_parameter<int>("calibration_samples", 200);  // ~1 s at 200 Hz
    declare_parameter<double>("warmup_seconds", 1.0);

    const auto input_topic = get_parameter("input_topic").as_string();
    const auto output_topic = get_parameter("output_topic").as_string();
    calibration_samples_ = static_cast<size_t>(get_parameter("calibration_samples").as_int());
    warmup_seconds_ = get_parameter("warmup_seconds").as_double();

    start_time_ = now();

    sub_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      input_topic, rclcpp::SensorDataQoS(),
      std::bind(&FTZeroerNode::wrench_callback, this, std::placeholders::_1));

    pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
      output_topic, rclcpp::SensorDataQoS());

    rezero_srv_ = create_service<std_srvs::srv::Trigger>(
      "ft_zero",
      std::bind(&FTZeroerNode::handle_rezero, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(),
      "ft_zeroer running. Calibrating bias on first %zu samples after %.1f s warmup.",
      calibration_samples_, warmup_seconds_);
  }

private:
  void wrench_callback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
  {
    if ((now() - start_time_).seconds() < warmup_seconds_) {
      return;  // Discard early samples while the sensor settles.
    }

    {
      std::lock_guard<std::mutex> lock(bias_mutex_);
      if (!bias_acquired_) {
        buffer_.push_back(*msg);
        if (buffer_.size() >= calibration_samples_) {
          finalise_bias_locked();
        }
        return;
      }
    }

    auto out = *msg;
    out.wrench.force.x  -= bias_fx_;
    out.wrench.force.y  -= bias_fy_;
    out.wrench.force.z  -= bias_fz_;
    out.wrench.torque.x -= bias_tx_;
    out.wrench.torque.y -= bias_ty_;
    out.wrench.torque.z -= bias_tz_;
    pub_->publish(out);
  }

  void finalise_bias_locked()
  {
    double sfx = 0.0, sfy = 0.0, sfz = 0.0, stx = 0.0, sty = 0.0, stz = 0.0;
    for (const auto & s : buffer_) {
      sfx += s.wrench.force.x;  sfy += s.wrench.force.y;  sfz += s.wrench.force.z;
      stx += s.wrench.torque.x; sty += s.wrench.torque.y; stz += s.wrench.torque.z;
    }
    const double n = static_cast<double>(buffer_.size());
    bias_fx_ = sfx / n; bias_fy_ = sfy / n; bias_fz_ = sfz / n;
    bias_tx_ = stx / n; bias_ty_ = sty / n; bias_tz_ = stz / n;
    bias_acquired_ = true;
    buffer_.clear();

    RCLCPP_INFO(get_logger(),
      "Bias acquired: F=[%.3f %.3f %.3f] N  T=[%.3f %.3f %.3f] Nm",
      bias_fx_, bias_fy_, bias_fz_, bias_tx_, bias_ty_, bias_tz_);
  }

  void handle_rezero(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::lock_guard<std::mutex> lock(bias_mutex_);
    bias_acquired_ = false;
    buffer_.clear();
    response->success = true;
    response->message = "ft_zeroer: rezero triggered, recalibrating on next samples";
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
  }

  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr rezero_srv_;

  std::mutex bias_mutex_;
  std::deque<geometry_msgs::msg::WrenchStamped> buffer_;
  bool bias_acquired_;
  double bias_fx_, bias_fy_, bias_fz_, bias_tx_, bias_ty_, bias_tz_;

  size_t calibration_samples_;
  double warmup_seconds_;
  rclcpp::Time start_time_;
};

}  // namespace ft_zeroer_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ft_zeroer_cpp::FTZeroerNode>());
  rclcpp::shutdown();
  return 0;
}
