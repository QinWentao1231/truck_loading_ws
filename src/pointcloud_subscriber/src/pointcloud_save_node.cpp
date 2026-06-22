#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl/conversions.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include <pcl_conversions/pcl_conversions.h>
#include <ament_index_cpp/get_package_share_directory.hpp> // 用于获取包路径

#include <filesystem>  // C++17 filesystem 支持

using namespace std;

class PointCloudSubscriber : public rclcpp::Node
{
public:
    PointCloudSubscriber() : Node("pointcloud_save_node"), frame_count_(0)
    {
        // 创建订阅者，订阅 "/lidar_points" 话题
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar_points1", 10,
            std::bind(&PointCloudSubscriber::pointCloudCallback, this, std::placeholders::_1)
        );
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 将 ROS 消息转换为 PCL 格式
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(*msg, cloud);

        // 将接收到的点云添加到合并点云中
        if (frame_count_ == 0)
        {
            merged_cloud_.header = cloud.header;  // 保持相同的头信息
        }
        merged_cloud_ += cloud;

        // 增加帧计数
        frame_count_++;

        // 当收到 100 帧点云时，保存并停止订阅
        if (frame_count_ >= 5)
        {
            // 获取功能包路径并确保 data 文件夹存在
            std::filesystem::path data_dir = std::filesystem::path(__FILE__).parent_path().parent_path() / "data";

            // 如果 data 文件夹不存在，创建它
            if (!std::filesystem::exists(data_dir))
            {
                std::filesystem::create_directory(data_dir);
            }

            // 保存合并的点云为 PCD 文件
            string filename = data_dir / "lidar_high.pcd";
            pcl::io::savePCDFileASCII(filename, merged_cloud_);
            RCLCPP_INFO(this->get_logger(), "Saved %d frames to %s", frame_count_, filename.c_str());

            // 停止订阅
            rclcpp::shutdown();
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    pcl::PointCloud<pcl::PointXYZ> merged_cloud_;  // 用于存储合并后的点云
    int frame_count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudSubscriber>());
    rclcpp::shutdown();
    return 0;
}
