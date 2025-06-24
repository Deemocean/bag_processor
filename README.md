# Bag Processor

A Python toolkit for processing ROS2 MCAP bag files with multi-bag support and topic nicknames.

## Features

- **Multi-bag processing** with flexible topic nickname system
- **Extensible message type support** through decorator-based handlers
- **Efficient CSV conversion** with progress tracking
- **Time synchronization** utilities for data alignment
- **Pandas DataFrame interface** for data analysis

## Architecture

- **`data_processor/bag_converter/`** - Direct MCAP to CSV conversion
- **`data_processor/data_loader/`** - High-level data loading interface
- **`data_processor/helpers/`** - Time synchronization utilities
- **`examples/`** - Usage examples

## Supported Message Types

- **`sensor_msgs/msg/NavSatFix`** - Standard GPS navigation satellite fix
- **`nav_msgs/msg/Odometry`** - Robot odometry data  
- **`gps_msgs/msg/GPSFix`** - GPS fix with quality metrics
- **`novatel_oem7_msgs/msg/BESTGNSSPOS`** - NovAtel GPS receiver data

## Quick Start

### Single Bag Processing

```python
from data_processor.bag_converter.bag_converter import BagConfig, BagLoader

config = BagConfig(
    bag_path="/path/to/bag.mcap",
    topics={"/gps/fix": "gps", "/odom": "odometry"}
)
loader = BagLoader(config, "/tmp/CSVs")
loader.run_all()
```

### Multi-Bag Processing with Nicknames

```python
bag_configs = [
    BagConfig(
        bag_path="/path/to/run1.mcap",
        topics={"/gps/fix": "gps_a", "/odom": "odom"},
        nickname="run1"  # Creates: gps_a_run1.csv, odom_run1.csv
    ),
    BagConfig(
        bag_path="/path/to/run2.mcap", 
        topics={"/gps/fix": "gps_a", "/odom": "odom"},
        nickname="run2"  # Creates: gps_a_run2.csv, odom_run2.csv
    )
]

loader = BagLoader(bag_configs, "/tmp/CSVs")
loader.run_all()
```

### Data Loading

```python
from data_processor.data_loader.data_loader import DataLoader

loader = DataLoader(bag_configs=bag_configs, output_dir="/tmp/CSVs")
db = loader.load_all()

# Access data as pandas DataFrames
gps_data = db.get_topic_data("gps_a_run1")
odom_data = db.get_topic_data("odom_run1")
```

## Adding Custom Message Types

```python
# In data_processor/bag_converter/topic_handler.py
@register_handler("custom_msgs/msg/MyMessage")
def my_message_handler(msg) -> dict:
    header_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return {
        "header_t": header_time,
        "custom_field": msg.custom_field
    }
```

## Dependencies

```bash
pip install rosbag2_py rclpy rosidl_runtime_py pandas numpy tqdm
```

Requires ROS2 installation with `ros2` command-line tools.

## Output

Generated files:
```
/tmp/CSVs/
├── gps_a_run1.csv
├── odom_run1.csv  
├── gps_a_run2.csv
├── odom_run2.csv
└── manifest.json
```