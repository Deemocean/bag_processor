# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROS2 MCAP bag processing toolkit for robotics data analysis, focusing on GPS/GNSS trajectory processing and visualization. The codebase converts ROS2 bag files to CSV format and provides utilities for data loading, synchronization, and visualization.

## Core Architecture

### Data Processing Pipeline
1. **BagLoader** (`data_processor/bag_converter/bag_converter.py`): Core MCAP bag reader that converts ROS2 messages to CSV files
2. **DataLoader** (`data_processor/data_loader/data_loader.py`): High-level interface that manages CSV generation and returns a Database object with pandas DataFrames
3. **Topic Handlers** (`data_processor/bag_converter/topic_handler.py`): Message-specific parsers registered via decorator pattern

### Key Components
- **Database class**: Container for multiple topic DataFrames with dictionary-like access
- **TrajectoryConverter**: Transforms GLIM trajectory outputs with isometric transformations
- **Helpers module**: Time synchronization utilities using pandas merge_asof for performance

### Message Type Support
The system handles these ROS2 message types via registered handlers:
- `sensor_msgs/msg/NavSatFix` → GPS coordinates with header timestamp
- `nav_msgs/msg/Odometry` → Position data
- `gps_msgs/msg/GPSFix` → GPS with GDOP quality metrics
- `novatel_oem7_msgs/msg/BESTGNSSPOS` → High-precision Novatel GPS

## Development Workflow

### Running Examples
```bash
# Convert bag to CSV files
python examples/convert_bag.py

# Load data via DataLoader (auto-converts if needed)
python examples/load_data.py

# Comprehensive visualization pipeline
python visualization/plot_all.py
```

### Adding New Message Types
1. Add handler function in `topic_handler.py`
2. Use `@register_handler("message/type/name")` decorator
3. Return dictionary with desired field mappings
4. Handler receives deserialized ROS message object

### Project Structure
- `data_processor/bag_converter/` - Core bag reading and CSV conversion
- `data_processor/data_loader/` - High-level data loading interface  
- `data_processor/helpers/` - Time synchronization and filtering utilities
- `data_processor/traj_converter/` - GLIM trajectory processing
- `visualization/` - Plotting scripts for GPS, odometry, and combined views
- `examples/` - Usage demonstrations

### Dependencies
- ROS2 packages: `rosbag2_py`, `rosidl_runtime_py`, `rclpy`
- Data processing: `pandas`, `numpy`, `scipy`
- Visualization: `matplotlib`, `pymap3d`, `open3d`
- Progress tracking: `tqdm`

## Time Synchronization

The helpers module provides efficient time synchronization between DataFrames:
- `sync_time()`: Merge two DataFrames by timestamp using pandas merge_asof
- `sync_times()`: Sequential synchronization of multiple DataFrames
- `select_time_range()`: Filter DataFrames by time bounds
- Supports "closest" matching and interpolation methods

## Data Flow

1. MCAP bag → BagLoader → CSV files + manifest.json
2. CSV files → DataLoader → Database object with topic DataFrames  
3. Database → helpers sync functions → time-aligned analysis
4. Synchronized data → visualization scripts → plots and analysis