import numpy as np

# registry map: msg_type_str -> handler_fn
handlers = {}
def register_handler(msg_type: str):
    """Decorator to register a handler for a given ROS msg type."""
    def decorator(fn):
        handlers[msg_type] = fn
        return fn
    return decorator

@register_handler("sensor_msgs/msg/NavSatFix")
def navsatfix_handler(msg) -> dict:
    header_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return {"header_t": header_time,
            "lat": msg.latitude,
            "lon": msg.longitude,
            "alt": msg.altitude,
            "covariance": np.array(msg.position_covariance).reshape(3, 3).tolist(),
            "status": msg.status.status
            }

@register_handler("nav_msgs/msg/Odometry")
def odom_handler(msg) -> dict:
    header_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return {"header_t": header_time,
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "z": msg.pose.pose.position.z,
            "vx": msg.twist.twist.linear.x,
            "vy": msg.twist.twist.linear.y,
            "vz": msg.twist.twist.linear.z,
            "qx": msg.pose.pose.orientation.x,
            "qy": msg.pose.pose.orientation.y,
            "qz": msg.pose.pose.orientation.z,
            "qw": msg.pose.pose.orientation.w,
            "pose_covariance": np.array(msg.pose.covariance).reshape(6, 6).tolist()
            }

@register_handler("gps_msgs/msg/GPSFix")
def gpsfix_handler(msg) -> dict:
    header_time  = msg.header.stamp.sec+ msg.header.stamp.nanosec * 1e-9
    return {"header_t": header_time,
            "gps_t": msg.time,
            "lat": msg.latitude,
            "lon": msg.longitude,
            "alt": msg.altitude,
            "gdop": msg.gdop}

@register_handler("novatel_oem7_msgs/msg/BESTGNSSPOS")
def gpsfix_handler(msg) -> dict:
    header_time  = msg.header.stamp.sec+ msg.header.stamp.nanosec * 1e-9
    #TODO: Add GPS TIME(UNIX CONVERT)
    return {"header_t": header_time,
            "lat": msg.lat,
            "lon": msg.lon,
            "alt": msg.hgt}

@register_handler("teleop_msgs/msg/Float32Stamped")
def float32_stamped_handler(msg) -> dict:
    header_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return {"header_t": header_time,
            "data": msg.data}
