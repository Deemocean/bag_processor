# Bag Processor
Helper module for ROS2 MCAP bags processing

## Data Processing
`bag_converter`: interface with ROS2 mcap bags and convert to csv files, modify `bag_converter/topic_handler.py` to add new topics. see `examples/convert_bag.py`.

`data_loader`: convert bag to csv files via `bag_converter` (if not exist), returns a simple `DataBase` object(of pandas DataFrames). see `examples/load_data.py`.

