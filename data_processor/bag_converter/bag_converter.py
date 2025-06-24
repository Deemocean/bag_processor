import sys
import os
import csv
import json     
from typing import List, Dict, Callable, Optional, Union
from dataclasses import dataclass
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions,StorageFilter


from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from tqdm import tqdm

from .topic_handler import handlers as topic_handlers

@dataclass
class BagConfig:
    """Configuration for a single bag with topic mapping."""
    bag_path: str
    topics: Dict[str, str]  # topic_name -> nickname
    nickname: Optional[str] = None  # bag-level nickname  


class BagLoader:
    def __init__(
        self,
        bag_configs: Union[BagConfig, List[BagConfig]],
        output_dir: str
    ):
        # Handle backward compatibility - single bag_path + topics
        if isinstance(bag_configs, str):
            raise ValueError("Please use BagConfig or list of BagConfig objects")
        
        if isinstance(bag_configs, BagConfig):
            self.bag_configs = [bag_configs]
        else:
            self.bag_configs = bag_configs
            
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.handlers: Dict[str, Callable] = topic_handlers
        
        # Pre-process all bags to get topic mappings and validate
        self._validate_and_prepare_bags()
    
    @classmethod
    def create_single_bag(
        cls,
        bag_path: str,
        topics: List[str],
        output_dir: str,
        nickname: Optional[str] = None
    ):
        """Backward compatibility constructor for single bag."""
        topic_map = {topic: topic for topic in topics}
        config = BagConfig(bag_path=bag_path, topics=topic_map, nickname=nickname)
        return cls(config, output_dir)
    
    def _validate_and_prepare_bags(self):
        """Validate all bag configurations and prepare topic mappings."""
        self.all_nicknames = set()
        self.bag_readers = []
        
        for config in self.bag_configs:
            # Validate bag exists
            if not os.path.exists(config.bag_path):
                raise ValueError(f"Bag file not found: {config.bag_path}")
            
            # Initialize reader for this bag
            reader_info = self._init_reader_for_bag(config)
            self.bag_readers.append(reader_info)
            
            # Collect all nicknames to check for conflicts
            for nickname in config.topics.values():
                final_nickname = self._get_final_nickname(nickname, config.nickname)
                if final_nickname in self.all_nicknames:
                    raise ValueError(f"Duplicate nickname: {final_nickname}")
                self.all_nicknames.add(final_nickname)
    
    def _get_final_nickname(self, topic_nickname: str, bag_nickname: Optional[str]) -> str:
        """Generate final topic nickname with optional bag suffix."""
        if bag_nickname:
            return f"{topic_nickname}_{bag_nickname}"
        return topic_nickname
    
    def _init_reader_for_bag(self, config: BagConfig) -> Dict:
        """Initialize reader for a single bag configuration."""
        opts = StorageOptions(uri=config.bag_path, storage_id="mcap")
        conv = ConverterOptions("", "")
        reader = SequentialReader()
        reader.open(opts, conv)
        
        # Get all topics and types for validation
        types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in types}
        
        # Check if requested topics exist in this bag
        topics_to_read = list(config.topics.keys())
        missing = set(topics_to_read) - set(type_map)
        if missing:
            raise ValueError(f"Topics not in bag {config.bag_path}: {missing}")
        
        # Apply topic filter for efficiency
        if topics_to_read:
            filt = StorageFilter(topics=topics_to_read)
            reader.set_filter(filt)
        
        # Create message factories only for needed topics
        msg_factories = {
            topic: get_message(type_map[topic])
            for topic in topics_to_read
        }
        
        return {
            'config': config,
            'reader': reader,
            'type_map': type_map,
            'msg_factories': msg_factories,
            'topics_to_read': topics_to_read
        }


    def _default_handler(self, msg) -> dict:
        row = {}
        for slot in msg.__slots__:
            val = getattr(msg, slot)
            row[slot] = list(val) if hasattr(val, "__len__") and not isinstance(val, str) else val
        return row

    def run_all(self) -> Dict[str, str]:
        """Process all bags and return nickname -> CSV filename mapping."""
        all_buffers: Dict[str, List[dict]] = {}  # final_nickname -> messages
        output_map: Dict[str, str] = {}  # final_nickname -> CSV filename
        
        # Process each bag
        print(f"\nProcessing {len(self.bag_readers)} bag(s)...")
        for bag_idx, reader_info in enumerate(self.bag_readers, 1):
            config = reader_info['config']
            reader = reader_info['reader']
            type_map = reader_info['type_map']
            msg_factories = reader_info['msg_factories']
            topics_to_read = reader_info['topics_to_read']
            
            # Initialize buffers for this bag's topics
            bag_buffers = {}
            for topic_name in topics_to_read:
                nickname = config.topics[topic_name]
                final_nickname = self._get_final_nickname(nickname, config.nickname)
                bag_buffers[topic_name] = []
                all_buffers[final_nickname] = []
            
            print(f"\n[{bag_idx}/{len(self.bag_readers)}] Reading MCAP file:\n  {config.bag_path}\nTopics:")
            for topic_name in topics_to_read:
                nickname = config.topics[topic_name]
                final_nickname = self._get_final_nickname(nickname, config.nickname)
                print(f"  • {topic_name} → {final_nickname}")
            print()
            
            # Read messages from this bag with enhanced progress bar
            bag_name = os.path.basename(config.bag_path)
            desc = f"[{bag_idx}/{len(self.bag_readers)}] {bag_name}"
            with tqdm(desc=desc, unit=" msg", ncols=100) as pbar:
                while reader.has_next():
                    name, data, _ = reader.read_next()
                    if name not in bag_buffers:
                        pbar.update(1)
                        continue
                    
                    msg_type = type_map[name]
                    handler = self.handlers.get(msg_type, self._default_handler)
                    
                    msg = deserialize_message(data, msg_factories[name])
                    bag_buffers[name].append(handler(msg))
                    pbar.update(1)
            
            # Move processed messages to final buffers with nicknames
            bag_total_msgs = 0
            for topic_name, messages in bag_buffers.items():
                nickname = config.topics[topic_name]
                final_nickname = self._get_final_nickname(nickname, config.nickname)
                all_buffers[final_nickname].extend(messages)
                bag_total_msgs += len(messages)
            
            print(f"  → Processed {bag_total_msgs:,} messages from {len(bag_buffers)} topics")
        
        # Write CSV files using final nicknames
        for final_nickname, rows in all_buffers.items():
            if not rows:
                print(f"No messages for {final_nickname}")
                continue
                
            outfile = os.path.join(
                self.output_dir,
                f"{final_nickname.replace('/', '_')}.csv"
            )
            headers = list(rows[0].keys())
            with open(outfile, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            print(f"→ Wrote {len(rows)} rows to {outfile}")
            output_map[final_nickname] = os.path.basename(outfile)
        
        # Update manifest with nicknames
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as mf:
                existing = json.load(mf)
        else:
            existing = {}
        merged = {**existing, **output_map}
        with open(manifest_path, "w") as mf:
            json.dump(merged, mf, indent=2)
        
        print(f"→ Wrote manifest to {manifest_path}")
        return output_map
