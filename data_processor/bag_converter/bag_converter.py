import sys
import os
import csv
import json     
from typing import List, Dict, Callable
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions,StorageFilter


from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from tqdm import tqdm

from .topic_handler import handlers as topic_handlers  


class BagLoader:
    def __init__(
        self,
        bag_path: str,
        topics: List[str],
        output_dir: str
    ):
        self.bag_path = bag_path
        self.topics = topics
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.handlers: Dict[str, Callable] = topic_handlers

        self.reader = self._init_reader()
        types = self.reader.get_all_topics_and_types()
        self.type_map = {t.name: t.type for t in types}
        missing = set(self.topics) - set(self.type_map)
        if missing:
            raise ValueError(f"Topics not in bag: {missing}")
        self.msg_factories = {
            topic: get_message(self.type_map[topic])
            for topic in self.topics
        }

    def _init_reader(self):
        opts = StorageOptions(uri=self.bag_path, storage_id="mcap")
        conv = ConverterOptions("", "")
        rdr = SequentialReader()
        rdr.open(opts, conv)

        if self.topics:
            filt = StorageFilter(topics=self.topics)
            rdr.set_filter(filt)                 
        return rdr

    def _default_handler(self, msg) -> dict:
        row = {}
        for slot in msg.__slots__:
            val = getattr(msg, slot)
            row[slot] = list(val) if hasattr(val, "__len__") and not isinstance(val, str) else val
        return row

    def run_all(self) -> Dict[str, str]:
        buffers: Dict[str, List[dict]] = {t: [] for t in self.topics}
        output_map: Dict[str, str] = {}        

        # pretty‐print the bag path and topic list
        print(f"\nReading MCAP file:\n  {self.bag_path}\nTopics:")
        for t in self.topics:
            print(f"  • {t}")
        print()  # blank line
        
        with tqdm(desc="Reading bag", unit=" msg") as pbar:
            while self.reader.has_next():
                name, data, _ = self.reader.read_next()
                if name not in buffers:
                    pbar.update(1); continue

                msg_type = self.type_map[name]
                handler = self.handlers.get(msg_type, self._default_handler) 

                msg = deserialize_message(data, self.msg_factories[name])
                buffers[name].append(handler(msg))  
                pbar.update(1)

        for topic, rows in buffers.items():
            if not rows:
                print(f"No messages on {topic}")
                continue
            outfile = os.path.join(
                self.output_dir,
                f"{os.path.splitext(os.path.basename(self.bag_path))[0]}_{topic.replace('/','_')}.csv"
            )
            headers = list(rows[0].keys())
            with open(outfile, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            print(f"→ Wrote {len(rows)} rows to {outfile}")
            output_map[topic] = os.path.basename(outfile)       

        # write manifest.json and return mapping
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
