# data_loader.py

import os
import json
import pandas as pd
from typing import List, Dict, Union, Optional

from data_processor.bag_converter.bag_converter import BagLoader, BagConfig

class Database:
    def __init__(self, tables: Dict[str, pd.DataFrame]):
        """Holds one DataFrame per topic."""
        self._tables = tables

    def get_topic(self, topic: str) -> pd.DataFrame:
        """Return the DataFrame for a topic (or empty DF if missing)."""
        return self._tables.get(topic, pd.DataFrame())

    def topics(self) -> List[str]:
        """List all topics loaded."""
        return list(self._tables.keys())
    
    def add_topic(self, topic: str, df: pd.DataFrame):
        """Add a new topic DataFrame to the database."""
        self._tables[topic] = df

    def __getitem__(self, topic: str) -> pd.DataFrame:
        return self.get_topic(topic)


class DataLoader:
    def __init__(
        self,
        bag_configs: Union[BagConfig, List[BagConfig], str] = None,
        topics: Optional[List[str]] = None,
        output_dir: str = None,
        # Backward compatibility parameters
        bag_path: Optional[str] = None
    ):
        """
        New API:
            bag_configs : BagConfig or list of BagConfig objects
            output_dir  : where CSVs + manifest live
        
        Backward compatibility:
            bag_path    : path to .mcap (deprecated, use bag_configs)
            topics      : list of topic names (deprecated)
        """
        # Handle backward compatibility
        if bag_path is not None and topics is not None:
            # Old API usage
            topic_map = {topic: topic for topic in topics}
            self.bag_configs = [BagConfig(bag_path=bag_path, topics=topic_map)]
            self.topics = topics  # Keep for manifest compatibility
        elif isinstance(bag_configs, str):
            # Mixed usage - bag_configs is actually bag_path
            if topics is None:
                raise ValueError("When providing bag_path as string, topics must be specified")
            topic_map = {topic: topic for topic in topics}
            self.bag_configs = [BagConfig(bag_path=bag_configs, topics=topic_map)]
            self.topics = topics
        else:
            # New API usage
            if isinstance(bag_configs, BagConfig):
                self.bag_configs = [bag_configs]
            else:
                self.bag_configs = bag_configs or []
            
            # Extract all nicknames for manifest compatibility
            self.topics = []
            for config in self.bag_configs:
                for topic_name, nickname in config.topics.items():
                    final_nickname = self._get_final_nickname(nickname, config.nickname)
                    self.topics.append(final_nickname)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest_f = os.path.join(self.output_dir, "manifest.json")
    
    def _get_final_nickname(self, topic_nickname: str, bag_nickname: Optional[str]) -> str:
        """Generate final topic nickname with optional bag suffix."""
        if bag_nickname:
            return f"{topic_nickname}_{bag_nickname}"
        return topic_nickname
    
    def _filter_unprocessed_configs(self) -> List[BagConfig]:
        """Filter out bag configs that have already been processed."""
        # Load existing manifest if it exists
        existing_topics = set()
        if os.path.exists(self.manifest_f):
            try:
                with open(self.manifest_f) as f:
                    existing_manifest = json.load(f)
                    existing_topics = set(existing_manifest.keys())
            except:
                # If manifest is corrupted, process everything
                return self.bag_configs
        
        # Filter configs
        configs_to_process = []
        for config in self.bag_configs:
            # Check if any topic from this config is missing from manifest
            config_topics = set()
            for topic_name, nickname in config.topics.items():
                final_nickname = self._get_final_nickname(nickname, config.nickname)
                config_topics.add(final_nickname)
            
            # If any topic from this config is missing, we need to process this config
            if not config_topics.issubset(existing_topics):
                configs_to_process.append(config)
                missing = config_topics - existing_topics
                print(f"Need to process config for '{config.nickname or 'unnamed'}': missing topics {missing}")
            else:
                print(f"Config for '{config.nickname or 'unnamed'}' already processed, skipping")
        
        return configs_to_process

    def _generate_manifest(self):
        # Check existing manifest to determine which configs need processing
        configs_to_process = self._filter_unprocessed_configs()
        
        if configs_to_process:
            print(f"Processing {len(configs_to_process)} out of {len(self.bag_configs)} bag configs...")
            BagLoader(configs_to_process, self.output_dir).run_all()
        else:
            print("All bag configs already processed, skipping generation...")

    def _load_manifest(self) -> Dict[str, str]:
        with open(self.manifest_f) as f:
            return json.load(f)

    def load_all(self) -> Database:
        """Ensure CSVs exist for every topic, then read each into a DataFrame."""

        # 1) Generate manifest if it doesn’t exist
        if not os.path.exists(self.manifest_f):
            self._generate_manifest()

        # 2) Load manifest and check for missing topics or files
        manifest = self._load_manifest()
        missing_topics = set(self.topics) - set(manifest.keys())

        # check for missing CSV files
        missing_files = []
        for topic, fname in manifest.items():
            path = os.path.join(self.output_dir, fname)
            if topic in self.topics and not os.path.exists(path):
                missing_files.append(topic)

        if missing_topics or missing_files:
            if missing_topics:
                print("\nMissing topics in manifest:")
                for t in missing_topics:
                    print(f"  • {t}")
            if missing_files:
                print("\nMissing CSV files for topics:")
                for t in missing_files:
                    print(f"  • {t} → {manifest.get(t)}")
            print("Regenerating manifest and CSVs for missing data...")
            # Only process configs that have missing data
            self._generate_manifest()
            manifest = self._load_manifest()

        # 3) Read only the topics we asked for
        tables: Dict[str, pd.DataFrame] = {}
        for topic in self.topics:
            fname = manifest.get(topic)
            if not fname:
                print(f"[ERROR] No CSV for topic {topic} even after regen")
                continue
            path = os.path.join(self.output_dir, fname)
            tables[topic] = pd.read_csv(path)

        return Database(tables)
