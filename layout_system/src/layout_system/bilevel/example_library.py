import glob
import json
import os
from collections import defaultdict

from layout_system.utils.tree_manipulation import (
    collect_leaf_paths,
    get_node_by_path,
    get_structure_signature,
)


class ExampleLibrary:
    def __init__(self, library_dir, debug: bool = False):
        self.library_dir = library_dir
        self.debug = debug
        self.examples = []
        # Map: reduced_struct_sig -> list of examples
        self.structure_map = defaultdict(list)
        # Map: reduced_struct_sig -> {full_struct_sig: count}
        self.structure_counts = defaultdict(lambda: defaultdict(int))

        self._load_library()

    def _load_library(self):
        json_files = glob.glob(os.path.join(self.library_dir, "*_layout.json"))
        if self.debug:
            print(
                f"[ExampleLibrary] Loading {len(json_files)} files from {self.library_dir}"
            )

        for fpath in json_files:
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                    # Support direct root, scene_tree wrapper, or scene_graph wrapper
                    if "scene_graph" in data:
                        root = data["scene_graph"]
                    elif "scene_tree" in data:
                        root = data["scene_tree"]
                    else:
                        root = data

                    self._extract_subtrees(root, fpath)
            except Exception as e:
                if self.debug:
                    print(f"[ExampleLibrary] Error loading {fpath}: {e}")

    def _extract_subtrees(self, node_data, source_file):
        # Register this node as example
        full_struct_sig = get_structure_signature(node_data)
        self._add_entry(node_data, source_file, None, full_struct_sig, full_struct_sig)
        
        children = node_data.get("children", [])
        if not children:
            return

        # Enumerate leaf nodes and create reduced signatures by excluding each leaf
        for leaf_path, leaf_node in collect_leaf_paths(node_data):
            reduced_sig = get_structure_signature(node_data, exclude_node=leaf_node)
            self._add_entry(
                node_data, source_file, leaf_path, full_struct_sig, reduced_sig
            )

        # Recursively extract
        for child in children:
            self._extract_subtrees(child, source_file)


    def _add_entry(
        self, node_data, source_file, v_source_path, full_struct_sig, reduced_sig
    ):
        entry = {
            "data": node_data,
            "source": source_file,
            "v_source_path": v_source_path,
            "full_struct_sig": full_struct_sig,
            "struct_sig": reduced_sig,
        }

        self.examples.append(entry)
        self.structure_map[reduced_sig].append(entry)
        self.structure_counts[reduced_sig][full_struct_sig] += 1

    def find_matches(self, t_target, v_target_path=None):
        if v_target_path == "":
            return []  # XXX: ignore root node matches for now

        v_target = get_node_by_path(t_target, v_target_path)
        reduced_sig = get_structure_signature(t_target, exclude_node=v_target)
        candidates = self.structure_map.get(reduced_sig, [])

        if not candidates:
            return []

        full_sig_counts = self.structure_counts[reduced_sig]
        total_count = sum(full_sig_counts.values()) or 1
        results = []
        for entry in candidates:
            v_source = get_node_by_path(entry["data"], entry["v_source_path"])
            if v_source is None:
                continue  # XXX: ignore full tree matches for now
            if v_target is not None and v_source.get("type") != v_target.get("type"):
                continue  # type mismatch, skip

            full_sig = entry["full_struct_sig"]
            cond_prob = full_sig_counts.get(full_sig, 0) / total_count
            results.append({"entry": entry, "cond_prob": cond_prob})

        return results
