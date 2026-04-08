import copy
import json
import math
import multiprocessing
import os
import traceback
from collections import defaultdict

from layout_system.bilevel.example_library import ExampleLibrary
from layout_system.evaluation import LayoutQualityEvaluator
from layout_system.hierarchical_optimizer import (
    HierarchicalOptimizer,
    OptimizationConfig,
)
from layout_system.utils.save_result import save_hierarchical_result
from layout_system.utils.tree_manipulation import (
    attach_final_bboxes_to_tree,
    get_node_by_path,
    get_structure_signature,
    join_node_path,
    replace_subtree,
    split_node_path,
)


# Global worker function must be defined at module level for pickle
def _process_variation(args):
    """
    Worker function for processing a single layout variation in a separate process.
    """
    (
        iteration,
        parent_idx,
        match_idx,
        new_tree,
        base_config_dict,
        debug,
        root_output_dir,
        bg_color,
    ) = args

    try:
        # Reconstruct config
        config = OptimizationConfig(**base_config_dict)

        # Setup sub output dir
        sub_output_dir = None
        if root_output_dir:
            sub_dir_name = f"iter{iteration}_p{parent_idx}_m{match_idx}"
            sub_output_dir = os.path.join(root_output_dir, sub_dir_name)
            os.makedirs(sub_output_dir, exist_ok=True)
            config.output_dir = sub_output_dir

        # Initialize fresh optimizer for this process
        optimizer = HierarchicalOptimizer(config)

        # Save debug tree
        if debug and sub_output_dir:
            tree_save_path = os.path.join(sub_output_dir, "scene_tree.json")
            with open(tree_save_path, "w") as f:
                json.dump(new_tree, f, indent=2)

        # Optimization
        result = optimizer.optimize_tree(new_tree)

        # Save visualization
        if debug and sub_output_dir:
            # Compute absolute bboxes and save annotated tree
            tree_with_bbox = copy.deepcopy(new_tree)
            attach_final_bboxes_to_tree(result, tree_with_bbox)
            tree_bbox_save_path = os.path.join(sub_output_dir, "scene_tree_with_bbox.json")
            with open(tree_bbox_save_path, "w") as f:
                json.dump(tree_with_bbox, f, indent=2)

            save_hierarchical_result(
                result,
                save_path=os.path.join(sub_output_dir, "hierarchical_result.png"),
                base_dir=config.base_dir or ".",
                bg_color=bg_color,
                debug=config.debug_save,
            )

        return {"result": result, "tree": new_tree, "success": True, "error": None}

    except Exception as e:
        return {"success": False, "error": f"{str(e)}\n{traceback.format_exc()}"}


class BilevelOptimizer:
    def __init__(
        self, config=None, library_dir="extracted_layout_results_new", debug=False
    ):
        self.config = config or OptimizationConfig()
        self.hierarchical_optimizer = HierarchicalOptimizer(self.config)

        # Resolve library dir relative to project root or absolute
        if not os.path.isabs(library_dir) and self.config.base_dir:
            self.library_dir = os.path.join(self.config.base_dir, library_dir)
        else:
            self.library_dir = library_dir

        self.example_library = ExampleLibrary(self.library_dir)
        self.evaluator = LayoutQualityEvaluator(base_dir=self.config.base_dir or ".")
        self.debug = debug

    def optimize(
        self, scene_tree_json, num_iterations=3, top_n=10, top_k=5, debug=True, bg_color=None
    ):
        import time as _time
        self.debug = debug

        # Ensure output directory exists for debug
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)

        print("[BilevelOptimizer] Starting initial optimization...")
        self._global_mins = None
        self._global_maxs = None
        self._all_raw_dicts = []
        t0 = _time.time()
        initial_tree = copy.deepcopy(scene_tree_json)
        initial_result = self.hierarchical_optimizer.optimize_tree(initial_tree)
        print(f'[Bilevel TIMER] initial hierarchical optimize: {_time.time()-t0:.2f}s')

        raw_dict = self._get_raw_losses(initial_tree, initial_result)
        self._update_global_minmax(raw_dict)
        candidates = [
            {"tree": initial_tree, "result": initial_result, "loss": 0.0, "loss_info": dict(raw_dict), "_raw": raw_dict, "variation_id": "initial"}
        ]
        self._compute_pool_totals(candidates)

        if self.debug and self.config.output_dir:
            initial_subdir = os.path.join(self.config.output_dir, "initial")
            os.makedirs(initial_subdir, exist_ok=True)

            # Compute absolute bboxes and save annotated tree
            initial_tree_with_bbox = copy.deepcopy(initial_tree)
            attach_final_bboxes_to_tree(initial_result, initial_tree_with_bbox)
            tree_bbox_save_path = os.path.join(initial_subdir, "scene_tree_with_bbox.json")
            with open(tree_bbox_save_path, "w") as f:
                json.dump(initial_tree_with_bbox, f, indent=2)

            save_hierarchical_result(
                initial_result,
                save_path=os.path.join(initial_subdir, "hierarchical_result.png"),
                base_dir=self.config.base_dir or ".",
                bg_color=bg_color,
                debug=getattr(self.config, "debug_save", False),
            )

        # Iteration loop
        for iteration in range(num_iterations):
            t_iter_start = _time.time()
            print(f"\n[BilevelOptimizer] === Iteration {iteration + 1}/{num_iterations} ===")
            print(f"Current pool size: {len(candidates)}, best loss: {min(c['loss'] for c in candidates):.4f}")

            next_generation = []

            # Process each candidate from previous generation (or top_k of them)
            # Sort candidates by loss
            candidates.sort(key=lambda x: x["loss"])
            current_generation_parents = candidates[:top_k]

            # Carry over the parents to next generation
            next_generation.extend(current_generation_parents)

            # Prepare tasks
            tasks = []

            for parent_idx, parent_cand in enumerate(current_generation_parents):
                parent_tree = parent_cand["tree"]
                parent_result = parent_cand["result"]

                # 2. Identify worst element
                # Traverse result and tree to find max mismatch
                # We need to link result nodes back to tree nodes to modify the tree?
                # The result structure matches tree structure.

                worst_node_path, max_loss = self._find_worst_element(parent_tree, parent_result)
                if self.debug:
                    worst_node = get_node_by_path(parent_tree, worst_node_path)
                    print(f"  [Parent {parent_idx}] Worst element {get_structure_signature(worst_node)} at {worst_node_path}, loss={max_loss:.4f}")

                if max_loss < 1e-4:
                    if self.debug:
                        print("  Loss is negligible, skipping.")
                    continue

                t_match_start = _time.time()
                match_pool = []
                path_parts = split_node_path(worst_node_path)

                for i in range(len(path_parts), -1, -1):
                    t_target_path = join_node_path(path_parts[:i])
                    v_target_path = join_node_path(path_parts[i:]) 

                    t_target = get_node_by_path(parent_tree, t_target_path)
                    if not t_target:
                        continue

                    # Currently we only consider editing existing nodes
                    # TODO: adding new node to scene graph (v_target_path=None)
                    # TODO: removing node from scene graph (v_source_path=None)
                    matches = self.example_library.find_matches(t_target, v_target_path)
                    for match in matches:
                        match_pool.append(
                            {
                                "t_target_path": t_target_path,
                                "v_target_path": v_target_path,
                                "entry": match["entry"],
                                "cond_prob": match["cond_prob"],
                            }
                        )

                if not match_pool:
                    continue

                if self.config.output_dir and True:  # TODO: add debug condition
                    pool_copy = copy.deepcopy(match_pool)
                    for m in pool_copy:
                        del m["entry"]["data"]  # Remove heavy data for debug save

                    debug_path = os.path.join(self.config.output_dir, f"iter{iteration}_p{parent_idx}_match_pool.json")
                    with open(debug_path, "w") as f:
                        json.dump(pool_copy, f, indent=2)

                selected = self._select_matches_uct(match_pool, top_n=top_n)
                print(f'[Bilevel TIMER]   find_matches + select: {_time.time()-t_match_start:.2f}s, pool={len(match_pool)}, selected={len(selected)}')

                for match_idx, match in enumerate(selected):
                    target_to_replace = get_node_by_path(parent_tree, match["t_target_path"])
                    if not target_to_replace:
                        continue

                    new_tree = copy.deepcopy(parent_tree)
                    target_to_replace = get_node_by_path(new_tree, match["t_target_path"])
                    replaced_subtree = replace_subtree(
                        match["entry"]["data"],
                        target_to_replace,
                        v_source_path=match["entry"]["v_source_path"],
                        v_target_path=match["v_target_path"],
                    )

                    target_path_parts = split_node_path(match["t_target_path"])
                    if not target_path_parts:
                        new_tree = replaced_subtree
                    else:
                        parent_path = join_node_path(target_path_parts[:-1])
                        parent_node = get_node_by_path(new_tree, parent_path)
                        if not parent_node:
                            continue
                        index = target_path_parts[-1]
                        parent_node["children"][index] = replaced_subtree

                    config_dict = self._get_pickleable_config()
                    args = (
                        iteration,
                        parent_idx,
                        match_idx,
                        new_tree,
                        config_dict,
                        self.debug,
                        self.config.output_dir,
                        bg_color,
                    )
                    tasks.append(args)

            # Parallel Process
            if tasks:
                print(f"  Running {len(tasks)} optimization tasks in parallel...")
                num_processes = min(len(tasks), os.cpu_count() or 1, 15)

                t_pool_start = _time.time()
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(processes=num_processes) as pool:
                    results = pool.map(_process_variation, tasks)
                print(f'[Bilevel TIMER]   parallel pool.map ({len(tasks)} tasks, {num_processes} workers): {_time.time()-t_pool_start:.2f}s')

                t_loss_start = _time.time()
                for i, res in enumerate(results):
                    if res["success"]:
                        new_tree = res["tree"]
                        new_result = res["result"]
                        it, pi, mi = tasks[i][0], tasks[i][1], tasks[i][2]
                        variation_id = f"iter{it}_p{pi}_m{mi}"

                        raw_dict = self._get_raw_losses(new_tree, new_result)
                        self._update_global_minmax(raw_dict)
                        next_generation.append(
                            {"tree": new_tree, "result": new_result, "loss": 0.0, "loss_info": dict(raw_dict), "_raw": raw_dict, "variation_id": variation_id}
                        )
                    else:
                        print(f"  Optimization failed for variation: {res['error']}")
                print(f'[Bilevel TIMER]   compute losses: {_time.time()-t_loss_start:.2f}s')
            else:
                print("  No tasks generated for this iteration.")

            self._compute_pool_totals(next_generation)
            next_generation.sort(key=lambda x: x["loss"])
            candidates = next_generation

            print(f'[Bilevel TIMER] iteration {iteration+1} total: {_time.time()-t_iter_start:.2f}s, {len(next_generation)} candidates')

        for c in candidates:
            c["_global_mins"] = dict(self._global_mins) if self._global_mins else {}
            c["_global_maxs"] = dict(self._global_maxs) if self._global_maxs else {}
        return candidates

    def _select_matches_uct(self, match_pool, top_n=10, uct_c=0.6):
        selected = []
        selected_counts = defaultdict(int)
        used_indices = set()

        def weight_for(entry, total_selected):
            base = entry["cond_prob"]
            if total_selected == 0:
                return base
            full_sig = entry["entry"]["full_struct_sig"]
            bonus = uct_c * math.sqrt(
                math.log(total_selected + 1) / (1 + selected_counts[full_sig])
            )
            return base + bonus

        while len(selected) < min(top_n, len(match_pool)):
            best_index = None
            best_weight = None
            total_selected = len(selected)

            for idx, cand in enumerate(match_pool):
                if idx in used_indices:
                    continue
                score = weight_for(cand, total_selected)
                if best_weight is None or score > best_weight:
                    best_weight = score
                    best_index = idx

            if best_index is None:
                break

            chosen = match_pool[best_index]
            used_indices.add(best_index)
            selected.append(chosen)
            selected_counts[chosen["entry"]["full_struct_sig"]] += 1

        return selected

    def _get_pickleable_config(self):
        # Convert config object to dictionary, filtering out unpickleable objects
        # We only need the basic params to recreate OptimizationConfig in worker
        d = {}
        # OptimizationConfig is a dataclass, so it has __dict__ or asdict
        # Check attributes
        config_attrs = {
            k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
        }

        for k, v in config_attrs.items():
            # Exclude complex objects
            if k in ["constraint_processors", "node_handlers", "strategy"]:
                # These will be re-initialized by default in worker's fresh HierarchicalOptimizer
                continue
            d[k] = v
        return d

    def _get_raw_losses(self, tree, result):
        return self.evaluator.compute_raw_losses(tree, result)

    def _update_global_minmax(self, raw_dict):
        self._all_raw_dicts.append(dict(raw_dict))
        self._global_mins, self._global_maxs = self.evaluator.compute_minmax_bounds(
            self._all_raw_dicts
        )

    def _compute_pool_totals(self, candidates):
        if not candidates:
            return

        if self._global_mins is None or self._global_maxs is None:
            self._global_mins, self._global_maxs = self.evaluator.compute_minmax_bounds(
                [c.get("_raw", {}) for c in candidates]
            )

        for c in candidates:
            raw = c.get("_raw", {})
            scored = self.evaluator.score_raw(raw, self._global_mins, self._global_maxs)

            c["loss"] = float(scored["total"])
            if "loss_info" in c:
                c["loss_info"]["total"] = float(scored["total"])
                c["loss_info"]["_normalized"] = dict(scored["normalized"])

    def _find_worst_element(self, tree, result):
        return self.evaluator.find_worst_element(tree, result)

    def _save_debug_json(self, data, name):
        if self.config.output_dir:
            path = os.path.join(self.config.output_dir, name)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
