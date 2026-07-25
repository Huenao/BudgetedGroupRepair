"""Deterministic overlapping candidate-query generation for BGR."""

from __future__ import annotations

import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .cell_features import (
    CellFeatures,
    SparseVector,
    build_cell_features,
    cosine_distance,
    cosine_similarity,
    merge_vectors,
    pattern_vectors,
    semantic_vectors,
)
from .data import SafeCell
from .group_context import (
    CanonicalMessages,
    GroupContextBuilder,
    PROMPT_SCHEMA_VERSION,
    canonical_messages,
    compute_query_id,
    messages_as_dicts,
)
from .prompt_policy import INFORMATION_POLICY
GROUP_VIEWS = ("singleton", "row", "pattern", "public_fd", "semantic")
DEFAULT_GROUP_SIZES = (1, 2, 4, 8)
EXACT_OPTIMAL_LINKAGE_LIMIT = 512


def _frozen_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in sorted(values.items(), key=lambda item: str(item[0])):
        if isinstance(value, Mapping):
            frozen[str(key)] = _frozen_mapping({str(k): v for k, v in value.items()})
        elif isinstance(value, list):
            frozen[str(key)] = tuple(value)
        else:
            frozen[str(key)] = value
    return MappingProxyType(frozen)


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


@dataclass(frozen=True)
class GroupQueryAction:
    """A fixed, fully costed group query action."""

    query_id: str
    suite: str
    dataset: str
    arm: str
    group_view: str
    cell_ids: tuple[str, ...]
    group_size: int
    prompt_schema_version: str
    prompt_information_policy: str
    messages: CanonicalMessages
    prompt_hash: str
    estimated_prompt_tokens: int
    completion_token_ceiling: int
    estimated_total_tokens: int
    group_features: Mapping[str, Any]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(str(identifier) for identifier in self.cell_ids))
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("cell_ids must be non-empty and unique")
        if int(self.group_size) != len(ordered):
            raise ValueError("group_size must equal len(cell_ids)")
        if self.group_view not in GROUP_VIEWS:
            raise ValueError(f"unsupported group_view: {self.group_view!r}")
        if self.arm not in {"singleton", "structured", "random"}:
            raise ValueError(f"unsupported experiment arm: {self.arm!r}")
        if self.prompt_information_policy != INFORMATION_POLICY:
            raise ValueError("query action uses the wrong prompt information policy")
        if int(self.estimated_prompt_tokens) <= 0:
            raise ValueError("estimated_prompt_tokens must be positive")
        if int(self.completion_token_ceiling) <= 0:
            raise ValueError("completion_token_ceiling must be positive")
        if int(self.estimated_total_tokens) < (
            int(self.estimated_prompt_tokens) + int(self.completion_token_ceiling)
        ):
            raise ValueError("estimated_total_tokens is below its prompt and completion components")
        object.__setattr__(self, "cell_ids", ordered)
        object.__setattr__(self, "group_size", len(ordered))
        object.__setattr__(self, "messages", canonical_messages(self.messages))
        object.__setattr__(self, "group_features", _frozen_mapping(dict(self.group_features)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "suite": self.suite,
            "dataset": self.dataset,
            "arm": self.arm,
            "group_view": self.group_view,
            "cell_ids": list(self.cell_ids),
            "group_size": self.group_size,
            "prompt_schema_version": self.prompt_schema_version,
            "prompt_information_policy": self.prompt_information_policy,
            "messages": messages_as_dicts(self.messages),
            "prompt_hash": self.prompt_hash,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "completion_token_ceiling": self.completion_token_ceiling,
            "estimated_total_tokens": self.estimated_total_tokens,
            "group_features": _plain_value(self.group_features),
        }


@dataclass(frozen=True)
class GroupGenerationResult:
    actions: tuple[GroupQueryAction, ...]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit", _frozen_mapping(dict(self.audit)))

    @property
    def by_cell(self) -> dict[str, tuple[GroupQueryAction, ...]]:
        membership: dict[str, list[GroupQueryAction]] = defaultdict(list)
        for action in self.actions:
            for cell_id in action.cell_ids:
                membership[cell_id].append(action)
        return {
            cell_id: tuple(sorted(actions, key=lambda action: action.query_id))
            for cell_id, actions in membership.items()
        }


def exact_windows(ordered_cell_ids: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    """Return exact-size half-stride windows and an explicit final tail window."""

    ordered = tuple(str(identifier) for identifier in ordered_cell_ids)
    width = int(size)
    if width <= 1:
        raise ValueError("non-singleton window size must be greater than one")
    if len(ordered) < width:
        return ()
    stride = max(1, width // 2)
    starts = list(range(0, len(ordered) - width + 1, stride))
    tail_start = len(ordered) - width
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)
    windows: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for start in starts:
        window = tuple(ordered[start : start + width])
        if window not in seen:
            seen.add(window)
            windows.append(window)
    return tuple(windows)


def _average_linkage_leaf_order_python(
    ordered_ids: Sequence[str],
    vectors: Mapping[str, Mapping[str, float]],
) -> tuple[str, ...]:
    """Deterministic Lance-Williams fallback when SciPy is unavailable."""

    identifiers = tuple(sorted(str(identifier) for identifier in ordered_ids))
    count = len(identifiers)
    if count <= 1:
        return identifiers
    active = set(range(count))
    sizes = {index: 1 for index in active}
    leaves = {index: (identifiers[index],) for index in active}
    distances: dict[tuple[int, int], float] = {}
    heap: list[tuple[float, str, str, int, int]] = []

    def pair_key(left: int, right: int) -> tuple[int, int]:
        return (left, right) if left < right else (right, left)

    def add_distance(left: int, right: int, distance: float) -> None:
        first, second = pair_key(left, right)
        distances[(first, second)] = float(distance)
        left_key = min(leaves[first])
        right_key = min(leaves[second])
        heapq.heappush(
            heap,
            (float(distance), min(left_key, right_key), max(left_key, right_key), first, second),
        )

    for left in range(count):
        for right in range(left + 1, count):
            add_distance(left, right, cosine_distance(vectors.get(identifiers[left], {}), vectors.get(identifiers[right], {})))

    next_cluster = count
    while len(active) > 1:
        while heap:
            _, _, _, left, right = heapq.heappop(heap)
            if left in active and right in active:
                break
        else:  # pragma: no cover - defensive invariant
            raise RuntimeError("average-linkage heap was exhausted")

        other_clusters = sorted(active.difference({left, right}))
        left_leaves = leaves[left]
        right_leaves = leaves[right]
        boundary_lr = cosine_distance(
            vectors.get(left_leaves[-1], {}), vectors.get(right_leaves[0], {})
        )
        boundary_rl = cosine_distance(
            vectors.get(right_leaves[-1], {}), vectors.get(left_leaves[0], {})
        )
        if boundary_lr < boundary_rl:
            merged_leaves = left_leaves + right_leaves
        elif boundary_rl < boundary_lr:
            merged_leaves = right_leaves + left_leaves
        else:
            merged_leaves = min(left_leaves + right_leaves, right_leaves + left_leaves)

        merged = next_cluster
        next_cluster += 1
        leaves[merged] = merged_leaves
        sizes[merged] = sizes[left] + sizes[right]
        active.remove(left)
        active.remove(right)
        active.add(merged)
        for other in other_clusters:
            left_distance = distances[pair_key(left, other)]
            right_distance = distances[pair_key(right, other)]
            average = (
                sizes[left] * left_distance + sizes[right] * right_distance
            ) / (sizes[left] + sizes[right])
            add_distance(merged, other, average)
    return leaves[next(iter(active))]


def stable_average_linkage_order(
    cell_ids: Sequence[str],
    vectors: Mapping[str, Mapping[str, float]],
) -> tuple[str, ...]:
    """Average-linkage cosine leaf order with stable input/tie ordering."""

    identifiers = tuple(sorted(str(identifier) for identifier in cell_ids))
    if len(identifiers) <= 1:
        return identifiers
    try:
        import numpy as np
        from scipy.cluster.hierarchy import leaves_list, linkage

        if len(identifiers) <= EXACT_OPTIMAL_LINKAGE_LIMIT:
            condensed = np.asarray(
                [
                    cosine_distance(
                        vectors.get(identifiers[left], {}),
                        vectors.get(identifiers[right], {}),
                    )
                    for left in range(len(identifiers))
                    for right in range(left + 1, len(identifiers))
                ],
                dtype=float,
            )
            optimal_ordering = True
        else:
            condensed = _sparse_cosine_condensed(identifiers, vectors)
            # Optimal leaf ordering is not part of average-linkage clustering
            # and has prohibitive super-quadratic cost for the largest CARE
            # columns. Linkage and cosine distances remain exact.
            optimal_ordering = False
        hierarchy = linkage(
            condensed,
            method="average",
            optimal_ordering=optimal_ordering,
        )
        return tuple(identifiers[int(index)] for index in leaves_list(hierarchy))
    except ImportError:
        return _average_linkage_leaf_order_python(identifiers, vectors)


def _sparse_cosine_condensed(
    identifiers: Sequence[str],
    vectors: Mapping[str, Mapping[str, float]],
) -> Any:
    """Vectorize the exact sparse cosine distances for a large linkage bucket."""

    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_distances

    ordered = tuple(str(identifier) for identifier in identifiers)
    feature_names = sorted(
        {
            str(name)
            for identifier in ordered
            for name, value in vectors.get(identifier, {}).items()
            if float(value) != 0.0
        }
    )
    if not feature_names:
        return np.zeros(len(ordered) * (len(ordered) - 1) // 2, dtype=float)
    feature_index = {name: index for index, name in enumerate(feature_names)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    empty_rows: list[int] = []
    for row, identifier in enumerate(ordered):
        nonzero = False
        for name, value in sorted(vectors.get(identifier, {}).items()):
            numeric = float(value)
            if numeric == 0.0:
                continue
            row_indices.append(row)
            column_indices.append(feature_index[str(name)])
            values.append(numeric)
            nonzero = True
        if not nonzero:
            empty_rows.append(row)
    matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(ordered), len(feature_names)),
        dtype=float,
    )
    distances = np.asarray(cosine_distances(matrix), dtype=float)
    if empty_rows:
        empty = np.asarray(empty_rows, dtype=int)
        distances[np.ix_(empty, empty)] = 0.0
    np.fill_diagonal(distances, 0.0)
    np.clip(distances, 0.0, 2.0, out=distances)
    return squareform(distances, checks=False)


def _cohesion(cell_ids: Sequence[str], vectors: Mapping[str, Mapping[str, float]]) -> float:
    similarities = [
        cosine_similarity(vectors.get(left, {}), vectors.get(right, {}))
        for index, left in enumerate(cell_ids)
        for right in cell_ids[index + 1 :]
    ]
    return round(sum(similarities) / len(similarities), 8) if similarities else 1.0


def _component_value(component: Any, name: str, default: Any) -> Any:
    if isinstance(component, Mapping):
        return component.get(name, default)
    return getattr(component, name, default)


class GroupGenerator:
    """Generate singleton plus overlapping row/pattern/public-FD/semantic actions."""

    def __init__(
        self,
        dataset: Any,
        cells: Sequence[SafeCell],
        baran_by_cell: Any = None,
        *,
        fd_components: Sequence[Any] = (),
        group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
        prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
        call_overhead_tokens: int = 0,
        similar_row_count: int = 3,
    ) -> None:
        self.dataset = dataset
        self.cells = tuple(sorted(cells, key=lambda cell: cell.cell_id))
        self.cell_by_id = {cell.cell_id: cell for cell in self.cells}
        if len(self.cell_by_id) != len(self.cells):
            raise ValueError("cells must have unique cell_id values")
        identities = {(str(cell.suite), str(cell.dataset)) for cell in self.cells}
        if len(identities) > 1:
            raise ValueError("all cells must belong to one dataset")
        self.suite, self.dataset_name = next(iter(identities), (str(getattr(dataset, "suite", "")), str(getattr(dataset, "name", ""))))
        requested_sizes = {int(size) for size in group_sizes}
        if any(size not in DEFAULT_GROUP_SIZES for size in requested_sizes):
            raise ValueError("group_sizes must be selected from 1, 2, 4, and 8")
        self.group_sizes = tuple(sorted({1, *requested_sizes}))
        self.non_singleton_sizes = tuple(size for size in self.group_sizes if size > 1)
        self.prompt_schema_version = str(prompt_schema_version)
        self.call_overhead_tokens = max(0, int(call_overhead_tokens))
        self.fd_components = tuple(fd_components)
        self.features = build_cell_features(dataset, self.cells, baran_by_cell)
        self.feature_by_id = {feature.cell_id: feature for feature in self.features}
        self.pattern_by_id = pattern_vectors(self.features)
        self.semantic_by_id = semantic_vectors(self.features)
        self.combined_by_id = {
            cell_id: merge_vectors(
                self.pattern_by_id.get(cell_id, {}),
                self.semantic_by_id.get(cell_id, {}),
                weights=(0.65, 0.35),
            )
            for cell_id in self.cell_by_id
        }
        self.context = GroupContextBuilder(
            dataset,
            self.cells,
            baran_by_cell,
            fd_components=self.fd_components,
            similar_row_count=similar_row_count,
        )

    def _buckets(self, view: str) -> list[tuple[str, tuple[str, ...], Mapping[str, SparseVector]]]:
        buckets: list[tuple[str, tuple[str, ...], Mapping[str, SparseVector]]] = []
        if view == "row":
            by_row: dict[int, list[str]] = defaultdict(list)
            for cell in self.cells:
                by_row[int(cell.row)].append(cell.cell_id)
            buckets.extend(
                (f"row:{row}", tuple(sorted(ids)), self.combined_by_id)
                for row, ids in sorted(by_row.items())
                if len(ids) >= 2
            )
        elif view in {"pattern", "semantic"}:
            by_column: dict[str, list[str]] = defaultdict(list)
            for cell in self.cells:
                by_column[str(cell.column)].append(cell.cell_id)
            vectors = self.pattern_by_id if view == "pattern" else self.semantic_by_id
            buckets.extend(
                (f"column:{column}", tuple(sorted(ids)), vectors)
                for column, ids in sorted(by_column.items())
                if len(ids) >= 2
            )
        elif view == "public_fd":
            for component in sorted(
                self.fd_components,
                key=lambda item: (
                    str(_component_value(item, "component_id", "")),
                    str(_component_value(item, "rule_id", "")),
                ),
            ):
                members = tuple(
                    sorted(
                        set(str(value) for value in _component_value(component, "cell_ids", ())).intersection(self.cell_by_id)
                    )
                )
                if len(members) >= 2:
                    component_id = str(_component_value(component, "component_id", ""))
                    rule_id = str(_component_value(component, "rule_id", ""))
                    buckets.append((f"fd:{rule_id}:{component_id}", members, self.combined_by_id))
        else:
            raise ValueError(f"unsupported non-singleton view: {view!r}")
        return buckets

    def _action(
        self,
        view: str,
        cell_ids: Sequence[str],
        *,
        bucket_id: str,
        vectors: Mapping[str, Mapping[str, float]],
    ) -> GroupQueryAction:
        ordered_ids = tuple(sorted(str(identifier) for identifier in cell_ids))
        cells = tuple(self.cell_by_id[identifier] for identifier in ordered_ids)
        query_id = compute_query_id(
            self.suite,
            self.dataset_name,
            view,
            ordered_ids,
            arm="singleton" if view == "singleton" else "structured",
            prompt_schema_version=self.prompt_schema_version,
            information_policy=INFORMATION_POLICY,
        )
        material = self.context.build_material(
            query_id,
            view,
            cells,
            prompt_schema_version=self.prompt_schema_version,
            call_overhead_tokens=self.call_overhead_tokens,
        )
        feature_rows: list[CellFeatures] = [self.feature_by_id[identifier] for identifier in ordered_ids]
        group_features = {
            "group_view": view,
            "group_size": len(ordered_ids),
            "bucket_id": bucket_id,
            "cohesion": _cohesion(ordered_ids, vectors),
            "same_row": int(len({feature.row for feature in feature_rows}) == 1),
            "same_column": int(len({feature.column for feature in feature_rows}) == 1),
            "dirty_type_count": len({feature.dirty_type for feature in feature_rows}),
            "baran_type_count": len({feature.baran_type for feature in feature_rows}),
            "baran_changed_share": round(
                sum(
                    self.cell_by_id[feature.cell_id].dirty_value != feature.baran_prediction
                    for feature in feature_rows
                ) / len(feature_rows),
                8,
            ),
        }
        return GroupQueryAction(
            query_id=query_id,
            suite=self.suite,
            dataset=self.dataset_name,
            arm="singleton" if view == "singleton" else "structured",
            group_view=view,
            cell_ids=ordered_ids,
            group_size=len(ordered_ids),
            prompt_schema_version=self.prompt_schema_version,
            prompt_information_policy=INFORMATION_POLICY,
            messages=material.messages,
            prompt_hash=material.prompt_hash,
            estimated_prompt_tokens=material.estimated_prompt_tokens,
            completion_token_ceiling=material.completion_token_ceiling,
            estimated_total_tokens=material.estimated_total_tokens,
            group_features=group_features,
        )

    def generate(self) -> GroupGenerationResult:
        candidates: list[GroupQueryAction] = []
        linkage_bucket_sizes: list[int] = []
        for cell in self.cells:
            candidates.append(
                self._action(
                    "singleton",
                    (cell.cell_id,),
                    bucket_id=f"cell:{cell.cell_id}",
                    vectors=self.combined_by_id,
                )
            )
        for view in GROUP_VIEWS[1:]:
            for bucket_id, member_ids, vectors in self._buckets(view):
                linkage_bucket_sizes.append(len(member_ids))
                leaf_order = stable_average_linkage_order(member_ids, vectors)
                for size in self.non_singleton_sizes:
                    for window in exact_windows(leaf_order, size):
                        candidates.append(
                            self._action(
                                view,
                                window,
                                bucket_id=bucket_id,
                                vectors=vectors,
                            )
                        )

        # Exact duplicate requests can arise from overlapping public FD
        # components.  Prompt hash is the physical-request identity.
        by_prompt_hash: dict[str, GroupQueryAction] = {}
        by_query_id: dict[str, GroupQueryAction] = {}
        for action in sorted(candidates, key=lambda item: item.query_id):
            previous = by_query_id.get(action.query_id)
            if previous is not None:
                previous_record = previous.as_dict()
                action_record = action.as_dict()
                previous_features = previous_record.pop("group_features")
                action_features = action_record.pop("group_features")
                if json.dumps(previous_record, sort_keys=True) != json.dumps(action_record, sort_keys=True):
                    raise RuntimeError(f"query_id collision with mutable request: {action.query_id}")
                # Multiple public-FD components can yield the same physical
                # request. Keep one deterministic feature provenance; request
                # identity, prompt, membership, and cost remain unchanged.
                if json.dumps(action_features, sort_keys=True) < json.dumps(previous_features, sort_keys=True):
                    by_query_id[action.query_id] = action
                continue
            by_query_id[action.query_id] = action
        for action in by_query_id.values():
            by_prompt_hash.setdefault(action.prompt_hash, action)
        actions = tuple(sorted(by_prompt_hash.values(), key=lambda action: action.query_id))

        singleton_counts = Counter(
            cell_id
            for action in actions
            if action.group_view == "singleton"
            for cell_id in action.cell_ids
        )
        if set(singleton_counts) != set(self.cell_by_id) or any(count != 1 for count in singleton_counts.values()):
            raise RuntimeError("singleton generation must cover every cell exactly once")
        memberships = Counter(cell_id for action in actions for cell_id in action.cell_ids)
        non_singleton_covered = {
            cell_id
            for action in actions
            if action.group_size > 1
            for cell_id in action.cell_ids
        }
        counts_by_view_size = Counter(
            f"{action.group_view}:{action.group_size}" for action in actions
        )
        membership_values = [memberships[cell_id] for cell_id in sorted(self.cell_by_id)]
        audit = {
            "candidate_count_before_prompt_dedup": len(candidates),
            "candidate_count_after_prompt_dedup": len(actions),
            "deduplicated_request_count": len(candidates) - len(actions),
            "counts_by_view_size": dict(sorted(counts_by_view_size.items())),
            "all_cells_have_singleton": True,
            "non_singleton_coverage_share": round(
                len(non_singleton_covered) / len(self.cells), 8
            ) if self.cells else 0.0,
            "membership_count_min": min(membership_values) if membership_values else 0,
            "membership_count_median": float(median(membership_values)) if membership_values else 0.0,
            "membership_count_max": max(membership_values) if membership_values else 0,
            "unsafe_annotation_fields_read": False,
            "linkage_exact_optimal_limit": EXACT_OPTIMAL_LINKAGE_LIMIT,
            "large_sparse_linkage_bucket_count": sum(
                size > EXACT_OPTIMAL_LINKAGE_LIMIT for size in linkage_bucket_sizes
            ),
            "maximum_linkage_bucket_size": max(linkage_bucket_sizes, default=0),
            "large_bucket_distance": "exact_sparse_cosine",
            "large_bucket_optimal_leaf_ordering": False,
        }
        return GroupGenerationResult(actions=actions, audit=audit)


def generate_group_queries(
    dataset: Any,
    cells: Sequence[SafeCell],
    baran_by_cell: Any = None,
    *,
    fd_components: Sequence[Any] = (),
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    call_overhead_tokens: int = 0,
    similar_row_count: int = 3,
) -> tuple[GroupQueryAction, ...]:
    """Return the immutable v1 singleton and overlapping group actions."""

    return GroupGenerator(
        dataset,
        cells,
        baran_by_cell,
        fd_components=fd_components,
        group_sizes=group_sizes,
        prompt_schema_version=prompt_schema_version,
        call_overhead_tokens=call_overhead_tokens,
        similar_row_count=similar_row_count,
    ).generate().actions


__all__ = [
    "DEFAULT_GROUP_SIZES",
    "EXACT_OPTIMAL_LINKAGE_LIMIT",
    "GROUP_VIEWS",
    "GroupGenerationResult",
    "GroupGenerator",
    "GroupQueryAction",
    "exact_windows",
    "generate_group_queries",
    "stable_average_linkage_order",
]

