#!/usr/bin/env python3
"""Treeline-shift analysis using the native 2 m swissALTI3D DEM.

Scientific definition
---------------------
delta_elevation_m = DEM(endpoint on the current treeline)
                    - DEM(sample point on the 1946 treeline)

Both endpoint elevations and the smoothed gradient used for matching come
directly from the same native 2 m DEM. No 0.5 m regional DEM is selected or
resampled. Each 1946 sample is traced both uphill and downhill; the first
nearby or exact intersection with the current treeline is a candidate.

Outputs are deliberately split into:
  * accepted_shift.csv             final seven-column measurements only
  * all_candidates.csv             both directions and rejection reasons
  * path_decisions.csv             one decision row for every 1946 sample
  * accepted_shift_paths.gpkg      final path geometry
  * all_candidate_paths.gpkg       all reached candidate geometry
  * run_summary.json               parameters, counts, and validation checks

All thresholds are CLI options so they can be sensitivity-tested. This single
script can process one region, run regions 00-19, combine accepted measurements,
and create the final statistical figure.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window, from_bounds
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from shapely import STRtree
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points, substring, unary_union
from shapely.prepared import prep


BASE_DIR = Path(
    r"xx"
)
TARGET_CRS = "EPSG:2056"
DEFAULT_DEM = BASE_DIR / "xx.tif"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "test"
DEM_RESOLUTION_M = 2.0
VHM_YEAR_GPKG = BASE_DIR.parent / "xx.gpkg"
VHM_YEAR_LAYER = "xx"
COMBINED_CSV = DEFAULT_OUTPUT_ROOT / "xx.csv"
FIGURE_STEM = DEFAULT_OUTPUT_ROOT / "delta_elevation_nature_style"
FIGURE_SOURCE_DATA = DEFAULT_OUTPUT_ROOT / "xx.csv"
FIGURE_CAPTION = DEFAULT_OUTPUT_ROOT / "xx.txt"
ACCEPTED_CSV_COLUMNS = [
    "region_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "delta_elevation_m",
    "vhm-year",
]

N_BOOTSTRAP = 4000
RANDOM_SEED = 20260904

NEGATIVE = "#6FA8C9"
STABLE = "#B9B9B9"
POSITIVE = "#D8894B"
MEAN = "#C75B39"
MEDIAN = "#4D4D4D"
CI_COLOR = "#595959"
GRID = "#E6E6E6"
TEXT = "#222222"


@dataclass(frozen=True)
class TraceConfig:
    step_m: float
    max_distance_m: float
    hit_tolerance_m: float
    overlap_tolerance_m: float
    min_gradient: float
    max_turn_deg: float
    direction_memory: float


@dataclass
class GradientSurface:
    elevation: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    valid_fraction: np.ndarray
    transform: Affine

    def sample_elevation(self, point: Point) -> float | None:
        """Bilinearly sample elevation from the native 2 m DEM."""
        col_corner, row_corner = (~self.transform) * (point.x, point.y)
        coords = np.array([[row_corner - 0.5], [col_corner - 0.5]])
        value = float(
            map_coordinates(
                self.elevation, coords, order=1, mode="constant", cval=np.nan
            )[0]
        )
        return value if np.isfinite(value) else None

    def sample_gradient(self, point: Point) -> tuple[float, float] | None:
        """Bilinearly sample dz/dx and dz/dy at an EPSG:2056 point."""
        col_corner, row_corner = (~self.transform) * (point.x, point.y)
        coords = np.array([[row_corner - 0.5], [col_corner - 0.5]])
        gx = float(map_coordinates(self.gx, coords, order=1, mode="constant", cval=np.nan)[0])
        gy = float(map_coordinates(self.gy, coords, order=1, mode="constant", cval=np.nan)[0])
        valid_fraction = float(
            map_coordinates(
                self.valid_fraction, coords, order=1, mode="constant", cval=0.0
            )[0]
        )
        if valid_fraction < 0.80 or not (np.isfinite(gx) and np.isfinite(gy)):
            return None
        return gx, gy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify and plot treeline elevation shifts using one native 2 m DEM",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        choices=("region", "batch", "plot", "all"),
        default="all",
        help=(
            "region: one region; batch: regions and combined CSV; "
            "plot: plot combined CSV; all: batch then plot"
        ),
    )
    parser.add_argument("--region", default="00", help="Region ID, e.g. 00 or 19")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=[f"{value:02d}" for value in range(20)],
        help="Region IDs for batch/all mode; default: 00 through 19",
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed regions")
    parser.add_argument(
        "-f",
        "--f",
        dest="_ipykernel_connection_file",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--old", type=Path, default=None)
    parser.add_argument("--new", type=Path, default=None)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-interval", type=float, default=10.0)
    parser.add_argument("--smooth-sigma", type=float, default=12.0)
    parser.add_argument("--trace-step", type=float, default=5.0)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=300.0,
        help="Maximum terrain-gradient search distance in metres",
    )
    parser.add_argument("--hit-tolerance", type=float, default=10.0)
    parser.add_argument(
        "--max-detour-ratio",
        type=float,
        default=2.0,
        help=(
            "Reject a traced path longer than this multiple of the direct distance "
            "to the nearest contemporary treeline"
        ),
    )
    parser.add_argument(
        "--max-vertical-shift",
        type=float,
        default=200.0,
        help="Reject, rather than clip, absolute delta elevations above this value",
    )
    parser.add_argument(
        "--large-shift-threshold",
        type=float,
        default=100.0,
        help=(
            "Absolute delta elevation treated as a large shift for ambiguity and "
            "steep-path quality checks"
        ),
    )
    parser.add_argument(
        "--max-large-shift-grade",
        type=float,
        default=1.5,
        help=(
            "For large shifts only, reject absolute elevation change divided by "
            "planimetric path length above this grade"
        ),
    )
    parser.add_argument("--overlap-tolerance", type=float, default=1.0)
    parser.add_argument("--min-gradient", type=float, default=0.005)
    parser.add_argument("--max-turn", type=float, default=65.0)
    parser.add_argument("--direction-memory", type=float, default=0.35)
    parser.add_argument("--vertical-zero-tolerance", type=float, default=2.0)
    parser.add_argument("--old-crossing-clearance", type=float, default=3.0)
    parser.add_argument("--endpoint-separation", type=float, default=2.0)
    argv = [] if Path(sys.argv[0]).stem == "ipykernel_launcher" else None
    args = parser.parse_args(argv)
    args.region = str(args.region).zfill(2)
    region_root = BASE_DIR / "treelinextractionv4-revised" / args.region
    if args.old is None:
        args.old = (
            region_root
            / "1946"
            / f"1946_region{args.region}_treeline_skeleton_line.shp"
        )
    if args.new is None:
        args.new = (
            region_root
            / "current"
            / f"current_region{args.region}_treeline_skeleton_line.shp"
        )
    if args.output is None:
        args.output = DEFAULT_OUTPUT_ROOT / f"region{args.region}"
    return args


def require_inputs(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input(s):\n  " + "\n  ".join(missing))


def add_vhm_year(table: pd.DataFrame) -> pd.DataFrame:
    """Assign the VHM acquisition year at every current endpoint."""
    result = table.drop(columns=["vhm-year"], errors="ignore").copy()
    if result.empty:
        result["vhm-year"] = pd.Series(dtype="Int64")
        return result

    require_inputs([VHM_YEAR_GPKG])
    grid = gpd.read_file(VHM_YEAR_GPKG, layer=VHM_YEAR_LAYER)[["year", "geometry"]]
    if grid.crs is None:
        raise ValueError(f"VHM year grid has no CRS: {VHM_YEAR_GPKG}")
    grid = grid.to_crs(TARGET_CRS).rename(columns={"year": "vhm-year"})
    grid["vhm-year"] = pd.to_numeric(grid["vhm-year"], errors="coerce").astype(
        "Int64"
    )
    if grid["vhm-year"].isna().any():
        raise ValueError(f"VHM year grid contains missing/non-numeric years: {VHM_YEAR_GPKG}")

    points = gpd.GeoDataFrame(
        {"_row_id": np.arange(len(result), dtype=int)},
        geometry=gpd.points_from_xy(result["end_x"], result["end_y"]),
        crs=TARGET_CRS,
    )

    def collapse_matches(joined: pd.DataFrame) -> dict[int, int]:
        matches: dict[int, int] = {}
        for row_id, values in joined.groupby("_row_id")["vhm-year"]:
            distinct = sorted({int(value) for value in values.dropna()})
            if len(distinct) > 1:
                raise ValueError(
                    f"Endpoint row {int(row_id)} intersects multiple VHM years: {distinct}"
                )
            if distinct:
                matches[int(row_id)] = distinct[0]
        return matches

    joined = gpd.sjoin(
        points, grid[["vhm-year", "geometry"]], how="left", predicate="intersects"
    )
    years = collapse_matches(joined)
    missing = [row_id for row_id in range(len(result)) if row_id not in years]
    if missing:
        nearest = gpd.sjoin_nearest(
            points[points["_row_id"].isin(missing)],
            grid[["vhm-year", "geometry"]],
            how="left",
            max_distance=1.0,
            distance_col="vhm_year_distance_m",
        )
        years.update(collapse_matches(nearest))
    missing = [row_id for row_id in range(len(result)) if row_id not in years]
    if missing:
        raise ValueError(
            f"No VHM acquisition year within 1 m for {len(missing)} endpoint(s)"
        )

    result["vhm-year"] = pd.array(
        [years[row_id] for row_id in range(len(result))], dtype="Int64"
    )
    return result


def read_clean_lines(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Input has no CRS: {path}")
    gdf = gdf.copy()
    gdf["source_fid"] = np.arange(len(gdf), dtype=int)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.to_crs(TARGET_CRS).explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.geom_type == "LineString"].copy()
    gdf = gdf[gdf.geometry.length > 0].reset_index(drop=True)
    gdf["line_id"] = np.arange(len(gdf), dtype=int)
    if gdf.empty:
        raise ValueError(f"No usable LineString geometry in {path}")
    return gdf


def union_bounds(*gdfs: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    bounds = np.vstack([gdf.total_bounds for gdf in gdfs])
    return (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].min()),
        float(bounds[:, 2].max()),
        float(bounds[:, 3].max()),
    )


def intersection_fraction(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    a_box = box(*a)
    b_box = box(*b)
    if a_box.area <= 0:
        return 0.0
    return float(a_box.intersection(b_box).area / a_box.area)


def validate_native_2m_dem(
    dem_path: Path,
    study_bounds: tuple[float, float, float, float],
    region_id: str,
) -> float:
    """Require one native 2 m DEM that completely covers the treelines."""
    require_inputs([dem_path])
    with rasterio.open(dem_path) as ds:
        if ds.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")
        if str(ds.crs).upper() != TARGET_CRS:
            raise ValueError(f"DEM must use {TARGET_CRS}; got {ds.crs}: {dem_path}")
        resolution = (abs(float(ds.res[0])), abs(float(ds.res[1])))
        if not all(math.isclose(value, DEM_RESOLUTION_M, abs_tol=1e-6) for value in resolution):
            raise ValueError(
                "This workflow only accepts a native 2 m DEM; "
                f"got {resolution[0]:g} x {resolution[1]:g} m: {dem_path}"
            )
        coverage = intersection_fraction(study_bounds, tuple(ds.bounds))
    if coverage < 0.999:
        raise RuntimeError(
            f"The 2 m DEM covers only {coverage:.3f} of region{region_id}: {dem_path}"
        )
    return coverage


def sample_old_treeline(
    old: gpd.GeoDataFrame, interval_m: float
) -> gpd.GeoDataFrame:
    """Sample line-cell midpoints without duplicated feature endpoints."""
    rows: list[dict[str, Any]] = []
    points: list[Point] = []
    sample_id = 0
    for row in old.itertuples():
        line = row.geometry
        length = float(line.length)
        n_cells = max(1, int(math.ceil(length / interval_m)))
        cell_length = length / n_cells
        for cell in range(n_cells):
            chainage = (cell + 0.5) * cell_length
            point = line.interpolate(chainage)
            rows.append(
                {
                    "sample_id": sample_id,
                    "old_line_id": int(row.line_id),
                    "old_source_fid": int(row.source_fid),
                    "old_chainage_m": float(chainage),
                    "start_x": float(point.x),
                    "start_y": float(point.y),
                }
            )
            points.append(point)
            sample_id += 1
    return gpd.GeoDataFrame(rows, geometry=points, crs=TARGET_CRS)


def clipped_window(ds: rasterio.DatasetReader, bounds: tuple[float, ...]) -> Window:
    requested = from_bounds(*bounds, transform=ds.transform)
    full = Window(0, 0, ds.width, ds.height)
    try:
        return requested.round_offsets().round_lengths().intersection(full)
    except rasterio.errors.WindowError as exc:
        raise ValueError("Study area does not overlap the selected DEM") from exc


def build_gradient_surface(
    ds: rasterio.DatasetReader,
    target_bounds: tuple[float, float, float, float],
    smooth_sigma_m: float,
) -> GradientSurface:
    if str(ds.crs).upper() != TARGET_CRS:
        raise ValueError(
            f"Test implementation expects the DEM in {TARGET_CRS}; got {ds.crs}"
        )
    source_res_x = abs(float(ds.transform.a))
    source_res_y = abs(float(ds.transform.e))
    if not (
        math.isclose(source_res_x, DEM_RESOLUTION_M, abs_tol=1e-6)
        and math.isclose(source_res_y, DEM_RESOLUTION_M, abs_tol=1e-6)
    ):
        raise ValueError(
            "Gradient tracing requires the native 2 m DEM; "
            f"got {source_res_x:g} x {source_res_y:g} m"
        )
    window = clipped_window(ds, target_bounds)
    masked = ds.read(1, window=window, masked=True)
    z = np.asarray(masked.filled(np.nan), dtype=np.float32)
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("DEM analysis window contains no valid elevation cells")

    transform = ds.window_transform(window)
    cell_x = abs(float(transform.a))
    cell_y = abs(float(transform.e))
    sigma_cells = max(0.0, smooth_sigma_m / math.sqrt(cell_x * cell_y))

    if sigma_cells > 0:
        # Keep the 2 m workflow in float32 and reuse buffers. Some regions have
        # tens of millions of cells, so float64 temporaries would require several
        # unnecessary gigabytes of RAM.
        smooth = np.where(valid, z, np.float32(0.0)).astype(np.float32)
        denominator = valid.astype(np.float32)
        gaussian_filter(smooth, sigma=sigma_cells, output=smooth)
        gaussian_filter(denominator, sigma=sigma_cells, output=denominator)
        np.divide(
            smooth,
            denominator,
            out=smooth,
            where=denominator > np.float32(1e-6),
        )
        smooth[denominator <= np.float32(1e-6)] = np.nan
    else:
        smooth = z.copy()
        denominator = valid.astype(np.float32)

    # Raster rows run southward, hence the negative y spacing.
    gy, gx = np.gradient(smooth, -cell_y, cell_x)
    unreliable = denominator < 0.80
    gx[unreliable] = np.nan
    gy[unreliable] = np.nan
    return GradientSurface(
        elevation=z,
        gx=gx.astype(np.float32),
        gy=gy.astype(np.float32),
        valid_fraction=denominator.astype(np.float32),
        transform=transform,
    )


def unit_vector(x: float, y: float) -> tuple[float, float] | None:
    norm = math.hypot(x, y)
    if not np.isfinite(norm) or norm <= 0:
        return None
    return x / norm, y / norm


def angle_degrees(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = float(np.clip(a[0] * b[0] + a[1] * b[1], -1.0, 1.0))
    return math.degrees(math.acos(dot))


def nearest_intersection_point(
    segment: LineString, start: Point, target: Any
) -> Point | None:
    intersection = segment.intersection(target)
    if intersection.is_empty:
        return None
    return nearest_points(start, intersection)[1]


def append_if_distinct(coords: list[tuple[float, float]], point: Point) -> None:
    xy = (float(point.x), float(point.y))
    if not coords or math.dist(coords[-1], xy) > 1e-8:
        coords.append(xy)


def trace_to_new_treeline(
    start: Point,
    direction: str,
    surface: GradientSurface,
    new_union: Any,
    prepared_new_buffer: Any,
    config: TraceConfig,
) -> tuple[dict[str, Any] | None, str]:
    sign = 1.0 if direction == "uphill" else -1.0
    coords = [(float(start.x), float(start.y))]
    current = start
    previous_vector: tuple[float, float] | None = None
    turns: list[float] = []
    accumulated = 0.0
    max_steps = int(math.ceil(config.max_distance_m / config.step_m))
    pending_snap: dict[str, Any] | None = None
    snap_lookahead_remaining = 0
    snap_lookahead_steps = max(
        2, int(math.ceil(2.0 * config.hit_tolerance_m / config.step_m)) + 1
    )

    def finish_pending_snap() -> tuple[dict[str, Any], str]:
        assert pending_snap is not None
        result = dict(pending_snap)
        snap_coords = result.pop("coords")
        result["geometry"] = LineString(
            snap_coords if len(snap_coords) > 1 else snap_coords * 2
        )
        return result, "reached_with_tolerance_after_lookahead"

    def stop_or_finish_pending(
        failure_status: str,
    ) -> tuple[dict[str, Any] | None, str]:
        """Keep a valid nearby hit when the gradient trace subsequently fails."""
        if pending_snap is None:
            return None, failure_status
        result, _ = finish_pending_snap()
        return result, f"reached_with_tolerance_before_{failure_status}"

    for _ in range(max_steps):
        gradient = surface.sample_gradient(current)
        if gradient is None:
            return stop_or_finish_pending("outside_or_nodata_gradient")
        magnitude = math.hypot(*gradient)
        if magnitude < config.min_gradient:
            return stop_or_finish_pending("gradient_too_flat")

        raw_vector = unit_vector(sign * gradient[0], sign * gradient[1])
        if raw_vector is None:
            return stop_or_finish_pending("invalid_gradient")
        vector = raw_vector
        if previous_vector is not None:
            raw_turn = angle_degrees(previous_vector, raw_vector)
            if raw_turn > config.max_turn_deg:
                return stop_or_finish_pending("gradient_turn_too_sharp")
            blended = unit_vector(
                (1.0 - config.direction_memory) * raw_vector[0]
                + config.direction_memory * previous_vector[0],
                (1.0 - config.direction_memory) * raw_vector[1]
                + config.direction_memory * previous_vector[1],
            )
            if blended is None:
                return stop_or_finish_pending("invalid_blended_direction")
            vector = blended
            turns.append(angle_degrees(previous_vector, vector))

        step = min(config.step_m, config.max_distance_m - accumulated)
        if step <= 1e-9:
            break
        next_point = Point(
            current.x + step * vector[0], current.y + step * vector[1]
        )
        segment = LineString([current, next_point])

        in_tolerance_corridor = prepared_new_buffer.intersects(segment)
        exact_point = (
            nearest_intersection_point(segment, current, new_union)
            if in_tolerance_corridor
            else None
        )
        if exact_point is not None:
            append_if_distinct(coords, exact_point)
            geometry = LineString(coords if len(coords) > 1 else coords * 2)
            return (
                {
                    "geometry": geometry,
                    "endpoint": exact_point,
                    "hit_type": "exact",
                    "endpoint_gap_m": 0.0,
                    "mean_turn_deg": float(np.mean(turns)) if turns else 0.0,
                    "max_turn_deg": float(max(turns)) if turns else 0.0,
                },
                "reached_exact",
            )

        # Keep tracing briefly after first entering the tolerance corridor.  In
        # most cases the streamline then crosses the line exactly.  If it only
        # grazes the corridor, retain the closest approach as an explicit snap.
        if in_tolerance_corridor:
            on_segment, endpoint = nearest_points(segment, new_union)
            gap = float(on_segment.distance(endpoint))
            if gap <= config.hit_tolerance_m + 1e-8:
                if pending_snap is None or gap < pending_snap["endpoint_gap_m"]:
                    snap_coords = list(coords)
                    append_if_distinct(snap_coords, on_segment)
                    append_if_distinct(snap_coords, endpoint)
                    pending_snap = {
                        "coords": snap_coords,
                        "endpoint": endpoint,
                        "hit_type": "tolerance_snap",
                        "endpoint_gap_m": gap,
                        "mean_turn_deg": float(np.mean(turns)) if turns else 0.0,
                        "max_turn_deg": float(max(turns)) if turns else 0.0,
                    }
                if snap_lookahead_remaining == 0:
                    snap_lookahead_remaining = snap_lookahead_steps
        elif pending_snap is not None:
            return finish_pending_snap()

        coords.append((float(next_point.x), float(next_point.y)))
        current = next_point
        accumulated += step
        previous_vector = vector

        if pending_snap is not None:
            snap_lookahead_remaining -= 1
            if snap_lookahead_remaining <= 0:
                return finish_pending_snap()

    if pending_snap is not None:
        return finish_pending_snap()
    return None, "max_distance_without_match"


def locate_new_line(endpoint: Point, new: gpd.GeoDataFrame) -> dict[str, Any]:
    distances = new.geometry.distance(endpoint).to_numpy()
    position = int(np.argmin(distances))
    row = new.iloc[position]
    return {
        "new_line_id": int(row.line_id),
        "new_source_fid": int(row.source_fid),
        "new_chainage_m": float(row.geometry.project(endpoint)),
        "end_to_new_m": float(distances[position]),
    }


def path_crosses_old(
    geometry: LineString, old_union: Any, clearance_m: float
) -> bool:
    if geometry.length <= clearance_m:
        return False
    trimmed = substring(geometry, clearance_m, geometry.length)
    return not trimmed.intersection(old_union).is_empty


def candidate_rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        0.0 if row["sign_consistent"] else 1.0,
        float(row["quality_score"]),
        float(row["path_length_m"]),
        float(row["endpoint_gap_m"]),
        0.0 if row["hit_type"] == "exact" else 1.0,
        0.0 if row["search_direction"] == "uphill" else 1.0,
    )


def build_topology_conflicts(
    selected: list[dict[str, Any]], endpoint_separation_m: float
) -> list[set[int]]:
    conflicts = [set() for _ in selected]
    if not selected:
        return conflicts
    geometries = [row["geometry"] for row in selected]
    tree = STRtree(geometries)
    for i, geometry in enumerate(geometries):
        for j_value in tree.query(geometry, predicate="intersects"):
            j = int(j_value)
            if j <= i:
                continue
            conflicts[i].add(j)
            conflicts[j].add(i)

    by_new_line: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(selected):
        by_new_line[int(row["new_line_id"])].append(i)
    for indices in by_new_line.values():
        indices.sort(key=lambda i: float(selected[i]["new_chainage_m"]))
        for offset, i in enumerate(indices):
            chainage_i = float(selected[i]["new_chainage_m"])
            for j in indices[offset + 1 :]:
                separation = float(selected[j]["new_chainage_m"]) - chainage_i
                if separation >= endpoint_separation_m:
                    break
                conflicts[i].add(j)
                conflicts[j].add(i)
    return conflicts


def choose_strict_non_crossing(
    selected: list[dict[str, Any]], endpoint_separation_m: float
) -> None:
    conflicts = build_topology_conflicts(selected, endpoint_separation_m)
    order = sorted(
        range(len(selected)),
        key=lambda i: (candidate_rank(selected[i]), int(selected[i]["sample_id"])),
    )
    accepted: set[int] = set()
    for i in order:
        blocking = sorted(conflicts[i].intersection(accepted))
        if blocking:
            selected[i]["accepted"] = 0
            selected[i]["reject_reason"] = "topology_conflict"
            selected[i]["conflict_candidate_ids"] = ";".join(
                str(selected[j]["candidate_id"]) for j in blocking
            )
        else:
            accepted.add(i)
            selected[i]["accepted"] = 1
            selected[i]["reject_reason"] = ""
            selected[i]["conflict_candidate_ids"] = ""


def dataframe_without_geometry(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    return frame


def prepare_for_geopackage(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Avoid collisions with GeoPackage's reserved integer primary key."""
    output = gdf.copy()
    geometry_name = output.geometry.name
    used = set(output.columns)
    renames: dict[str, str] = {}
    for column in output.columns:
        if column == geometry_name or column.lower() != "fid":
            continue
        candidate = "source_fid"
        suffix = 1
        while candidate in used:
            candidate = f"source_fid_{suffix}"
            suffix += 1
        renames[column] = candidate
        used.add(candidate)
    return output.rename(columns=renames) if renames else output


def write_gpkg(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> Path:
    target = path
    if target.exists():
        try:
            target.unlink()
        except PermissionError:
            target = path.with_name(f"{path.stem}_calibrated{path.suffix}")
            try:
                if target.exists():
                    target.unlink()
            except PermissionError:
                target = path.with_name(
                    f"{path.stem}_calibrated_{int(time.time())}{path.suffix}"
                )
            print(
                f"      warning: {path.name} is open in another program; "
                f"writing {target.name} instead"
            )
    prepare_for_geopackage(gdf).to_file(
        target, layer=layer, driver="GPKG", index=False
    )
    return target


def run_region(args: argparse.Namespace) -> None:
    started = time.time()
    require_inputs([args.old, args.new])

    print("[1/7] Reading and harmonizing treelines...")
    old = read_clean_lines(args.old)
    new = read_clean_lines(args.new)
    study_bounds = union_bounds(old, new)
    dem_path = args.dem
    dem_coverage = validate_native_2m_dem(dem_path, study_bounds, args.region)
    print(f"      old lines={len(old)}, new lines={len(new)}")
    print(f"      selected DEM: {dem_path.name}")

    config = TraceConfig(
        step_m=args.trace_step,
        max_distance_m=args.max_distance,
        hit_tolerance_m=args.hit_tolerance,
        overlap_tolerance_m=args.overlap_tolerance,
        min_gradient=args.min_gradient,
        max_turn_deg=args.max_turn,
        direction_memory=args.direction_memory,
    )

    print("[2/7] Sampling the 1946 treeline...")
    samples = sample_old_treeline(old, args.sample_interval)
    old_length = float(old.geometry.length.sum())
    print(f"      samples={len(samples):,}, treeline length={old_length:.1f} m")

    buffer_m = args.max_distance + max(50.0, 3.0 * args.smooth_sigma)
    gradient_bounds = (
        study_bounds[0] - buffer_m,
        study_bounds[1] - buffer_m,
        study_bounds[2] + buffer_m,
        study_bounds[3] + buffer_m,
    )

    print("[3/7] Building the smoothed DEM-gradient surface...")
    with rasterio.open(dem_path) as dem:
        surface = build_gradient_surface(
            dem,
            gradient_bounds,
            args.smooth_sigma,
        )
        print(
            f"      native DEM resolution={abs(dem.res[0]):g} x {abs(dem.res[1]):g} m"
        )

        print("[4/7] Tracing uphill and downhill candidates...")
        new_union = unary_union(new.geometry.to_numpy())
        old_union = unary_union(old.geometry.to_numpy())
        new_buffer = new_union.buffer(args.hit_tolerance)
        prepared_new_buffer = prep(new_buffer)
        start_elevations = [
            surface.sample_elevation(point) for point in samples.geometry
        ]

        candidate_rows: list[dict[str, Any]] = []
        trace_status: dict[int, dict[str, str]] = defaultdict(dict)
        endpoints: list[Point] = []
        candidate_id = 0

        for sample_position, sample in enumerate(samples.itertuples()):
            start = sample.geometry
            distance_to_new = float(start.distance(new_union))
            traces: list[tuple[str, dict[str, Any], str]] = []

            if distance_to_new <= args.overlap_tolerance:
                endpoint = nearest_points(start, new_union)[1]
                coords = [(start.x, start.y), (endpoint.x, endpoint.y)]
                geometry = LineString(coords)
                traces.append(
                    (
                        "stable",
                        {
                            "geometry": geometry,
                            "endpoint": endpoint,
                            "hit_type": "overlap_or_near_overlap",
                            "endpoint_gap_m": distance_to_new,
                            "mean_turn_deg": 0.0,
                            "max_turn_deg": 0.0,
                        },
                        "overlap_or_near_overlap",
                    )
                )
                trace_status[int(sample.sample_id)]["uphill"] = "not_run_overlap"
                trace_status[int(sample.sample_id)]["downhill"] = "not_run_overlap"
            else:
                for direction in ("uphill", "downhill"):
                    traced, status = trace_to_new_treeline(
                        start,
                        direction,
                        surface,
                        new_union,
                        prepared_new_buffer,
                        config,
                    )
                    trace_status[int(sample.sample_id)][direction] = status
                    if traced is not None:
                        traces.append((direction, traced, status))

            for direction, traced, status in traces:
                endpoint = traced.pop("endpoint")
                geometry = traced.pop("geometry")
                new_location = locate_new_line(endpoint, new)
                straight_distance = float(start.distance(endpoint))
                path_length = float(geometry.length)
                sinuosity = path_length / max(straight_distance, 1e-9)
                row = {
                    "candidate_id": candidate_id,
                    "sample_id": int(sample.sample_id),
                    "old_line_id": int(sample.old_line_id),
                    "old_source_fid": int(sample.old_source_fid),
                    "old_chainage_m": float(sample.old_chainage_m),
                    "search_direction": direction,
                    "trace_status": status,
                    "start_x": float(start.x),
                    "start_y": float(start.y),
                    "end_x": float(endpoint.x),
                    "end_y": float(endpoint.y),
                    "start_elevation_m": start_elevations[sample_position],
                    "nearest_new_distance_m": distance_to_new,
                    "path_length_m": path_length,
                    "straight_distance_m": straight_distance,
                    "sinuosity": float(sinuosity),
                    **traced,
                    **new_location,
                    "geometry": geometry,
                }
                candidate_rows.append(row)
                endpoints.append(endpoint)
                candidate_id += 1

            if (sample_position + 1) % 250 == 0:
                print(
                    f"      {sample_position + 1:,}/{len(samples):,} samples, "
                    f"{len(candidate_rows):,} reached candidates",
                    end="\r",
                )
        print()

        endpoint_elevations = [surface.sample_elevation(point) for point in endpoints]
        dem_metadata = {
            "path": dem_path.resolve().as_posix(),
            "crs": str(dem.crs),
            "native_resolution_m": [abs(float(dem.res[0])), abs(float(dem.res[1]))],
            "coverage_fraction": round(float(dem_coverage), 6),
            "nodata": None if dem.nodata is None else float(dem.nodata),
            "bounds": [float(value) for value in dem.bounds],
        }

    for row, end_elevation in zip(candidate_rows, endpoint_elevations):
        row["end_elevation_m"] = end_elevation
        start_elevation = row["start_elevation_m"]
        if start_elevation is None or end_elevation is None:
            row["delta_elevation_m"] = None
            row["delta_direction"] = "unknown"
            row["sign_consistent"] = 0
        else:
            delta = float(end_elevation - start_elevation)
            row["delta_elevation_m"] = delta
            if abs(delta) <= args.vertical_zero_tolerance:
                delta_direction = "stable"
            elif delta > 0:
                delta_direction = "uphill"
            else:
                delta_direction = "downhill"
            row["delta_direction"] = delta_direction
            row["sign_consistent"] = int(
                row["search_direction"] == "stable"
                or delta_direction == "stable"
                or row["search_direction"] == delta_direction
            )

        row["self_intersection"] = int(not row["geometry"].is_simple)
        row["crosses_old_treeline"] = int(
            path_crosses_old(row["geometry"], old_union, args.old_crossing_clearance)
        )
        row["quality_score"] = float(
            row["path_length_m"]
            + 2.0 * row["endpoint_gap_m"]
            + 80.0 * max(0.0, row["sinuosity"] - 1.0)
            + 0.4 * row["mean_turn_deg"]
            + (0.0 if row["hit_type"] == "exact" else 5.0)
            + (0.0 if row["sign_consistent"] else 1000.0)
        )
        hard_reasons: list[str] = []
        if row["delta_elevation_m"] is None:
            hard_reasons.append("missing_dem_elevation")
        if not row["sign_consistent"]:
            hard_reasons.append("delta_sign_mismatch")
        if row["self_intersection"]:
            hard_reasons.append("self_intersection")
        if row["crosses_old_treeline"]:
            hard_reasons.append("crosses_old_treeline")
        if row["path_length_m"] > args.max_distance + args.hit_tolerance + 1e-6:
            hard_reasons.append("path_exceeds_limit")
        detour_reference = max(
            float(row["nearest_new_distance_m"]), float(args.hit_tolerance)
        )
        if (
            row["path_length_m"]
            > args.max_detour_ratio * detour_reference + 1e-6
        ):
            hard_reasons.append("excessive_detour")
        if (
            row["delta_elevation_m"] is not None
            and abs(float(row["delta_elevation_m"])) > args.large_shift_threshold
            and abs(float(row["delta_elevation_m"]))
            / max(float(row["path_length_m"]), 1e-9)
            > args.max_large_shift_grade
        ):
            hard_reasons.append("large_shift_on_excessive_grade")
        if (
            row["delta_elevation_m"] is not None
            and abs(float(row["delta_elevation_m"])) > args.max_vertical_shift
        ):
            hard_reasons.append("vertical_shift_exceeds_limit")
        row["candidate_eligible"] = int(not hard_reasons)
        row["selected_for_sample"] = 0
        row["accepted"] = 0
        row["reject_reason"] = ";".join(hard_reasons)
        row["conflict_candidate_ids"] = ""

    print("[5/7] Selecting one candidate per sample and resolving topology...")
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_sample[int(row["sample_id"])].append(row)

    local_selected: list[dict[str, Any]] = []
    for rows in by_sample.values():
        large_trace_directions = {
            str(row["search_direction"])
            for row in rows
            if row["delta_elevation_m"] is not None
            and row["sign_consistent"]
            and abs(float(row["delta_elevation_m"]))
            > args.large_shift_threshold
        }
        if {"uphill", "downhill"}.issubset(large_trace_directions):
            for row in rows:
                if row["candidate_eligible"]:
                    row["candidate_eligible"] = 0
                    row["reject_reason"] = "ambiguous_bidirectional_large_shift"
            continue
        eligible = [row for row in rows if row["candidate_eligible"]]
        if not eligible:
            continue
        chosen = min(eligible, key=candidate_rank)
        chosen["selected_for_sample"] = 1
        local_selected.append(chosen)
        for row in rows:
            if row is not chosen and row["candidate_eligible"]:
                row["reject_reason"] = "alternate_not_selected"

    choose_strict_non_crossing(local_selected, args.endpoint_separation)
    accepted_rows = [row for row in local_selected if row["accepted"] == 1]
    print(
        f"      reached candidates={len(candidate_rows):,}, "
        f"local matches={len(local_selected):,}, accepted={len(accepted_rows):,}"
    )

    candidate_gdf = gpd.GeoDataFrame(candidate_rows, geometry="geometry", crs=TARGET_CRS)
    local_gdf = gpd.GeoDataFrame(local_selected, geometry="geometry", crs=TARGET_CRS)
    accepted_gdf = gpd.GeoDataFrame(accepted_rows, geometry="geometry", crs=TARGET_CRS)

    print("[6/7] Writing decision tables and geospatial outputs...")
    args.output.mkdir(parents=True, exist_ok=True)
    all_candidates_csv = args.output / "all_candidates.csv"
    accepted_csv = args.output / "accepted_shift.csv"
    decisions_csv = args.output / "path_decisions.csv"
    candidate_gpkg = args.output / "all_candidate_paths.gpkg"
    accepted_gpkg = args.output / "accepted_shift_paths.gpkg"
    samples_gpkg = args.output / "samples_1946.gpkg"

    dataframe_without_geometry(candidate_gdf).sort_values(
        ["sample_id", "candidate_id"]
    ).to_csv(all_candidates_csv, index=False)
    accepted_table = dataframe_without_geometry(accepted_gdf).sort_values("sample_id")
    accepted_table = add_vhm_year(accepted_table)
    accepted_table.insert(0, "region_id", args.region)
    accepted_table.loc[:, ACCEPTED_CSV_COLUMNS].to_csv(accepted_csv, index=False)

    decision_rows: list[dict[str, Any]] = []
    selected_by_sample = {int(row["sample_id"]): row for row in local_selected}
    for sample_position, sample in enumerate(samples.itertuples()):
        sample_id = int(sample.sample_id)
        selected = selected_by_sample.get(sample_id)
        base = {
            "sample_id": sample_id,
            "old_line_id": int(sample.old_line_id),
            "old_source_fid": int(sample.old_source_fid),
            "old_chainage_m": float(sample.old_chainage_m),
            "start_x": float(sample.start_x),
            "start_y": float(sample.start_y),
            "start_elevation_m": start_elevations[sample_position],
            "uphill_trace_status": trace_status[sample_id].get("uphill", "not_run"),
            "downhill_trace_status": trace_status[sample_id].get(
                "downhill", "not_run"
            ),
        }
        if selected is None:
            reached = by_sample.get(sample_id, [])
            reasons = sorted(
                {
                    row["reject_reason"] or "no_eligible_candidate"
                    for row in reached
                }
            )
            base.update(
                {
                    "candidate_id": None,
                    "accepted": 0,
                    "decision": "no_valid_match",
                    "reject_reason": (
                        ";".join(reasons) if reasons else "no_reached_candidate"
                    ),
                    "end_x": None,
                    "end_y": None,
                    "end_elevation_m": None,
                    "delta_elevation_m": None,
                    "delta_direction": "unknown",
                    "path_length_m": None,
                    "hit_type": None,
                }
            )
        else:
            base.update(
                {
                    "candidate_id": int(selected["candidate_id"]),
                    "accepted": int(selected["accepted"]),
                    "decision": "accepted"
                    if selected["accepted"]
                    else "topology_rejected",
                    "reject_reason": selected["reject_reason"],
                    "end_x": selected["end_x"],
                    "end_y": selected["end_y"],
                    "end_elevation_m": selected["end_elevation_m"],
                    "delta_elevation_m": selected["delta_elevation_m"],
                    "delta_direction": selected["delta_direction"],
                    "path_length_m": selected["path_length_m"],
                    "hit_type": selected["hit_type"],
                }
            )
        decision_rows.append(base)
    decisions = pd.DataFrame(decision_rows).sort_values("sample_id")
    decisions.to_csv(decisions_csv, index=False)

    if not candidate_gdf.empty:
        candidate_gpkg = write_gpkg(
            candidate_gdf, candidate_gpkg, "all_candidates"
        )
    if not accepted_gdf.empty:
        accepted_gpkg = write_gpkg(
            accepted_gdf, accepted_gpkg, "accepted_shift_paths"
        )
    samples_gpkg = write_gpkg(samples, samples_gpkg, "samples_1946")

    print("[7/7] Validating delta arithmetic and endpoint geometry...")
    algebra_error = 0.0
    max_start_distance = None
    max_end_distance = None
    pairwise_intersections = 0
    if not accepted_gdf.empty:
        algebra_error = float(
            np.nanmax(
                np.abs(
                    accepted_gdf["delta_elevation_m"]
                    - (
                        accepted_gdf["end_elevation_m"]
                        - accepted_gdf["start_elevation_m"]
                    )
                )
            )
        )
        max_start_distance = float(
            max(Point(row.start_x, row.start_y).distance(old_union) for row in accepted_gdf.itertuples())
        )
        max_end_distance = float(
            max(Point(row.end_x, row.end_y).distance(new_union) for row in accepted_gdf.itertuples())
        )
        accepted_tree = STRtree(list(accepted_gdf.geometry))
        pairs: set[tuple[int, int]] = set()
        for i, geometry in enumerate(accepted_gdf.geometry):
            for j_value in accepted_tree.query(geometry, predicate="intersects"):
                j = int(j_value)
                if j > i:
                    pairs.add((i, j))
        pairwise_intersections = len(pairs)

    if algebra_error > 1e-8:
        raise AssertionError(f"Delta arithmetic validation failed: {algebra_error}")
    if max_start_distance is not None and max_start_distance > 1e-6:
        raise AssertionError(f"Accepted start is not on old treeline: {max_start_distance}")
    if max_end_distance is not None and max_end_distance > 1e-6:
        raise AssertionError(f"Accepted end is not on new treeline: {max_end_distance}")
    if pairwise_intersections:
        raise AssertionError(
            f"Accepted paths contain {pairwise_intersections} pairwise intersections"
        )

    deltas = accepted_gdf["delta_elevation_m"] if not accepted_gdf.empty else pd.Series(dtype=float)
    directions = (
        accepted_gdf["delta_direction"].value_counts().to_dict()
        if not accepted_gdf.empty
        else {}
    )
    summary = {
        "method": "bidirectional_smoothed_2m_dem_gradient_streamline",
        "region_id": args.region,
        "definition": "delta_elevation_m = DEM(current endpoint) - DEM(1946 start)",
        "endpoint_elevation_source": "same native 2 m swissALTI3D DEM at both endpoints",
        "old_treeline": args.old.resolve().as_posix(),
        "new_treeline": args.new.resolve().as_posix(),
        "dem": dem_metadata,
        "parameters": {
            "sample_interval_m": args.sample_interval,
            "dem_resolution_m": DEM_RESOLUTION_M,
            "smooth_sigma_m": args.smooth_sigma,
            "trace_step_m": args.trace_step,
            "max_distance_m": args.max_distance,
            "hit_tolerance_m": args.hit_tolerance,
            "max_detour_ratio": args.max_detour_ratio,
            "max_vertical_shift_m": args.max_vertical_shift,
            "large_shift_threshold_m": args.large_shift_threshold,
            "max_large_shift_grade": args.max_large_shift_grade,
            "overlap_tolerance_m": args.overlap_tolerance,
            "min_gradient": args.min_gradient,
            "max_turn_deg": args.max_turn,
            "direction_memory": args.direction_memory,
            "vertical_zero_tolerance_m": args.vertical_zero_tolerance,
            "old_crossing_clearance_m": args.old_crossing_clearance,
            "endpoint_separation_m": args.endpoint_separation,
        },
        "counts": {
            "old_lines": int(len(old)),
            "new_lines": int(len(new)),
            "samples": int(len(samples)),
            "reached_candidates": int(len(candidate_gdf)),
            "local_selected": int(len(local_gdf)),
            "accepted": int(len(accepted_gdf)),
            "topology_rejected": int(
                (local_gdf["reject_reason"] == "topology_conflict").sum()
                if not local_gdf.empty
                else 0
            ),
            "no_valid_match": int((decisions["decision"] == "no_valid_match").sum()),
            "directions": {str(key): int(value) for key, value in directions.items()},
        },
        "delta_statistics_m": {
            "mean": None if deltas.empty else float(deltas.mean()),
            "median": None if deltas.empty else float(deltas.median()),
            "p05": None if deltas.empty else float(deltas.quantile(0.05)),
            "p95": None if deltas.empty else float(deltas.quantile(0.95)),
            "min": None if deltas.empty else float(deltas.min()),
            "max": None if deltas.empty else float(deltas.max()),
        },
        "validation": {
            "max_delta_arithmetic_error_m": algebra_error,
            "max_start_distance_to_1946_m": max_start_distance,
            "max_end_distance_to_current_m": max_end_distance,
            "accepted_pairwise_intersections": pairwise_intersections,
            "accepted_sign_mismatches": int(
                (accepted_gdf["sign_consistent"] == 0).sum()
                if not accepted_gdf.empty
                else 0
            ),
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "outputs": {
            "accepted_csv": accepted_csv.resolve().as_posix(),
            "all_candidates_csv": all_candidates_csv.resolve().as_posix(),
            "decisions_csv": decisions_csv.resolve().as_posix(),
            "accepted_gpkg": accepted_gpkg.resolve().as_posix(),
            "candidate_gpkg": candidate_gpkg.resolve().as_posix(),
            "samples_gpkg": samples_gpkg.resolve().as_posix(),
        },
    }
    summary_path = args.output / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["delta_statistics_m"], ensure_ascii=False, indent=2))
    print(f"Completed in {summary['elapsed_seconds']:.1f} s")
    print(f"Final usable CSV: {accepted_csv}")


def output_complete(output_dir: Path) -> bool:
    """Return True only when all essential outputs for one region exist."""
    required = ("run_summary.json", "accepted_shift.csv", "path_decisions.csv")
    if not all((output_dir / name).exists() for name in required):
        return False
    try:
        summary = json.loads(
            (output_dir / "run_summary.json").read_text(encoding="utf-8")
        )
        accepted_gpkg = Path(summary["outputs"]["accepted_gpkg"])
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False
    return accepted_gpkg.exists()


def completed_row(region_id: str, output_dir: Path, status: str) -> dict[str, Any]:
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    return {
        "region_id": region_id,
        "status": status,
        "return_code": 0,
        **summary.get("counts", {}),
        **{
            f"delta_{key}_m": value
            for key, value in summary.get("delta_statistics_m", {}).items()
        },
        "output_dir": str(output_dir),
    }


def write_batch_summary(rows: list[dict[str, Any]], started: float) -> None:
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("region_id")
    table.to_csv(DEFAULT_OUTPUT_ROOT / "batch_summary_00_19.csv", index=False)
    payload = {
        "regions_requested": [row["region_id"] for row in rows],
        "completed": [row["region_id"] for row in rows if row["return_code"] == 0],
        "failed": [row["region_id"] for row in rows if row["return_code"] != 0],
        "elapsed_seconds_so_far": round(time.time() - started, 2),
        "per_region": rows,
    }
    (DEFAULT_OUTPUT_ROOT / "batch_summary_00_19.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def combine_accepted(regions: list[str]) -> Path:
    tables: list[pd.DataFrame] = []
    for region_id in regions:
        region_dir = DEFAULT_OUTPUT_ROOT / f"region{region_id}"
        table = pd.read_csv(
            region_dir / "accepted_shift.csv", dtype={"region_id": str}
        )
        if "region_id" not in table.columns:
            table.insert(0, "region_id", region_id)
        table["region_id"] = table["region_id"].astype(str).str.zfill(2)
        if "vhm-year" not in table.columns or table["vhm-year"].isna().any():
            table = add_vhm_year(table)
        tables.append(table.loc[:, ACCEPTED_CSV_COLUMNS])
    combined = pd.concat(tables, ignore_index=True)
    output_path = COMBINED_CSV
    try:
        combined.to_csv(output_path, index=False)
    except PermissionError:
        output_path = COMBINED_CSV.with_name(
            f"{COMBINED_CSV.stem}_calibrated{COMBINED_CSV.suffix}"
        )
        try:
            combined.to_csv(output_path, index=False)
        except PermissionError:
            output_path = COMBINED_CSV.with_name(
                f"{COMBINED_CSV.stem}_calibrated_{int(time.time())}"
                f"{COMBINED_CSV.suffix}"
            )
            combined.to_csv(output_path, index=False)
        print(
            f"Combined CSV is open in another program; wrote {output_path.name} instead"
        )
    return output_path


def vertical_tolerance_from_summaries(region_ids: Iterable[str]) -> float:
    """Read the common classification threshold without storing it in shift CSVs."""
    tolerances: set[float] = set()
    for region_id in sorted({str(value).zfill(2) for value in region_ids}):
        summary_path = DEFAULT_OUTPUT_ROOT / f"region{region_id}" / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tolerances.add(float(summary["parameters"]["vertical_zero_tolerance_m"]))
    if len(tolerances) != 1:
        raise ValueError(
            "All regions must use one vertical-zero tolerance before plotting; "
            f"found {sorted(tolerances)}"
        )
    return tolerances.pop()


def region_command(args: argparse.Namespace, region_id: str) -> list[str]:
    script_path = (Path(__file__).resolve() if "__file__" in globals()
                   else DEFAULT_OUTPUT_ROOT / "treeline_shift_quantification.py")
    command = [
        sys.executable,
        "-u",
        "-X",
        "utf8",
        str(script_path),
        "--mode",
        "region",
        "--region",
        region_id,
        "--dem",
        str(args.dem),
        "--sample-interval",
        str(args.sample_interval),
        "--smooth-sigma",
        str(args.smooth_sigma),
        "--trace-step",
        str(args.trace_step),
        "--max-distance",
        str(args.max_distance),
        "--hit-tolerance",
        str(args.hit_tolerance),
        "--max-detour-ratio",
        str(args.max_detour_ratio),
        "--max-vertical-shift",
        str(args.max_vertical_shift),
        "--large-shift-threshold",
        str(args.large_shift_threshold),
        "--max-large-shift-grade",
        str(args.max_large_shift_grade),
        "--overlap-tolerance",
        str(args.overlap_tolerance),
        "--min-gradient",
        str(args.min_gradient),
        "--max-turn",
        str(args.max_turn),
        "--direction-memory",
        str(args.direction_memory),
        "--vertical-zero-tolerance",
        str(args.vertical_zero_tolerance),
        "--old-crossing-clearance",
        str(args.old_crossing_clearance),
        "--endpoint-separation",
        str(args.endpoint_separation),
    ]
    return command


def run_batch(args: argparse.Namespace) -> Path:
    """Run requested regions as isolated processes, then concatenate accepted rows."""
    regions = [str(region).zfill(2) for region in args.regions]
    invalid = [r for r in regions if not r.isdigit() or not 0 <= int(r) <= 19]
    if invalid:
        raise ValueError(f"Only region IDs 00-19 are allowed: {invalid}")
    require_inputs([args.dem])

    started = time.time()
    rows: list[dict[str, Any]] = []
    for position, region_id in enumerate(regions, start=1):
        output_dir = DEFAULT_OUTPUT_ROOT / f"region{region_id}"
        print(f"[{position}/{len(regions)}] region{region_id}", flush=True)
        if output_complete(output_dir) and not args.force:
            rows.append(completed_row(region_id, output_dir, "completed_existing"))
            write_batch_summary(rows, started)
            continue

        region_started = time.time()
        result = subprocess.run(region_command(args, region_id), check=False)
        elapsed = round(time.time() - region_started, 2)
        if result.returncode == 0 and output_complete(output_dir):
            row = completed_row(region_id, output_dir, "completed_new")
            row["elapsed_seconds_batch"] = elapsed
        else:
            row = {
                "region_id": region_id,
                "status": "failed",
                "return_code": int(result.returncode or 2),
                "elapsed_seconds_batch": elapsed,
                "output_dir": str(output_dir),
            }
        rows.append(row)
        write_batch_summary(rows, started)

    failed = [row for row in rows if row["return_code"] != 0]
    if failed:
        raise RuntimeError(f"Batch failed for regions: {[row['region_id'] for row in failed]}")
    combined = combine_accepted(regions)
    print(f"Combined accepted CSV: {combined}")
    return combined


def summarize_delta_elevation(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    tolerance: float,
) -> dict[str, float | int]:
    """Summarize accepted delta elevations with every observation counted once."""
    values = frame["delta_elevation_m"].dropna().to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite delta_elevation_m values to summarize")
    boot = np.empty(N_BOOTSTRAP, dtype=float)
    for first in range(0, N_BOOTSTRAP, 100):
        last = min(first + 100, N_BOOTSTRAP)
        draws = rng.integers(0, values.size, size=(last - first, values.size))
        boot[first:last] = values[draws].mean(axis=1)

    total = values.size
    return {
        "accepted_n": int(total),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "sample_variance_m2": None if total < 2 else float(np.var(values, ddof=1)),
        "sample_sd_m": None if total < 2 else float(np.std(values, ddof=1)),
        "ci95_low_m": float(np.quantile(boot, 0.025)),
        "ci95_high_m": float(np.quantile(boot, 0.975)),
        "downhill_pct": 100 * float(np.count_nonzero(values < -tolerance)) / total,
        "stable_pct": 100 * float(np.count_nonzero(np.abs(values) <= tolerance)) / total,
        "uphill_pct": 100 * float(np.count_nonzero(values > tolerance)) / total,
    }


def plot_combined(input_csv: Path = COMBINED_CSV) -> None:
    """Create the summary statistics and three-panel publication figure."""
    require_inputs([input_csv])
    data = pd.read_csv(input_csv, dtype={"region_id": str})
    if data.empty:
        raise ValueError(f"No accepted measurements in {input_csv}")
    missing = [column for column in ACCEPTED_CSV_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Combined CSV is missing required columns: {missing}")
    data["region_id"] = data["region_id"].astype(str).str.zfill(2)
    tolerance = vertical_tolerance_from_summaries(data["region_id"].unique())
    tolerance_label = f"{tolerance:g}"
    rng = np.random.default_rng(RANDOM_SEED)
    rows = [
        {
            "region_id": region_id,
            **summarize_delta_elevation(group, rng, tolerance),
        }
        for region_id, group in data.groupby("region_id", sort=True)
    ]
    regional = pd.DataFrame(rows).sort_values("mean_m", ascending=False)
    overall = {
        "region_id": "Overall",
        **summarize_delta_elevation(data, rng, tolerance),
    }
    source = pd.concat((pd.DataFrame([overall]), regional), ignore_index=True)
    source.insert(1, "vertical_zero_tolerance_m", tolerance)
    source.to_csv(FIGURE_SOURCE_DATA, index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(7.2, 7.0), facecolor="white")
    grid = figure.add_gridspec(
        3, 2, height_ratios=(5.2, 0.34, 1.35), width_ratios=(1.45, 1),
        hspace=0.18, wspace=0.22
    )
    effect = figure.add_subplot(grid[0, 0])
    direction = figure.add_subplot(grid[0, 1], sharey=effect)
    legend = figure.add_subplot(grid[1, :])
    distribution = figure.add_subplot(grid[2, :])
    y = np.arange(len(source))

    for index in range(1, len(source), 2):
        effect.axhspan(index - 0.5, index + 0.5, color="#F6F6F6", zorder=0)
        direction.axhspan(index - 0.5, index + 0.5, color="#F6F6F6", zorder=0)
    effect.axvline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.9)
    effect.hlines(
        y, source["ci95_low_m"], source["ci95_high_m"],
        color=[TEXT] + [CI_COLOR] * (len(source) - 1), linewidth=0.9
    )
    effect.scatter(
        source["mean_m"], y, s=[30] + [18] * (len(source) - 1),
        c=[TEXT] + [MEAN] * (len(source) - 1), edgecolor="white", linewidth=0.4, zorder=3
    )
    effect.scatter(source["median_m"], y, marker="|", s=65, c=MEDIAN)
    effect.set(yticks=y, yticklabels=source["region_id"], xlabel=r"$\Delta$ elevation (m)")
    effect.invert_yaxis()
    effect.set_title("a  Regional estimates", loc="left")
    effect.grid(axis="x", color=GRID, linewidth=0.55)

    downhill = source["downhill_pct"].to_numpy(float)
    stable = source["stable_pct"].to_numpy(float)
    uphill = source["uphill_pct"].to_numpy(float)
    direction.barh(y, downhill, height=0.56, color=NEGATIVE)
    direction.barh(y, stable, left=downhill, height=0.56, color=STABLE)
    direction.barh(y, uphill, left=downhill + stable, height=0.56, color=POSITIVE)
    direction.set(xlim=(0, 100), xlabel="Accepted delta-elevation observations (%)")
    direction.set_title("b  Direction composition", loc="left")
    direction.tick_params(axis="y", left=False, labelleft=False)
    direction.grid(axis="x", color=GRID, linewidth=0.55)
    for yi, count in zip(y, source["accepted_n"]):
        direction.text(98, yi, f"{int(count):,}", ha="right", va="center", fontsize=5.8)

    legend.axis("off")
    legend.legend(
        handles=(
            Line2D([0], [0], marker="o", color=CI_COLOR, markerfacecolor=MEAN,
                   markersize=4, label="Mean (95% bootstrap CI)"),
            Line2D([0], [0], marker="|", color="none", markeredgecolor=MEDIAN,
                   markersize=8, label="Median"),
            Patch(facecolor=NEGATIVE, label=f"Downhill (<−{tolerance_label} m)"),
            Patch(facecolor=STABLE, label=f"Stable (|Δ|≤{tolerance_label} m)"),
            Patch(facecolor=POSITIVE, label=f"Uphill (>+{tolerance_label} m)"),
        ),
        loc="center", ncol=5
    )

    values = data["delta_elevation_m"].dropna().to_numpy(float)
    values = values[np.isfinite(values)]
    edges = np.arange(np.floor(values.min() / 10) * 10, np.ceil(values.max() / 10) * 10 + 10, 10)
    density = gaussian_filter1d(np.histogram(values, bins=edges, density=True)[0], 1.1)
    centres = (edges[:-1] + edges[1:]) / 2
    distribution.fill_between(centres, 0, density, where=centres <= 0, color=NEGATIVE, alpha=0.82)
    distribution.fill_between(centres, 0, density, where=centres >= 0, color=POSITIVE, alpha=0.82)
    distribution.plot(centres, density, color="#666666", linewidth=0.55)
    distribution.axvspan(-tolerance, tolerance, color=STABLE, alpha=0.5)
    distribution.axvline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.9)
    distribution.axvline(overall["mean_m"], color=MEAN, linewidth=1)
    distribution.axvline(overall["median_m"], color=MEDIAN, linestyle=":", linewidth=1)
    distribution.set(xlabel=r"$\Delta$ elevation (m) = DEM(current) $-$ DEM(1946)", yticks=[])
    distribution.set_title("c  Pooled signed distribution", loc="left")
    distribution.spines["left"].set_visible(False)

    figure.subplots_adjust(left=0.105, right=0.965, top=0.955, bottom=0.095)
    figure.savefig(FIGURE_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".pdf"), facecolor="white")
    figure.savefig(FIGURE_STEM.with_suffix(".svg"), facecolor="white")
    plt.close(figure)

    FIGURE_CAPTION.write_text(
        "Fig. X | Elevational shifts of the alpine treeline between 1946 and current. "
        "Regional statistics and directional proportions are calculated directly "
        f"from accepted delta-elevation observations; 95% intervals use {N_BOOTSTRAP:,} "
        "bootstrap resamples of those observations. Stable shifts satisfy "
        f"|Δ elevation| ≤ {tolerance_label} m. Positive values indicate higher "
        "current endpoints and negative values lower current endpoints relative to 1946.",
        encoding="utf-8",
    )
    print(FIGURE_STEM.with_suffix(".png"))


def main() -> None:
    args = parse_args()
    if args.mode == "region":
        run_region(args)
    elif args.mode == "batch":
        run_batch(args)
    elif args.mode == "plot":
        plot_combined()
    else:
        plot_combined(run_batch(args))


if __name__ == "__main__":
    main()
