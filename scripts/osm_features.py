import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import osmnx as ox
from shapely.geometry import box
from shapely.strtree import STRtree

ox.settings.max_query_area_size = 50_000_000_000

AOI_POLY = box(72.78134, 18.90838, 72.97509, 19.26522)
PROJECTED_CRS = "EPSG:32643"
CBD_LON, CBD_LAT = 72.8352, 18.9398


def fetch_osm_layer(tags):
    try:
        gdf = ox.features_from_polygon(AOI_POLY, tags=tags)
        return gdf
    except Exception as e:
        print(f"  Warning: fetch failed for tags {tags}: {e}")
        return gpd.GeoDataFrame(geometry=[])


def nearest_distance(points_gdf, features_gdf, label):
    if len(features_gdf) == 0:
        print(f"  No features found for {label} - filling with NaN")
        return np.full(len(points_gdf), np.nan)
    features_proj = features_gdf.to_crs(PROJECTED_CRS)
    points_proj = points_gdf.to_crs(PROJECTED_CRS)
    geoms = list(features_proj.geometry.values)
    tree = STRtree(geoms)
    distances = []
    for pt in points_proj.geometry:
        nearest_idx = tree.nearest(pt)
        nearest_geom = geoms[nearest_idx]
        distances.append(pt.distance(nearest_geom))
    return np.array(distances)


def main(grid_csv_path, out_csv_path):
    df = pd.read_csv(grid_csv_path)
    points_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["center_lon"], df["center_lat"]), crs="EPSG:4326"
    )

    print("Fetching OSM layers for Mumbai (one-time pull)...")
    railways = fetch_osm_layer({"railway": True})
    print(f"  Railways: {len(railways)} features")
    industrial = fetch_osm_layer({"landuse": "industrial"})
    print(f"  Industrial land: {len(industrial)} features")
    roads = fetch_osm_layer({"highway": True})
    print(f"  Roads: {len(roads)} features")
    water = fetch_osm_layer({"natural": "water"})
    print(f"  Water bodies: {len(water)} features")

    print("Computing distances...")
    df["dist_to_railway_m"] = nearest_distance(points_gdf, railways, "railways")
    df["dist_to_industry_m"] = nearest_distance(points_gdf, industrial, "industrial land")
    df["dist_to_road_m"] = nearest_distance(points_gdf, roads, "roads")
    df["dist_to_water_m"] = nearest_distance(points_gdf, water, "water")

    cbd_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([CBD_LON], [CBD_LAT]), crs="EPSG:4326")
    cbd_proj = cbd_point.to_crs(PROJECTED_CRS).geometry.iloc[0]
    points_proj = points_gdf.to_crs(PROJECTED_CRS)
    df["dist_to_cbd_m"] = points_proj.geometry.distance(cbd_proj)

    df.to_csv(out_csv_path, index=False)
    print(f"Wrote {out_csv_path} with {len(df)} cells")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python osm_features.py <grid.csv> <out.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
