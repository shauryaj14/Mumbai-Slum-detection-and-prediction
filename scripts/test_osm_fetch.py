import osmnx as ox

ox.settings.max_query_area_size = 50_000_000_000  # 50,000 km^2 - comfortably larger than Mumbai's ~780 km^2

AOI_BBOX = (72.78134, 18.90838, 72.97509, 19.26522)  # west, south, east, north

tags_list = [
    ("railway", {"railway": True}),
    ("industrial", {"landuse": "industrial"}),
    ("roads", {"highway": True}),
    ("water", {"natural": "water"}),
]

for name, tags in tags_list:
    try:
        gdf = ox.features_from_bbox(bbox=(AOI_BBOX[3], AOI_BBOX[1], AOI_BBOX[2], AOI_BBOX[0]), tags=tags)
        print(f"{name}: {len(gdf)} features found")
    except Exception as e:
        print(f"{name}: FAILED - {type(e).__name__}: {e}")
