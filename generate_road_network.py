from road_network import RoadNetwork

city_country = "Porto, Portugal"
path = "data/network"

network = RoadNetwork(city_country, network_type="drive", retain_all=True, truncate_by_edge=True)
network.save(path=path)
print(f"{city_country} network saved to {path}.")