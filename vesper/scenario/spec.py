"""Scenario spec: the seeded JSON contract every lane consumes.

Step 2 scope: mission geometry + environment placeholders. The randomizer
(sweeps over these fields) lands in Step 7.
"""
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScenarioSpec:
    seed: int = 0
    world: str = "flat"                    # world generator key; real worlds in Step 3
    takeoff_alt_m: float = 3.0
    # mission waypoints as (north_m, east_m, alt_m) relative to spawn/home
    waypoints: list = field(default_factory=list)
    wind_speed_ms: float = 0.0
    wind_dir_deg: float = 0.0
    visibility_m: float = 10000.0
    # buildings: [{"center":[north,east], "size":[dn,de], "height":h}], meters
    buildings: list = field(default_factory=list)
    max_sim_s: float = 150.0
    spawn_east: float = 0.0
    range_noise_std: float = 0.1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))


def square_scenario(seed: int = 0, side_m: float = 4.0, alt_m: float = 3.0) -> ScenarioSpec:
    s = side_m
    return ScenarioSpec(
        seed=seed, takeoff_alt_m=alt_m,
        waypoints=[[s, 0, alt_m], [s, s, alt_m], [0, s, alt_m], [0, 0, alt_m]],
    )


def city_scenario(seed: int = 0, alt_m: float = 3.0) -> ScenarioSpec:
    """Fly north through a building corridor and back."""
    from vesper.worlds import sample_city_block
    return ScenarioSpec(
        seed=seed, world="city", takeoff_alt_m=alt_m,
        waypoints=[[5, 0, alt_m], [10, 0, alt_m], [14, 0, alt_m], [0, 0, alt_m]],
        buildings=sample_city_block(seed),
    )


def crash_scenario(seed: int = 0, alt_m: float = 3.0) -> ScenarioSpec:
    """Same corridor with a prism blocking it: the mission commands flight
    through the building; PhysX gets the last word."""
    from vesper.worlds.layout import blocking_building
    spec = city_scenario(seed, alt_m)
    spec.world = "city-crash"
    spec.buildings = spec.buildings + [blocking_building(8.0)]
    spec.waypoints = [[14, 0, alt_m]]
    spec.max_sim_s = 75.0
    return spec
