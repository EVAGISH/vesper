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
