"""Standalone MAVSDK mission: arm, takeoff to 3m, 4x4m square, land.

Runs in its OWN process (vanilla asyncio) -- Isaac Sim's kit runtime patches
asyncio in-process and kills MAVSDK's gRPC connection, so the mission must
live outside it. Telemetry-paced throughout; exits 0 after touchdown.
"""
import asyncio
import sys

from vesper.scenario import ScenarioSpec


async def wait_alt(drone, target, above):
    async for pos in drone.telemetry.position():
        alt = pos.relative_altitude_m
        if (alt > target) if above else (alt < target):
            return


async def wait_near_global(drone, lat, lon, tol_m=0.8):
    async for p in drone.telemetry.position():
        dn = (p.latitude_deg - lat) * 111111.0
        de = (p.longitude_deg - lon) * 111111.0 * 0.7  # cos(lat) rough, fine at this scale
        if (dn * dn + de * de) ** 0.5 < tol_m:
            return


async def run(spec: ScenarioSpec):
    from mavsdk import System

    drone = System()
    await drone.connect(system_address="udp://:14540")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("mission: connected", flush=True)
            break
    async for health in drone.telemetry.health():
        if health.is_armable:
            break
    for name in ("MPC_XY_CRUISE", "MPC_XY_VEL_MAX"):
        try:
            await drone.param.set_param_float(name, float(spec.cruise_ms))
        except Exception as e:  # older PX4 param names; fly at defaults
            print(f"mission: could not set {name}: {e}", flush=True)
    await drone.action.arm()
    print("mission: armed", flush=True)
    await drone.action.set_takeoff_altitude(spec.takeoff_alt_m)
    await drone.action.takeoff()
    await wait_alt(drone, spec.takeoff_alt_m - 0.4, above=True)
    print("mission: at altitude", flush=True)

    # square via PX4's own navigator (goto), avoiding the offboard plugin,
    # whose mavsdk_server backend crashes under lockstep timing
    home = None
    async for h in drone.telemetry.home():
        home = h
        break
    import math
    lat0, lon0 = home.latitude_deg, home.longitude_deg
    m2lat = 1.0 / 111111.0
    m2lon = 1.0 / (111111.0 * math.cos(math.radians(lat0)))
    for n, e, alt in spec.waypoints:
        lat, lon = lat0 + n * m2lat, lon0 + e * m2lon
        abs_alt = home.absolute_altitude_m + alt
        await drone.action.goto_location(lat, lon, abs_alt, 0.0)
        await wait_near_global(drone, lat, lon)
        print(f"mission: corner ({n},{e})", flush=True)
    await drone.action.land()
    await wait_alt(drone, 0.3, above=False)
    print("mission: landed", flush=True)


if __name__ == "__main__":
    asyncio.run(run(ScenarioSpec.load(sys.argv[1])))
