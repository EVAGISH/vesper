# Bringing a Fab / Unreal level into vesper

The realism ceiling for our worlds is game-engine content. Fab levels are Unreal
projects, so one manual step happens on the Mac in Unreal Engine: export the level
to USD. Everything after that is automated (`scripts/prepare_world.py` on the droplet).

## 1. One-time setup (Mac)

1. Install the **Epic Games Launcher**, sign in, install **Unreal Engine 5.4 or newer**
   (~40 GB; the Mac build is fine for exporting, no GPU work involved).
2. On [fab.com](https://www.fab.com) claim the level. First target:
   **Medieval Village Megascans Sample** (Quixel, free) — photogrammetry village with
   a forest edge, exactly the "forest edge with a town" brief. Other good free ones:
   Electric Dreams Env (jungle), City Sample (city), Open World Demo Collection (highlands).
3. In the Launcher: **Library → Fab Library → Create Project** for the sample. Open it.
4. **Edit → Plugins**: enable **USD Importer** (ships with the engine, off by default). Restart.

## 2. Export the level (5 minutes)

Open the level (for the Medieval sample it is the `MedievalVillage` map).

**Bake procedural content first.** Levels that scatter foliage with PCG graphs export
empty unless the graph is baked: select the PCG volume(s) → Details → **Generate**,
and if there is a "Bake"/"Convert to static" button use it. Painted foliage (Foliage
mode) needs nothing — it exports as USD PointInstancers.

Then **File → Export All…**, pick file type **Universal Scene Description (.usda)**,
save to e.g. `~/vesper-exports/medieval/medieval.usda`, and in the options dialog set:

| Option | Value | Why |
|---|---|---|
| Stage Options → Meters Per Unit | leave `0.01` | cm units; `prepare_world` rescales |
| Stage Options → Up Axis | leave `Z` | |
| Asset Options → **Bake Materials** | **on** | Unreal material graphs don't transfer; baking writes PBR textures |
| Asset Options → Material Baking → resolution | 2048 (4096 for hero meshes) | texel density |
| Asset Options → **Export Static Mesh Source Data** | **on** | Nanite meshes otherwise export the coarse fallback mesh |
| Asset Options → Lowest / Highest Mesh LOD | 0 / 0 | full detail only |
| Landscape → Lowest / Highest Landscape LOD | 0 / 0 | full-resolution terrain |
| Landscape → Landscape Bake Resolution | 4096 × 4096 | the ground texture is what the camera sees most |
| Export Sublayers | off | one file is easier to ship |
| Export Actor Folders | on | keeps the prim tree readable |

Or from **Tools → Execute Python Script** / the Output Log python prompt (same options):

```python
import unreal
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
opts = unreal.LevelExporterUSDOptions()
opts.stage_options.meters_per_unit = 0.01
opts.inner.export_actor_folders = True
opts.inner.export_sublayers = False
opts.inner.lowest_landscape_lod = 0
opts.inner.highest_landscape_lod = 0
opts.inner.landscape_bake_resolution = unreal.IntPoint(4096, 4096)
opts.inner.asset_options.bake_materials = True
opts.inner.asset_options.export_static_mesh_source_data = True
opts.inner.asset_options.lowest_mesh_lod = 0
opts.inner.asset_options.highest_mesh_lod = 0
task = unreal.AssetExportTask()
task.object, task.options = world, opts
task.filename = "/Users/theooltean/vesper-exports/medieval/medieval.usda"
task.automated = task.replace_identical = True
task.prompt = False
unreal.Exporter.run_asset_export_task(task)
```

Expect a few hundred MB to a few GB (textures dominate). The export writes the level
file plus an asset folder next to it — ship the whole directory.

## 3. Into the sim (automated)

```bash
rsync -az ~/vesper-exports/medieval/ root@<droplet>:vesper/assets/medieval/
infra/do/ssh.sh 'cd vesper/docker && docker compose run --rm sim \
    /isaac-sim/python.sh scripts/prepare_world.py assets/medieval/medieval.usda --spawn 0 0'
```

`prepare_world.py` writes `medieval_world.usda` (meters, Z-up, static triangle-mesh
colliders on every mesh including foliage prototypes), reports bounds and the ground
height at the spawn, and prints the `terrain` block for a ScenarioSpec. Then:

```bash
python3 scripts/make_scenario.py imported 0 medieval0.json \
    --usd assets/medieval/medieval_world.usda --spawn <x> <y> --ground <z>
docker compose run --rm sim /isaac-sim/python.sh scripts/fly_mission.py medieval0.json
scripts/capture_pull.sh
```

## What does not transfer

- Wind animation on foliage, procedural weather layers, decals with runtime blending.
- Lumen / Nanite / virtual textures — engine tricks; RTX path tracing replaces the first,
  source-data export covers the second, baked textures the third.
- Translucent and subsurface materials bake approximately (leaves look fine, glass less so).
