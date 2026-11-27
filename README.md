# R&S @ hackaTUM 2026 - From the Gate to the Hand

Team repo for the Rohde & Schwarz challenge at hackaTUM 2026: next-generation
security screening with a **handheld device that combines radar and a camera**.

## The challenge in two lines

- **Task 1 - "Where is my handheld?"** One patch, then a pose: estimate the
  device trajectory from video and IMU, accurately enough that every radar
  glimpse knows where it belongs.
- **Task 2 - "Airport screening is taken, what else?"** Find a new purpose for
  the device: materials, industry, medicine, infrastructure, rescue, or a place
  nobody has named yet.

Details and open questions: [docs/challenge.md](docs/challenge.md).

## Team and tracks

| Who | Track | Folder |
|---|---|---|
| Teammate A | Classical visual-inertial SLAM: ORB-SLAM3 | [tracks/orbslam3](tracks/orbslam3) |
| Alp | Spatial AI: learned VO/SLAM and 3D foundation models | [tracks/spatial-ai](tracks/spatial-ai) |
| Teammate C | Data, calibration, evaluation, radar-patch stitching | [data](data), [eval](eval) |
| Teammate D | Task 2 use case, demo and pitch | [tracks/task2](tracks/task2) |

C and D are proposals; swap freely. The one thing everyone shares is the
**trajectory contract** in [docs/architecture.md](docs/architecture.md): every
pipeline writes the same file format, so one script scores them all.

## Start here

1. [docs/ROADMAP.md](docs/ROADMAP.md) - phases, owners, checkpoints.
2. [docs/architecture.md](docs/architecture.md) - how the tracks fit together.
3. Your track folder's `README.md`.

## Repo layout

```
docs/            challenge brief, roadmap, architecture, decisions
data/            sequence layout (raw data is gitignored)
eval/            trajectory evaluation (evo) and result table
tracks/
  orbslam3/      ORB-SLAM3 mono-inertial pipeline
  spatial-ai/    learned and foundation-model pose estimation
  task2/         new-use-case research and pitch material
```

The R&S slide deck is company restricted: do not commit it or copy it verbatim.
