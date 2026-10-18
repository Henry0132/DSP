# Environment animation sources

These clips illustrate benchmark environments; they are not DSP policy rollouts.
The GIFs preserve each source clip's duration and sequence, converted to 320 × 320,
12 fps, a 128-color palette, and an infinite loop.

| GIF | Source clip |
| --- | --- |
| `antmaze-giant-v0.gif` | [OGBench](https://seohong.me/projects/ogbench/assets/videos/antmaze-giant-navigate-v0_data.mp4) |
| `antsoccer-arena-v0.gif` | [OGBench](https://seohong.me/projects/ogbench/assets/videos/antsoccer-arena-stitch-v0_data.mp4) |
| `cube-double-v0.gif` | [InFOM](https://chongyi-zheng.github.io/assets/papers/infom/videos/cube-double-v0_task5.mp4) |
| `scene-v0.gif` | [OGBench](https://seohong.me/projects/ogbench/assets/videos/scene-v0_data.mp4) |
| `puzzle-4x4-v0.gif` | [OGBench](https://seohong.me/projects/ogbench/assets/videos/visual-puzzle-4x4-v0_data.mp4) |

Project pages: [OGBench](https://seohong.me/projects/ogbench/),
[InFOM](https://chongyi-zheng.github.io/infom/).

Regenerate with `docs/tools/build_environment_gifs.py` using the source clips named
after their corresponding GIFs (with the `.mp4` extension).
