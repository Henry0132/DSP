"""Convert the source environment clips into local, looping GIFs.

Usage: python docs/tools/build_environment_gifs.py --source-dir output/environment-clips
Requires imageio-ffmpeg. Source URLs are recorded in assets/environments/SOURCES.md.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import imageio_ffmpeg


def convert(name, source_dir, output_dir):
    output = output_dir / f"{name}.gif"
    filters = (
        "fps=12,scale=320:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
        "-i", str(source_dir / f"{name}.mp4"), "-filter_complex", filters,
        "-an", "-loop", "0", str(output),
    ], check=True)
    print(f"{output.name}: {output.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parents[1] / "assets" / "environments"
    names = ["antmaze-giant-v0", "antsoccer-arena-v0", "cube-double-v0", "scene-v0", "puzzle-4x4-v0"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda name: convert(name, args.source_dir, output_dir), names))
