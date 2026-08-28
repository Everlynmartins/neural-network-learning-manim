from __future__ import annotations

import argparse
from pathlib import Path

import av


def inspect_video(video_path: Path, output_dir: Path, times: list[float]) -> None:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base)
        fps = float(stream.average_rate)
        print(f"width={stream.width}")
        print(f"height={stream.height}")
        print(f"fps={fps:.3f}")
        print(f"frames={stream.frames}")
        print(f"duration={duration:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for time_seconds in times:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            container.seek(
                int(time_seconds * av.time_base),
                backward=True,
                any_frame=False,
            )
            for frame in container.decode(stream):
                if float(frame.time or 0) >= time_seconds:
                    destination = output_dir / f"frame_{time_seconds:05.1f}s.png"
                    frame.to_image().save(destination)
                    print(destination)
                    break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("times", nargs="+", type=float)
    args = parser.parse_args()
    inspect_video(args.video, args.output_dir, args.times)


if __name__ == "__main__":
    main()
