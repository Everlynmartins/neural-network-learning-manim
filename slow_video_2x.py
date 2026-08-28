from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av


def slow_video(source: Path, destination: Path, factor: int = 2) -> None:
    """Duplicate each video frame, preserving the original FPS and visuals."""
    if factor < 1:
        raise ValueError("factor deve ser um inteiro positivo")
    if not source.is_file():
        raise FileNotFoundError(f"Vídeo de origem não encontrado: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Origem e destino devem ser arquivos diferentes")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        rate = input_stream.average_rate or Fraction(60, 1)

        with av.open(str(destination), mode="w") as output_container:
            output_stream = output_container.add_stream(
                "libx264", rate=rate, options={"bf": "0", "crf": "18"}
            )
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = "yuv420p"

            output_index = 0
            for frame in input_container.decode(input_stream):
                rgb_frame = frame.to_ndarray(format="rgb24")
                for _ in range(factor):
                    # Each duplicate needs its own frame object because the
                    # encoder may retain a reference while reordering frames.
                    duplicate = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
                    duplicate.pts = output_index
                    for packet in output_stream.encode(duplicate):
                        output_container.mux(packet)
                    output_index += 1

            for packet in output_stream.encode():
                output_container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduz a velocidade de um vídeo duplicando cada quadro.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()
    slow_video(args.source, args.destination, args.factor)
    print(args.destination)


if __name__ == "__main__":
    main()
