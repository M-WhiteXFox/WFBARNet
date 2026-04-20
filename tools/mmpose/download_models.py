from pathlib import Path


def main() -> None:
    target = Path("assets/weights/pose")
    target.mkdir(parents=True, exist_ok=True)
    print(f"Place your RTMPose checkpoints under: {target.resolve()}")


if __name__ == "__main__":
    main()
