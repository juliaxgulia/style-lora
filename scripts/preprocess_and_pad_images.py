import argparse
from pathlib import Path
from PIL import Image, ImageOps


def pad_image(image: Image.Image, target_size: int = 1024, fill_color=(255, 255, 255)) -> Image.Image:
    """Resize and pad an image to a square target size.

    Args:
        image: Input PIL Image.
        target_size: Desired output side length.
        fill_color: RGB padding color.

    Returns:
        Padded PIL Image.
    """
    # Convert to RGB to avoid issues with grayscale or alpha channels
    image = image.convert("RGB")

    # Preserve aspect ratio using thumbnail
    image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    # Create new canvas and paste centered image
    new_image = Image.new("RGB", (target_size, target_size), fill_color)
    offset = ((target_size - image.width) // 2, (target_size - image.height) // 2)
    new_image.paste(image, offset)
    return new_image


def process_directory(input_dir: Path, output_dir: Path, target_size: int, fill_color: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Parse hex color string if provided
    color = tuple(int(fill_color[i:i+2], 16) for i in (0, 2, 4)) if fill_color else (255, 255, 255)
    for path in input_dir.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            with Image.open(path) as img:
                padded = pad_image(img, target_size=target_size, fill_color=color)
                padded.save(output_dir / f"{path.stem}_padded.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize and pad images")
    parser.add_argument("input_dir", type=Path, help="Directory with input images")
    parser.add_argument("output_dir", type=Path, help="Where to store padded images")
    parser.add_argument("--size", type=int, default=1024, help="Output dimension (square)")
    parser.add_argument("--color", type=str, default="ffffff", help="Padding color in hex (e.g., ffffff)")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.size, args.color)


if __name__ == "__main__":
    main()
