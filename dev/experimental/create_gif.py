from PIL import Image
import os

def create_gif_from_images(image_folder, pattern="plant_snapshot_", output_path="plant_animation.gif", duration=200):
    """
    Create an animated GIF from PNG files in a folder.

    Parameters:
    - image_folder: folder where images are stored
    - pattern: common prefix for PNGs (e.g., "plant_snapshot_")
    - output_path: full path for output GIF
    - duration: time per frame in milliseconds
    """
    # Get sorted image list
    images = sorted([img for img in os.listdir(image_folder) if img.startswith(pattern) and img.endswith(".png")])

    if not images:
        print("No matching images found.")
        return

    # Open and convert to RGB (in case they're RGBA)
    frames = [Image.open(os.path.join(image_folder, img)).convert("RGB") for img in images]

    # Save as animated GIF
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {output_path}")


def main():
    folder = "figures/normal"                     
    filename_prefix = "plant_"
    output_gif = os.path.join(folder, "plant_growth.gif")
    frame_duration_ms = 300

    create_gif_from_images(folder, filename_prefix, output_gif, frame_duration_ms)


if __name__ == "__main__":
    main()