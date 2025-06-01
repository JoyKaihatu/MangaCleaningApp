import subprocess
import os

class InpaintingScript:
    def __init__(self, image_dir, mask_dir, output_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir

    def run_inpainting(self):
        os.makedirs(self.output_dir, exist_ok=True)
        command = [
            "iopaint", "run",
            "--model=anime-lama",
            "--device=cpu",
            f"--image={self.image_dir}",
            f"--mask={self.mask_dir}",
            f"--output={self.output_dir}"
        ]
        subprocess.run(command)
        print(f"Inpainting completed. Output saved to {self.output_dir}")
        return