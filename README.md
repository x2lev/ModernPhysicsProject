# Simulating 2D Wavefunction Evolution with Matplotlib

Numerical solver for the 2D time-dependent Schrödinger equation using the split-operator method.

## Usage

```bash
python main.py
```

Select a configuration file and choose visualization type. Output saved to `media/output.mp4`.

## Requirements

- Python
- CuPy*
- NumPy
- Matplotlib
- imageio
- commentjson
- questionary

*requires CUDA- or ROCm-capable GPU. If you wish to get this code up and running without such a GPU, please reach out to lev.kryvenko@proton.me!

## Configuration

Edit JSON files in `configurations/` to change simulation parameters.

See `mpp.pdf` for the theoretical derivations of the split-operator method.

## Result

Output video files are saved as 'media/output.mp4'.