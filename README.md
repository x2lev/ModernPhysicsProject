# Simulating 2D Wavefunction Evolution with Matplotlib

Numerical solver for the 2D time-dependent Schrödinger equation using the split-operator method.

## Usage

```bash
python main.py
```

Select a configuration file and choose visualization type. Output saved to `media/output.mp4`.

## Requirements

- Python
- CuPy (requires CUDA- or ROCm-capable GPU. If you wish to get this code up and running without such a GPU, please reach out!)
- NumPy
- Matplotlib
- imageio
- commentjson
- questionary

## Configuration

Edit JSON files in `configurations/` to change simulation parameters.

See `mpp.pdf` for theory and implementation details.