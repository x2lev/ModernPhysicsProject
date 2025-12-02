# Simulating 2D Wavefunction Evolution with Matplotlib

Numerical solver for the 2D time-dependent Schrödinger equation using the split-operator method.

## Usage

```bash
python main.py
```

Select a configuration file and choose visualization type. Output saved to `media/output.mp4`. Some example videos can be found in [this playlist on YouTube](https://youtube.com/playlist?list=PLHH1EiZsu_gt0se-nknGVan4IIjMcRxDP&si=WmvT9c9WGguMfkac).

## Requirements

- Python
- CuPy*
- NumPy
- Matplotlib
- imageio
- commentjson
- questionary

*requires CUDA- or ROCm-capable GPU. If you wish to get this code up and running without such a GPU, please reach out to me at lev.kryvenko@proton.me!

## Configuration

Edit the JSONC files in `configurations/` to change simulation parameters.

See `tex/mpp.pdf` for a theoretical derivation of the split-operator method.