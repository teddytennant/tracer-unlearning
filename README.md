Tracer is a small JAX implementation of erased-concept identification from arXiv:2610.05601.
pip install -e .
python3 -m pytest
This is the spectral footprint, whitening, coverage pursuit, and the confidence drop that estimates how many concepts were erased. It is not a diffusion model and it does not attack Stable Diffusion. The short toy uses B=3 and alpha=0.5, not the paper's B=8 and alpha=0.30.
