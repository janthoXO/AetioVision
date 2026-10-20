"""Registry of 2D text-to-medical-image adapters, one module per model."""

from . import medisyn, minim, roentgen

MODELS = [*medisyn.MODELS, *roentgen.MODELS, *minim.MODELS]
