# AetioVision baselines

The original text-to-image CLI, kept for comparison with the inpainting
pipeline in the repository root. It generates synthetic 2D medical images
from text prompts with off-the-shelf models. Run all commands from this
directory (`cd baselines`): model weights are read from `weights/` and images
are written to `outputs/`, both relative to it.

Run `python main.py` to get a numbered menu of the available models, each
listing its body parts and modality. Pick one, type a prompt, and the image
is saved as a timestamped PNG in `outputs/`, named `<model-key>_<timestamp>.png`.

For non-interactive use, pass the model and prompt directly:

```bash
python main.py --model roentgen "Chest X-ray of a patient showing an oblique fracture of the 4th rib"
```

CLI flags (`python main.py --help` for the full list):

- `--model` — model key; omit to choose from the interactive menu.
- `--output-dir` — directory for generated images (default: `outputs`).
- `--device` — `auto` (default, prefers MPS), `mps`, `cuda`, or `cpu`.
- `--dtype` — `auto` (default, `float32` on MPS/CPU), `float32`, `float16`, `bfloat16`.
- `--steps` — number of diffusion steps (default: 30).
- `--seed` — optional reproducibility seed.
- `--no-attention-slicing` — disable attention slicing on MPS.

## Models

| Key | Body parts | Modality | Weights |
| --- | --- | --- | --- |
| `medisyn` | Generalist: radiology, pathology, dermatology, ophthalmology, gastroenterology, surgery | CT, X-ray, MRI, ultrasound, endoscopy, microscopy, fundus, OCT, dermoscopy, clinical photos | [`hiesingerlab/MediSyn`](https://huggingface.co/hiesingerlab/MediSyn) UNet on top of `stable-diffusion-v1-5/stable-diffusion-v1-5`, downloaded automatically. License: CC-BY-NC-ND-4.0. |
| `roentgen` | Chest / thorax only | Chest X-ray (frontal radiograph) | [`stanfordmimi/RoentGen-v2`](https://huggingface.co/stanfordmimi/RoentGen-v2), a gated repo. Accept the MIMIC-CXR terms on that page (requires PhysioNet credentialing), then run `hf auth login`. |
| `minim-oct` | Retina (eye) | Optical coherence tomography | See MINIM below. |
| `minim-fundus` | Retina (eye) | Fundus photography | See MINIM below. |
| `minim-chest-xray` | Chest / lungs | Chest X-ray | See MINIM below. |
| `minim-chest-ct` | Chest / lungs | Chest CT (2D slice) | See MINIM below. |
| `minim-brain-mri` | Brain | Brain MRI (2D slice) | See MINIM below. |
| `minim-breast-mri` | Breast | Breast MRI (2D slice) | See MINIM below. |

### MINIM weights

MINIM's fine-tuned UNets are only distributed via Baidu Pan, with no mirror
(see the open GitHub issue #12): https://pan.baidu.com/s/1cxeonEof1oJRMF_Y-rF0UA?pwd=1234.

1. Install the Baidu Netdisk client and log in with a Baidu account.
2. Open the link, enter extraction code `1234`, and download the `RunMINIM` folder.
3. `RunMINIM/model/unets` holds four UNets. According to the maintainer
   (GitHub issue #8), they are 1: OCT, 2: chest X-ray, 3: fundus, 4: MRI + CT.
   Copy each UNet folder (`config.json` + `diffusion_pytorch_model.*`) to
   `weights/minim/oct/`, `weights/minim/chest-xray/`, `weights/minim/fundus/`
   and `weights/minim/mri-ct/`. The `chest-ct`, `brain-mri` and `breast-mri`
   menu entries all use `mri-ct`.

The UNets run on top of the `CompVis/stable-diffusion-v1-4` pipeline, which
is downloaded automatically.

### Adding a model

Create `models/<name>.py` exposing `MODELS = [Model(...)]`, where `Model`
(from `models/common.py`) takes a `key`, `name`, `body_parts`, `modalities`,
and a `generate(prompt, args) -> PIL.Image.Image` function. Then add the
module to the `MODELS` list in `models/__init__.py`.

Only 2D models are in scope. MedImage-Toolkit / MedicalSD (from
`models.md`) were left out on purpose: the GitHub repo's pipeline only
produces simulated noise images, and the MedicalSD Hugging Face org doesn't
exist.

## Setup on an M3 Mac

Use an arm64 Python 3.11 or 3.12 installation, then create and activate a
project-local virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PyTorch extra index in `requirements.txt` is harmless on macOS and helps
Linux users select the CPU wheels. On macOS, PyTorch uses the MPS backend when
the CLI detects it.

All outputs are synthetic research images, not medical diagnoses. Check each
model's license before any use outside research.
