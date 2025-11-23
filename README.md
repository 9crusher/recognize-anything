# Recognize Anything Plus Model (RAM++)

**Open-Set Image Tagging with Multi-Grained Text Supervision**

[[Paper](https://arxiv.org/abs/2310.15200)]

RAM++ is a strong image tagging model that can **recognize any category with high accuracy**, including both **predefined common categories and diverse open-set categories**.

## Highlights

- **Superior Image Recognition Capability**: RAM++ outperforms existing SOTA models on common tags, uncommon tags, and human-object interaction phrases
- **Open-Set Recognition**: Recognize custom categories by providing tag descriptions
- **Zero-Shot Generalization**: Strong performance on unseen categories without fine-tuning

## Installation

### Quick Install

```bash
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

### From Source

```bash
git clone https://github.com/xinyu1205/recognize-anything.git
cd recognize-anything
pip install -e .
```

## Checkpoint

Download the RAM++ checkpoint:

| Model | Backbone | Checkpoint |
|-------|----------|------------|
| RAM++ (14M) | Swin-Large | [Download](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) |

Place the checkpoint in `pretrained/ram_plus_swin_large_14m.pth`.

## Usage

### Open-Set Image Tagging

Recognize custom categories using LLM-generated tag descriptions:

```bash
python inference_ram_plus_openset.py \
  --image path/to/your/image.jpg \
  --pretrained pretrained/ram_plus_swin_large_14m.pth \
  --llm_tag_des path/to/tag_descriptions.json
```

### Custom Tag Descriptions

Tag descriptions should be in JSON format:

```json
[
  {"category_name": "A detailed description of the category..."},
  {"another_category": "Another description..."}
]
```

### Python API

```python
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram_openset as inference
from ram import get_transform
from ram.utils import build_openset_llm_label_embedding
import json

# Load model
model = ram_plus(
    pretrained='pretrained/ram_plus_swin_large_14m.pth',
    image_size=384,
    vit='swin_l'
)

# Load custom tag descriptions
with open('tag_descriptions.json', 'r') as f:
    llm_tag_des = json.load(f)

# Build label embeddings
openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

# Configure model for open-set
model.tag_list = openset_categories
model.label_embed = torch.nn.Parameter(openset_label_embedding.float())
model.num_class = len(openset_categories)
model.class_threshold = torch.ones(model.num_class) * 0.5

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Process image
transform = get_transform(image_size=384)
image = transform(Image.open('your_image.jpg')).unsqueeze(0).to(device)

# Inference
tags = inference(image, model)
print("Detected tags:", tags)
```

## Model Architecture

RAM++ integrates:
- **Swin Transformer** backbone for visual feature extraction
- **BERT-based** query decoder for multi-label tagging
- **CLIP** text embeddings for open-set category understanding
- **Multi-grained text supervision** for improved recognition

## Citation

```bibtex
@article{huang2023open,
  title={Open-Set Image Tagging with Multi-Grained Text Supervision},
  author={Huang, Xinyu and Huang, Yi-Jie and Zhang, Youcai and Tian, Weiwei and Feng, Rui and Zhang, Yuejie and Xie, Yanchun and Li, Yaqian and Zhang, Lei},
  journal={arXiv e-prints},
  pages={arXiv--2310},
  year={2023}
}
```

## License

This project is licensed under the terms found in the LICENSE file.

## Acknowledgements

This work builds upon [BLIP](https://github.com/salesforce/BLIP) and uses components from the Swin Transformer and CLIP models.
