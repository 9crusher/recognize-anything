#!/bin/bash
docker run --rm -v $(pwd)/datasets:/data ram-plus python /app/inference_ram_plus_openset.py --image /data/images/1.jpg --pretrained /models/ram_plus_swin_large_14m.pth --llm_tag_des /data/tag_defs/openimages_rare_200_llm_tag_descriptions.json
