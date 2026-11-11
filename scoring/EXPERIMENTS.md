# Experiment Overview

## object_halbench_coco (weight 0.8)

Establishes whether an adapted model describes images in detail while asserting fewer objects that are not there. Detailed descriptions are generated for held-out MSCOCO images under eight different prompt phrasings and compared with human object annotations. Two hallucination rates (per description and per asserted object) measure the benefit; object coverage measures that the descriptions still name the objects that are present, so describing less cannot pass as hallucinating less. Runs whose coverage falls below 0.8x the unaligned model's are scored as failed.

## vqav2_general (weight 0.2)

Establishes that the gain in grounding is not bought with general visual question answering ability. Short-answer VQAv2 questions on the same held-out images are scored with the official VQA accuracy. Runs more than 5 points below the unaligned model are scored as failed.
