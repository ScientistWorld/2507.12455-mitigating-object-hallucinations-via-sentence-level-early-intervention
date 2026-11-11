# Problem Definition

## Research Question

How can a multimodal large language model generate image-grounded descriptions with substantially fewer object hallucinations, without losing detail or its general visual question answering ability, and without human annotation or a larger proprietary model to supervise it?

Object hallucination means the model asserts that an object is in the image when it is not. The problem is to reduce such assertions in long, detailed descriptions while keeping the descriptions informative and the model generally capable, at a fixed base-model scale.

## Why It Matters

Hallucinated visual content makes model outputs unreliable wherever users depend on the model to describe or reason about images. Methods that reduce hallucination by making descriptions terse, by adding expensive inference-time machinery, or by damaging general ability do not solve the problem.

## Setting in this environment

- Base model: LLaVA-v1.5-7B (fixed). Methods change how the model is adapted, not which model is used.
- Unlabelled training images: MSCOCO train2017 images listed in `data/sentinel_train_images.txt` (pixels only; no annotations of any image may be used for training).
- Evaluation inputs: MSCOCO val2017 images with eight detailed-description prompts, plus VQAv2 short-answer questions on the same images (`data/eval_inputs.jsonl`). Answers are held by the evaluator.
- Public tools that may be used during training: the downloaded open-vocabulary detectors (GroundingDINO, OWLv2), NLP tooling, and the base model itself.

## Success Criteria

A successful system:

- lowers the share of descriptions that assert any absent object, and the share of asserted objects that are absent;
- keeps describing the objects that are present (coverage);
- keeps its short-answer visual question answering accuracy;
- produces each output with the adapted model's own single greedy decoding pass.
