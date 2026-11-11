# Scientist Constraints

## Model

- The submitted system is LLaVA-v1.5-7B (`data/downloads/models/llava-1.5-7b-hf`) with adapted weights (fine-tuning, adapters, preference training, etc.). No other or larger generative model may produce, rewrite or filter the outputs.
- Every prediction is produced by a single greedy decoding pass of the adapted model on the given prompt and image. No inference-time sampling-and-selection, detector filtering, post-editing or retrieval.

## Data

- Training may use the pixels of the images listed in `data/sentinel_train_images.txt` (and more unlabelled COCO train2017 images), the base model's own outputs, and the downloaded open-vocabulary detectors and NLP tools.
- No human annotation of any image may be used for training, tuning or prediction: no MSCOCO instance/caption annotations, no VQAv2 answers, no Visual Genome labels, and no labels of the evaluation images from any source.
- Do not train on the evaluation images or on the development answers in `eval/train/`. Using the development scores to choose between a few settings is allowed.

## Evaluation

- Write `outputs/<method>_predictions.jsonl` with one `{"id", "prediction"}` row for every input of `data/eval_inputs.jsonl`; `prediction` is the raw generated text.
- Do not modify `eval/`, `data/eval_inputs.jsonl` or the frozen baseline. Every metric is recomputed by the evaluator; extra fields are ignored.

## Compliance Checklist

- [ ] Base model is LLaVA-v1.5-7B; outputs come from one greedy pass of the adapted model.
- [ ] No human labels of any image used anywhere.
- [ ] No training on evaluation images.
- [ ] Prediction file covers every input with raw generated text.
- [ ] Coverage and VQA accuracy stay above the failure floors.
