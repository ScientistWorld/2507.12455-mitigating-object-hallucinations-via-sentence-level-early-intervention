"""Reference configuration of the reproduced SENTINEL run (LLaVA-v1.5-7B).

Values marked [paper] follow the paper (Sec. 4.1, Table 10).  Values marked
[scaled] are reduced to fit the compute budget; see PROGRESS.md.
"""

# In-domain preference data construction (Sec. 3.2-3.3, Algorithm 1)
TRAIN_IMAGES_LIST = "data/sentinel_train_images.txt"  # unlabelled COCO train2017 images
NUM_IMAGES = 4000             # image pool; construction stops at the time budget or MAX_PAIRS
MAX_PAIRS = 8600              # [paper] 8.6K preference pairs for LLaVA-v1.5-7B
SAMPLES_PER_STEP = 8          # n in-domain candidates sampled per context
SAMPLE_TEMPERATURE = 1.0      # plain sampling from the model distribution
MAX_SENTENCE_TOKENS = 48      # generation halts at sentence end ('.') or </s>
MAX_ICB_STEPS = 10            # bootstrapping continues until </s>, capped
BATCH_IMAGES = 10             # images processed together per sampling call

# Context-aware preference learning (Sec. 3.4, Table 10)
LORA_R = 128                  # [paper]
LORA_ALPHA = 256              # [paper]
LORA_DROPOUT = 0.05           # LLaVA LoRA default
LEARNING_RATE = 2e-6          # [paper] Table 10 (7B)
WEIGHT_DECAY = 0.0            # [paper]
EPOCHS = 1                    # [paper] one pass over 8.6K pairs ...
OPT_STEPS = 134               # ... = 8600 / 64 optimizer updates; with fewer pairs built
                              # [scaled], training keeps the paper's update count by
                              # cycling over the pairs (see PROGRESS.md)
GLOBAL_BATCH = 64             # [paper]
MICRO_BATCH = 2               # pairs per forward pass (gradient accumulation)
DPO_BETA = 0.1                # [paper]
WARMUP_RATIO = 0.03           # LLaVA training default; cosine schedule [paper]
SEED = 42
