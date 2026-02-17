## Training Language Models - PyTorch

This repository provides an implementation of the paper, [Training Language Models on Neural Cellular Automata]() 

## Content
```
.
├── src/
|   ├── nca_ppt.py              - NCA pre-pretraining script
|   ├── language_train.py       - Language model pre-training (HF datasets)
|   ├── openwebtext_pt.py       - OpenWebText pre-training
|   ├── datasets/
|   |   └── preprocess.py       - Dataset tokenization & preprocessing
|   └── eval/
|       ├── bigbench.py         - BIG-Bench evaluation
|       ├── humaneval.py        - HumanEval code generation eval
|       └── gsm8k.py            - GSM8K math reasoning eval
├── utils/
|   ├── nca.py                  - NCA model definitions (Flax/JAX)
|   ├── models.py               - Llama-based language model definitions
|   ├── dataset_utils.py        - Dataset loading & batching utilities
|   ├── tokenizers.py           - Tokenizer wrappers (tiktoken, JAX)
|   ├── training_args.py        - Shared training argument dataclasses
|   └── util.py                 - General helpers (seeding, logging, checkpointing)
├── scripts/
|   ├── nca_pretraining.sh      - launch for NCA pre-pretraining
|   ├── owt_ft.sh               - launch for OWT fine-tuning
|   └── codeparrot_ft.sh        - launch for CodeParrot fine-tuning
├── docs/                       - Internal dev notes & audit logs
├── README.md
├── requirements.txt            - pip dependencies
└── environment.yml             - Conda environment spec
```

## Usage

Before running any code please make sure to activate your environment. We provide examples of NCA Pre-pretraining:

```bash
scripts/nca_pretraining.sh
```

And Language Pre-Training:

```bash
scripts/owt_ft.sh
scripts/codeparrot_ft.sh
```

## Citations

```bibtex
@misc{leehan2026traininglmnca,
    title   = {Training Language Models via Neural Cellular Automata}, 
    author  = {Dan Lee and Seungwook Han and Akarsh Kumar and Pulkit Agrawal},
    year    = {2026},
    eprint  = {},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {}, 
}
```