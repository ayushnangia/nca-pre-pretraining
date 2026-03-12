# Running on Vulcan (Alliance Canada, L40S-48GB)

## Before you start

1. **Replace `aip-FIXME`** with your account in ALL `.sh` files:
   ```bash
   sed -i 's/aip-FIXME/aip-yourpi/g' scripts/vulcan/*.sh
   ```

2. **Edit `MODEL_PATH` / `MODEL_FILE`** in each script after the previous phase completes. Check your checkpoint filenames:
   ```bash
   ls $SCRATCH/nca-ppt/checkpoints/*/
   ```

## Workflow

```
┌─────────────────────────┐
│  setup_env.sh (once)    │  ← login node, no sbatch
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 01_nca_prepretraining   │  ← Phase 1: NCA (24h, resubmit with --resume)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 02_preprocess_owt       │  ← CPU job, only if using OpenWebText
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 03_owt_pretraining      │  ← Phase 2: language pre-training
│   OR 03_codeparrot_...  │     (pick OWT or CodeParrot)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 04_ft_gsm8k / 04_ft_bbl │  ← Phase 3: fine-tuning (pick task)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 05_eval_*               │  ← Evaluation
└─────────────────────────┘
```

## Quick start

```bash
# 1. SSH into Vulcan
ssh youruser@vulcan.alliancecan.ca

# 2. Clone repo
cd $HOME
git clone <your-repo-url> nca-pre-pretraining

# 3. Setup environment (one-time, on login node)
bash nca-pre-pretraining/scripts/vulcan/setup_env.sh

# 4. Fix account name
sed -i 's/aip-FIXME/aip-yourpi/g' nca-pre-pretraining/scripts/vulcan/*.sh

# 5. Submit Phase 1
sbatch nca-pre-pretraining/scripts/vulcan/01_nca_prepretraining.sh

# 6. Monitor
sq                    # check queue
tail -f *-nca-ppt.out # watch output
```

## Storage layout

```
$HOME/
├── nca-pre-pretraining/      # code (git repo)
└── nca-ppt-env/              # virtualenv

$SCRATCH/nca-ppt/
├── data/owt/                 # preprocessed .bin files
├── checkpoints/
│   ├── nca_prepretraining/   # Phase 1 checkpoints
│   ├── owt_pretraining/      # Phase 2 checkpoints
│   ├── codeparrot_pretraining/
│   ├── ft_gsm8k/             # Phase 3 checkpoints
│   └── ft_bbl/
└── results/
    ├── eval_gsm8k/           # evaluation output (JSONL)
    ├── eval_humaneval/
    └── eval_bbl/
```

## Tips

- **Resubmitting**: All scripts use `--resume`. Just `sbatch` the same script again.
- **Wall time**: Phase 1 (NCA) may need multiple 24h submissions. It resumes from the latest checkpoint automatically.
- **W&B**: Should work on Vulcan (compute nodes have internet access). If issues arise, add `export WANDB_MODE=offline` and sync later with `wandb sync`.
- **Check GPU usage**: `srun --jobid=JOBID --pty nvidia-smi`
- **Claude Code**: Vulcan compute nodes have internet access, so you can run Claude Code directly on them via `salloc`.
