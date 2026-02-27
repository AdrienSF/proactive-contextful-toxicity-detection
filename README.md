# Context-Aware Toxicity Detection in Multiplayer Games: Integrating Domain-Adaptive Pretraining and Match Metadata
Metrics for all experiments and hyperparameters can be viewed in `experiments/vis.ipynb`.

## Repository Structure

```
├── data/
│   ├── platform_guidelines.json        # Toxicity / harassment policy excerpts from 11 platforms
│   │                                     (Xbox, PSN, Riot, Steam, YouTube, Twitch, Discord, etc.)
│   ├── dota2/
│   │   └── labeled/
│   │       ├── train_labels.csv            # Primary labeled training set (~3.4k messages)
│   │       ├── test_labels.csv             # Held-out labeled test set (~2.7k messages)
│   │       └── devinanzelmo_train_labels.csv  # Additional labeled train data (~2.7k messages)
│   └── metrics/
│       ├── dota2_metrics.jsonl         # Experiment metrics for Dota 2 chat
│       └── cod_metrics.jsonl           # Experiment metrics for Call of Duty chat
│
├── experiments/
│   ├── pretrain.py                     # Domain-adaptive MLM pretraining script
│   ├── finetune.py                     # Supervised toxicity classification finetuning script
│   ├── eda.ipynb                       # Exploratory data analysis notebook
│   ├── vis.ipynb                       # Experiment metrics visualisation & comparison notebook
│   ├── dota2_metrics.jsonl             # Copy/snapshot of Dota 2 experiment metrics
│   ├── cod_metrics.jsonl               # Copy/snapshot of CoD experiment metrics
│   └── custom/
│       └── datasets.py                 # PyTorch datasets & dialog representation utilities
│
├── requirements.txt                    # Python dependencies
└── README.md
```

### `data/`

| File | Description |
|------|-------------|
| `platform_guidelines.json` | Excerpts of toxicity and harassment policies from 11 gaming and social platforms, used as contextual reference. |
| `dota2/labeled/*.csv` | Labeled Dota 2 match chat data. Each row contains a chat message, player slot, match ID, and a toxicity label. |
| `metrics/*.jsonl` | Per-epoch experiment metrics (accuracy, balanced accuracy, precision, recall, F1, AUC, train/test loss) along with full hyperparameter metadata. |

### `experiments/`

| File | Description |
|------|-------------|
| `pretrain.py` | Runs masked language modelling on unlabeled match chat using Hugging Face `Trainer`. Adds custom special tokens (`[PlayerN]`, `[TeamN]`, `<msep>`) and saves checkpoints for downstream finetuning. |
| `finetune.py` | Finetuning loop with weighted cross-entropy loss, stratified batching, cosine-annealing LR schedule, and a hyperparameter search grid over base models, dialog representations, and learning rates. Logs per-epoch JSONL metrics. |
| `custom/datasets.py` | Custom PyTorch `Dataset` classes for streaming match chat as dialog context. Implements multiple dialog representation strategies (`add_msep`, `add_team_player_tokens`, `point_seperate_messages`, current-player-only variants) and label shifting for proactive toxicity prediction. |
| `eda.ipynb` | Exploratory data analysis of the chat datasets. |
| `vis.ipynb` | Visualisation of all experiment metrics and hyperparameters. |

## Setup

```bash
pip install -r requirements.txt
```

### Key Dependencies

`torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `jsonlines`, `matplotlib`, `seaborn`, `nltk`
