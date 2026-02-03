# FineTuningGPTmodels

This repository contains small, script-based examples for:

- Preprocessing the CNN/DailyMail summarization dataset.
- Fine-tuning a `t5-small` model for summarization.
- Running inference from a fine-tuned checkpoint.
- Using a fine-tuned checkpoint inside a LangChain summarization flow.
- Making a simple OpenAI Chat Completions API request.

All scripts live in `scripts/` and can be run independently.

## Run on Google Colab

### 1) Create a new Colab notebook

Open <https://colab.research.google.com/> and create a new notebook.

### 2) Clone this repository

Run this in a Colab cell:

```bash
!git clone <YOUR_REPO_URL>
%cd FineTuningGPTmodels
```

Replace `<YOUR_REPO_URL>` with the URL of your GitHub repository.

### 3) Install dependencies

```bash
!pip -q install datasets transformers accelerate torch sentencepiece langchain python-dotenv requests
```

### 4) (Optional) Download the dataset as CSV

```bash
!python scripts/preprocess.py
```

This downloads CNN/DailyMail and saves train/validation/test CSVs into a `data/` folder.

### 5) Fine-tune the summarization model

```bash
!python scripts/train.py
```

This fine-tunes `t5-small` on a small subset of CNN/DailyMail and stores checkpoints under
`models/results`.

### 6) Run inference with the fine-tuned checkpoint

```bash
!python scripts/huggingface_model.py
```

> Note: `scripts/huggingface_model.py` uses `models/results/checkpoint-89` by default. If your
> training run produces a different checkpoint, update that path in the script.

### 7) Run the LangChain summarizer

```bash
!python scripts/langchain_summarizer.py
```

This loads the fine-tuned checkpoint and runs a sample summarization prompt.

### 8) (Optional) OpenAI API example

To run the OpenAI script, add your key to the environment:

```bash
import os
os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"
```

Then run:

```bash
!python scripts/openai_model.py
```

## Scripts Overview

- `scripts/preprocess.py`: Downloads the CNN/DailyMail dataset and exports CSVs.
- `scripts/train.py`: Fine-tunes `t5-small` for summarization.
- `scripts/huggingface_model.py`: Runs inference from a fine-tuned checkpoint.
- `scripts/langchain_summarizer.py`: Demonstrates LangChain usage with a fine-tuned model.
- `scripts/openai_model.py`: Example OpenAI Chat Completions API call.
