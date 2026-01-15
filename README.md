# AI-RESUME-PARSER-JOB-RECOMMEDER


This repository implements a Resume (CV) Parsing pipeline using spaCy 3 and Named Entity Recognition (NER). The goal is to convert unstructured resume text into structured fields (name, email, phone, degree, university, skills, designation, experience, etc.) by training a custom NER model.

This README summarizes the approach demonstrated in a companion tutorial video and documents the main steps, code workflow, and commands to reproduce the training and evaluation.

Status: ongoing
---

## Overview

Resume parsing extracts structured information from free-text resumes by training a custom Named Entity Recognition (NER) model. This project uses spaCy 3 and the `spacy-transformers` integration (e.g., Hugging Face BERT or other transformer backbones) to leverage contextual embeddings for improved NER performance.

The example dataset contains ~200 annotated resumes (JSON) used for training and validation in the tutorial.

---

## Key Technologies

- Python 3.8+
- spaCy 3.x
- spacy-transformers
- PyTorch (GPU recommended)
- scikit-learn (for train/test split)
- Annotation tools (for dataset creation): Doccano, Label Studio
- Hugging Face transformer models (e.g., `bert-base-uncased`)

---

## Dataset & Annotation

- The training data is provided as JSON (annotated resumes). In the tutorial, the dataset contains 200 resumes.
- Annotation fields typically include:
  - NAME, EMAIL, PHONE, SKILL, DEGREE, UNIVERSITY, DESIGNATION, EXPERIENCE, ORGANIZATION, DATE, etc.
- Tools used for annotation:
  - Doccano
  - Label Studio

Annotation best practices:
- Use consistent label names.
- Avoid overlapping entity spans where possible; overlapping annotations are handled (skipped) during conversion if they cannot be mapped to non-overlapping spans.
- Keep a log of problematic spans for later correction.

---

## Project Structure (high level)

Example files and artifacts you should see or create:

- `train_data.json` — annotated resumes (original JSON)
- `preprocess.py` — script that converts JSON -> spaCy `Doc` objects -> `DocBin` (`train_data.spacy`, `test_data.spacy`)
- `config.cfg` — spaCy training configuration (generated from a base config and tuned)
- `train_data.spacy` — binary training data
- `test_data.spacy` — binary dev/validation data
- `output/` — training outputs (contains `model-best`, `model-last`)
- `error.txt` — logged problematic/invalid spans during conversion

---

## Setup & Installation

1. Clone the repo:
   ```
   git clone https://github.com/Deb2003-21/AI-RESUME-PARSER-JOB-RECOMMEDER.git
   cd AI-RESUME-PARSER-JOB-RECOMMEDER
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate        # Linux / macOS
   .venv\Scripts\activate           # Windows (PowerShell)
   pip install -U pip
   pip install -r requirements.txt  # if provided
   ```

   Or install the main packages directly:
   ```
   pip install -U spacy spacy-transformers torch scikit-learn
   ```

3. Verify GPU availability (optional, recommended if training with transformer backbone):
   - For PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
   - For spaCy training: pass `--gpu-id` (e.g., `--gpu-id 0`)

---

## Preparing the spaCy Config

1. Obtain a base config from spaCy website (choose pipeline components: transformer, NER, system settings like GPU).
2. Fill and customize the base config to create `config.cfg`:
   ```
   python -m spacy init fill-config base_config.cfg config.cfg
   ```
3. Edit `config.cfg`:
   - Set paths for training and dev data (or pass them via CLI)
   - Configure transformer model (e.g., `"model": {"name": "bert-base-uncased"}` under the transformer component)
   - Adjust training hyperparameters:
     - `training.max_steps` (e.g., larger than tutorial demo; tutorial used a low value for quick demo)
     - `training.eval_frequency`
     - `training.dropout`
     - Batch sizes and optimizer settings
   - If using GPU, confirm the `pytorch` section is configured as needed.

---

## Preparing Training Data (JSON -> DocBin)

The core preprocessing steps for preparing spaCy binary training files:

1. Load the annotated `train_data.json`:
   - Each training example should contain:
     - `text` (resume text)
     - `annotations` (list of entity spans: start, end, label)

2. Convert JSON examples to spaCy `Doc` objects:
   - Create a spaCy `nlp` object using the same tokenizer as your pipeline.
   - For each example:
     - Use `doc.char_span(start, end, label=...)` to create entity spans.
     - Handle overlapping or invalid spans:
       - If `doc.char_span` returns `None` (invalid/unaligned), log the issue to `error.txt`.
       - If entities overlap and cannot be represented, skip or resolve them (the tutorial skipped problematic overlapping spans).
     - Append validated spans to `doc.ents`.
   - Add `Doc` to `DocBin`.

3. Example split and saving:
   ```
   from sklearn.model_selection import train_test_split
   train, test = train_test_split(all_examples, test_size=0.30, random_state=42)
   # Convert and save as:
   docbin_train.to_disk("train_data.spacy")
   docbin_test.to_disk("test_data.spacy")
   ```

Important notes:
- Use the same tokenizer / whitespace rules as the eventual training pipeline.
- Keep a log of errors; fix annotations if many spans fail.

---

## Train / Evaluate

Once `config.cfg`, `train_data.spacy`, and `test_data.spacy` are ready:

1. Training (GPU example):
   ```
   python -m spacy train config.cfg --output ./output --paths.train ./train_data.spacy --paths.dev ./test_data.spacy --gpu-id 0
   ```

   Or set CUDA visible devices and run:
   ```
   CUDA_VISIBLE_DEVICES=0 python -m spacy train config.cfg --output ./output --paths.train ./train_data.spacy --paths.dev ./test_data.spacy
   ```

2. Outputs:
   - `./output/model-best` — best checkpoint (based on evaluation metric)
   - `./output/model-last` — last checkpoint after training

3. Evaluate / test:
   - Use spaCy's `evaluate` or run custom inference using the saved model to compute precision/recall/F1 for each label.

---

## Important Implementation Notes (from the tutorial)

- get_spacy_doc function (conceptual summary):
  - Iterates through JSON text + annotations.
  - For each annotation, uses `doc.char_span()` to convert character indices to spans.
  - Skips overlapping entities that can't be converted cleanly.
  - Writes errors (spans that could not be converted) to `error.txt`.
  - Adds valid spans to `doc.ents` and appends the doc to a `DocBin`.

- Overlapping Entities:
  - The tutorial demonstrates skipping overlapping annotations that cannot be represented as non-overlapping `doc.ents`. Prefer to fix annotations to avoid many skips.

- Dataset split:
  - Example split used in tutorial: 70% training (≈140 resumes) / 30% testing (≈60 resumes).

- Training steps:
  - The tutorial set `max_steps` low for demo; increase it appropriately for real training.
  - Monitor evaluation metrics and use `model-best` as the saved best checkpoint.

---

## Tips & Next Steps

- Improve data quality:
  - More annotated resumes → better generalization.
  - Standardize entity labeling and resolve overlaps.

- Model choices:
  - Use stronger transformer backbones for better NER (but requires more GPU).
  - Consider domain-adaptive pretraining if resumes contain domain-specific jargon.

- Hyperparameter tuning:
  - Experiment with `max_steps`, learning rate, dropout, and batch sizes.
  - Use `evaluation_frequency` to observe model performance during training.

- Production / Inference:
  - Wrap the `model-best` into a lightweight inference API.
  - Post-process extracted entities (e.g., normalize phone/email, map degree names).

- Debugging:
  - Inspect `error.txt` for problematic spans to improve annotation alignment.
  - Visualize predictions vs. ground truth using spaCy's `displacy` or small evaluation scripts.

---

## References & Acknowledgements

- spaCy 3 documentation: https://spacy.io/usage
- spaCy Transformers integration: https://github.com/explosion/spacy-transformers
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Annotation tools: Doccano, Label Studio

This README was created based on a tutorial walkthrough covering:
- setup and data loading
- spaCy config generation and customization
- JSON -> DocBin conversion with `doc.char_span`
- training command and outputs
- handling overlapping entities and logging errors

---

## License

Specify a license for this repository (e.g., MIT). Update this section to reflect the desired license.

---

If you'd like, I can:
- produce a minimal `requirements.txt` and `preprocess.py` template that implements the JSON -> DocBin conversion (with the `get_spacy_doc` logic explained above),
- or update the README with explicit, repo-specific paths and commands based on your current file layout.
```
