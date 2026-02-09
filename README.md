# Language Neuron Detection

This repository explores language-specific neurons in LLMs. It is organized around three steps:

1. Tokenize a text dataset.
2. Record neuron activations on the tokenized dataset, separately for each language to identify language-specific neurons.
3. Finetune the LLM on downstream tasks and test interventions using the identified neurons. This step is not implemented yet and currently contains only a demo example.

## Setup

Use a simple pip setup with the requirements file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

- Step 3 is a placeholder and will be implemented later.
