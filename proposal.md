## Project name
Language-Specific Neuron Discovery and Targeted Adaptation of Multilingual LLMs

### Abstract
The goal of this project is to develop and validate an efficient methodology for identifying language-specific neurons in multilingual large language models (LLMs), and to use these neurons for targeted model adaptation with substantially lower training cost than full-parameter finetuning.

Recent multilingual LLMs perform well on high-resource languages, but adaptation to medium and lower-resource languages remains expensive and often causes regressions in cross-lingual behavior. This project addresses that challenge by combining neuron-level interpretability with selective training. Building on our current language-neuron detection pipeline, we will analyze activation statistics across multiple languages, identify language-specific neurons using entropy-based LAPE selection, and evaluate causal effects through controlled ablations and cross-language perplexity matrices. We will then finetune the model by updating only the selected neuron-aligned parameters, enabling efficient specialization while preserving broader multilingual competence.

The main objective is to establish a reproducible, compute-efficient workflow for language-aware adaptation of open LLMs. To fulfill this objective, we define the following tasks:
- Run multilingual tokenization and activation recording for the target language set on large text corpora, collecting layer-wise neuron statistics for MLP and attention components.
- Identify language-specific neurons using LAPE and related filtering strategies, and compare deterministic and random-selection baselines.
- Perform systematic ablation studies and cross-language perplexity evaluation to quantify language specificity, transfer effects, and interference between language groups.
- Finetune 3B-class models with gradient updates restricted to selected neuron subsets, then compare against full-model and broader-mask baselines.
- Evaluate final models on multilingual perplexity and selected downstream tasks to validate quality/cost trade-offs.

The project will lead to the following outcomes:
- Publicly available source code and experiment configurations for the full pipeline: tokenization, activation recording, neuron identification, ablation, and targeted finetuning.
- Publicly available neuron-selection artifacts and language-specific model checkpoints for reproducible follow-up research.
- A practical benchmark protocol for comparing selective-neuron finetuning against standard adaptation approaches.

The project will have the following impacts:
- A resource-efficient adaptation strategy for multilingual LLM development, reducing compute barriers for research groups with limited budgets.
- Better understanding of how language-specific behavior is represented internally in transformer models, supporting safer and more controllable multilingual deployment.
- Reusable methodology for future work on interpretability-guided training and sparse parameter updates.

### Justification for the Utilization of HPC Resources

The requested CPU and GPU hours will support multilingual activation extraction, neuron-selection sweeps, ablation-based evaluation, and targeted finetuning on 1B-3B LLMs.

GPU usage is driven by: (1) activation recording over large token volumes across many languages, (2) repeated cross-language ablation inference runs to build perplexity matrices, and (3) baseline and masked-parameter finetuning experiments. CPU usage is needed for data streaming/preprocessing, experiment orchestration, and metric aggregation.

The workflow also produces sizable intermediate artifacts (token shards, activation tensors, selected-neuron masks, evaluation matrices, checkpoints), requiring sustained storage and I/O throughput.

Without additional HPC allocation, we would need to reduce language coverage, ablation settings, and baseline comparisons, which would weaken statistical reliability and reduce the scientific value and reproducibility of the project outcomes.
