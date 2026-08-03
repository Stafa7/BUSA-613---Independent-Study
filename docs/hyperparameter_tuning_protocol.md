# Frozen hyperparameter-tuning protocol

This protocol was committed before the archival experiment run. It applies to
the frozen train, validation, and test manifests. Test labels are never used to
select model settings, calibration methods, thresholds, or stopping epochs.

## Traditional text models

TF-IDF Logistic Regression and Linear SVM are selected independently for each
dataset, split design, text representation, and sensitivity cohort. Candidate
models are fit on the training partition and ranked by validation macro F1.
Exact ties use the first candidate in the declared grid, whose order favors the
smaller vocabulary, stronger document-frequency filter, ordinary term
frequency, and stronger regularization.

The frozen grid is:

- maximum TF-IDF features: 10,000 or 20,000;
- word n-grams: 1--2 (fixed to preserve the approved unigram/bigram design);
- minimum document frequency: 5 or 2;
- sublinear term frequency: false or true; and
- regularization `C`: 0.1, 1.0, or 10.0.

This gives 24 candidates per linear model. The selected configuration is refit
on the training partition and evaluated once on the test partition. Every
candidate score and the selected settings are stored with the experiment.

## Transformer comparison

Transformer tuning is deliberately narrow because the datasets are modest and
the three-architecture, three-seed robustness grid is already computationally
large. DistilBERT is the protocol-calibration architecture. Learning rates
`1e-5`, `2e-5`, and `3e-5` are compared using seed 613 on the standard training
and validation partitions of both CoAID and FakeHealth. The rate with the
highest mean validation macro F1 across the two datasets is locked for the full
DistilBERT, ModernBERT, and BiomedBERT comparison. An exact tie uses the first
rate in the declared grid.

All other settings remain common: maximum sequence length 512, batch size 8,
at most three epochs, early-stopping patience one, and seeds 613--615 for the
final comparison. This limited search answers whether the default learning
rate is materially poor without tailoring each architecture to a different
budget or repeatedly mining small validation partitions.

## Interpretation

Hyperparameter tuning reduces avoidable underfitting or overfitting; it does
not repair label disagreement, source shortcuts, domain shift, missing content,
or small effective sample sizes. Robustness splits, artifact masking, transfer,
calibration, grouped uncertainty, and human audits remain necessary.
