System Architecture Overview
The proposed system implements a four-branch ensemble architecture where each branch captures different aspects of email characteristics. The branches operate in parallel, producing independent feature representations that are concatenated and fed to a meta-learner for final classification:
•	Classical Feature Extraction: Linguistic patterns, URL properties, and email authentication metadata
•	Semantic Embeddings: Sentence-BERT (SBERT) encodings capturing contextual meaning
•	Sequential Analysis: Bidirectional LSTM processing tokenized text sequences
•	Anomaly Detection: Isolation Forest scoring based on deviation from legitimate email patterns
The meta-learner employs XGBoost, a gradient boosted decision tree algorithm, to combine the 403-dimensional feature vector into a final phishing probability score. This architecture enables the system to leverage both learned representations (BERT, LSTM) and domain-specific features (URLs, keywords), while the anomaly detector provides robustness against novel attack patterns.

Dependencies and Environment
Key dependencies include:
•	TensorFlow 2.13+: Deep learning framework for LSTM implementation
•	sentence-transformers 2.2+: SBERT embeddings with automatic model caching
•	XGBoost 2.0+: Gradient boosting with GPU acceleration support
•	scikit-learn 1.3+: Preprocessing, Isolation Forest, evaluation metrics
•	BeautifulSoup4: HTML parsing and text extraction
•	URLExtract 1.8+: URL detection with Unicode support
•	tldextract 3.4+: Domain parsing and TLD extraction

Experimental Results
The system demonstrates balanced performance across both classes with precision exceeding 90% for both legitimate and phishing categories. The overall accuracy of 94.24% is accompanied by a ROC-AUC of 0.9547 and PR-AUC of 0.9234, indicating strong discrimination capability across varying decision thresholds.
The confusion matrix reveals 16 false positives (legitimate emails incorrectly flagged) and 15 false negatives (phishing emails missed). The false positive rate of 4.0% is acceptable for enterprise deployment, where occasional manual review is preferable to security breaches. The false negative rate of 10.9% suggests opportunities for improvement, particularly in detecting sophisticated phishing attempts.
