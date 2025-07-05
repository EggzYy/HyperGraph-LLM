# WORK IN PROGRESS
# Medical Hypergraph Pipeline

A modular, research-grade pipeline for extracting medical entities, context, and relations from clinical or biomedical text, and representing them as a hypergraph for advanced analysis and visualization.

---

## 🚀 Features
- **Chunking**: Efficiently splits large documents for scalable NLP processing.
- **NER & Linking**: Uses SciSpacy and medspaCy for medical entity recognition and UMLS linking.
- **Context & Relations**: Extracts context (negation, temporality, etc.) and sentence-level relations.
- **Hypergraph Construction**: Represents entities and their relations as a hypergraph using HyperNetX.
- **Extensible Rules**: Easily add or modify target/context rules for custom use cases.
- **Visualization**: Tools for visualizing and analyzing the resulting hypergraph.

---

## 🏗️ Project Structure

```
scispacy/
├── src/
│   ├── medical_hypergraph_pipeline.py   # Main pipeline logic
│   ├── rel_normalizer.py                # Relation normalization utilities
│   ├── visualize2.py, visualizegemin.py # Visualization scripts
│   ├── kb_init.py                       # Knowledge base initialization
│   ├── ...
│   ├── resources/                       # Context and section rules
│   ├── results/                         # Output hypergraphs and results
│   └── kb/                              # Knowledge base files
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚡ Quickstart

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download models**
   - Download the `en_core_sci_scibert` model:
     ```bash
     pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
     ```

3. **Run the pipeline**
   ```python
   from src.medical_hypergraph_pipeline import MedicalHypergraphPipeline
   pipeline = MedicalHypergraphPipeline(config={
       "target_rules_path": "src/results/target_rules_prod.json",
       "context_rules_path": "src/resources/context_rules.json",
       # ...other config options...
   })
   H, data = pipeline.process_text("Your medical text here.")
   ```

4. **Visualize or analyze the hypergraph**
   - See `src/visualize2.py` or `src/visualizegemin.py` for examples.

---

## 🧩 Key Components
- **medical_hypergraph_pipeline.py**: Main pipeline class and logic.
- **rel_normalizer.py**: Utilities for normalizing relation outputs.
- **visualize2.py / visualizegemin.py**: Example scripts for visualizing hypergraphs.
- **resources/context_rules.json**: Customizable context rules for medspaCy.
- **results/**: Output directory for hypergraph and results.

---

## 📦 Requirements
See [requirements.txt](requirements.txt) for all dependencies.

---

## 🤝 Contributing
Pull requests, issues, and suggestions are welcome! Please open an issue to discuss your ideas or report bugs.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Maintainer:** [Your Name]  
**Contact:** [your.email@example.com]
