# file: main.py
import json
import pathlib
from medical_hypergraph_pipeline import MedicalHypergraphPipeline
import gc

if __name__ == "__main__":
    # --- Configuration ---
    # Centralize all configurable paths and parameters.
    # Most NLP config is now in `config.cfg`.
    CONFIG = {
        "spacy_llm_config_path": "config.cfg",
        "hypergraph_atoms_path": "results/hg_atoms_data_prod.json",
        "hypergraph_gpickle_path": "results/med_ner_hypergraph_prod.gpickle",
        "model_name": "en_core_sci_scibert",
        "target_rules_path": "results/target_rules_prod.json",
        "context_rules_path": "resources/context_rules.json",
        "section_rules_path": "resources/section_rules.json",
        "rush_rules_path": "resources/rush_rules.tsv",
        "linker_config": {
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.7,
        },
    }

    # --- Create results directory ---
    pathlib.Path("results").mkdir(exist_ok=True)

    # --- Load Input Text ---
    input_path = pathlib.Path("input.txt")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path.resolve()}")
    medical_text = input_path.read_text(encoding="utf-8")

    # --- 1. Initialize the Pipeline ---
    print("⇢ Initializing the medical hypergraph pipeline...")
    pipeline = MedicalHypergraphPipeline(CONFIG)

    # --- 2. Process Text and Build Hypergraph ---
    print("\n⇢ Processing text and building hypergraph...")
    H, hypergraph_data = pipeline.process_text(medical_text)

    # --- 3. Report and Save Results ---
    print("\n--- Extraction Summary ---")
    print(f"  • Entity Instances Found: {len(hypergraph_data['entity_atoms'])}")
    print(
        f"  • Hypergraph: {len(H.nodes)} nodes (concepts), {len(H.edges)} hyperedges (contexts)"
    )

    atoms_path = pathlib.Path(CONFIG["hypergraph_atoms_path"])
    with atoms_path.open("w", encoding="utf8") as f:
        json.dump(hypergraph_data, f, indent=2)
    print(f"\n✓ Hypergraph atoms saved to {atoms_path.resolve()}")

    db_path = pathlib.Path(CONFIG["hypergraph_gpickle_path"])
    pipeline.save_hypergraph(H, db_path)

    # --- 4. Demonstrate Persistence ---
    print("\n⇢ Demonstrating persistence...")
    H2 = pipeline.load_hypergraph(db_path)
    if len(H.nodes) == len(H2.nodes) and len(H.edges) == len(H2.edges):
        print("✓ Hypergraph round-trip load successful.")
    else:
        print("✗ Hypergraph round-trip load failed.")

    # --- 5. Example Queries ---
    print("\n--- Example Queries ---")

    section_edges = [
        H.edges[edge_id]
        for edge_id in H.edges
        if H.edges[edge_id].attrs.get("level") == "section"
    ]

    if section_edges:
        # Sort to make the output deterministic
        for section_edge in sorted(section_edges, key=lambda e: e.uid):
            section_name = section_edge.attrs.get("text", "Unknown Section")
            print(f"\n--- Entities in Section: '{section_name}' ---")

            # Get nodes in this hyperedge
            nodes = [H.nodes[node_id] for node_id in section_edge.elements]
            # Sort nodes by their text attribute for deterministic output
            for node in sorted(nodes, key=lambda n: n.attrs.get("text", "")):
                node_attrs = node.attrs
                print(
                    f"  - {node_attrs.get('text')} (Label: {node_attrs.get('label')})"
                )
    else:
        print("\n--- No semantic sections found in the text ---")

    gc.collect()
