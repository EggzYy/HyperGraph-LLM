# file: visualizegemin.py
import json
import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, Optional
from collections import defaultdict


class Node:
    """Represents a canonical node (a concept)."""

    def __init__(self, canonical_id, text, label):
        self.canonical_id = canonical_id
        self.text = text
        self.label = label
        self.mentions = []
        self.umls_cuis = set()

    def add_mention(
        self, instance_id, sentence_text, start_char, end_char, local_modifiers
    ):
        self.mentions.append(
            {
                "instance_id": instance_id,
                "sentence_text": sentence_text,
                "start_char": start_char,
                "end_char": end_char,
                "local_modifiers": local_modifiers,
            }
        )

    def add_umls_cui(self, cui):
        if cui:
            self.umls_cuis.add(cui)

    def __repr__(self):
        return (
            f"Node(id='{self.canonical_id}', text='{self.text}', label='{self.label}')"
        )


class Hyperedge:
    """Represents a hyperedge with a specific type."""

    def __init__(self, name, edge_type, nodes=None, text_content=""):
        self.name = name
        self.type = edge_type
        self.nodes = nodes if nodes else set()
        self.text_content = text_content

    def add_node(self, canonical_id):
        self.nodes.add(canonical_id)

    def __repr__(self):
        return f"Hyperedge(name='{self.name}', type='{self.type}', nodes={len(self.nodes)})"


class AtomHypergraph:
    """A robust class to fully rebuild and visualize the hypergraph from atom data."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.hyperedges: Dict[str, Hyperedge] = {}
        self._instance_to_canonical: Dict[str, str] = {}
        self._span_to_canonical: Dict[tuple[int, int], str] = {}

    def get_node(self, canonical_id: str) -> Optional[Node]:
        return self.nodes.get(canonical_id)

    def build_from_json_data(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            print("Error: Input data is not a dictionary.")
            return

        # Pass 1: Create ALL nodes (Entities) from entity_atoms
        for atom in data.get("entity_atoms", []):
            cid = atom.get("canonical_id")
            if cid and cid not in self.nodes:
                self.nodes[cid] = Node(cid, atom["text"], atom["label"])
            if atom.get("instance_id") and atom.get("start_char") is not None:
                self._instance_to_canonical[atom["instance_id"]] = cid
                self._span_to_canonical[(atom["start_char"], atom["end_char"])] = cid

        # Pass 2: Populate entity nodes with mentions and UMLS data
        for atom in data.get("entity_atoms", []):
            node = self.get_node(atom.get("canonical_id"))
            if node:
                node.add_mention(
                    atom["instance_id"],
                    atom["sentence_text"],
                    atom["start_char"],
                    atom["end_char"],
                    atom.get("contextual_modifiers", []),  # Use .get for safety
                )
                for link in atom.get("umls_linking", []):
                    node.add_umls_cui(link.get("cui"))

        # Pass 3: Reconstruct hyperedges from the hypergraph structure saved in the gpickle
        # This part is complex without loading the gpickle. A simpler approach is to
        # rebuild hyperedges based on sentence/paragraph grouping of atoms, similar to the pipeline.

        # Group atoms by sentence
        sentences = defaultdict(set)
        for atom in data.get("entity_atoms", []):
            sentences[atom["sentence_text"]].add(atom["canonical_id"])

        for i, (text, cids) in enumerate(sentences.items()):
            edge_name = f"sent_{i+1}"
            self.hyperedges[edge_name] = Hyperedge(edge_name, "sentence", cids, text)

        # If no sentence edges were created but nodes exist, create one big document edge
        if not self.hyperedges and self.nodes:
            all_node_ids = set(self.nodes.keys())
            self.hyperedges["doc_0"] = Hyperedge(
                "doc_0", "document", all_node_ids, "Full Document"
            )

    def print_summary(self):
        print("Hypergraph Summary:")
        print(f"  Total Nodes Reconstructed: {len(self.nodes)}")
        print(f"  Total Hyperedges Reconstructed: {len(self.hyperedges)}")

    def save_to_json(self, filepath="hypergraph_processed.json"):
        """Saves the structured hypergraph data to a JSON file."""
        output_data = {"nodes": [], "hyperedges": []}
        for node_obj in self.nodes.values():
            node_dict = {
                "id": node_obj.canonical_id,
                "text": node_obj.text,
                "label": node_obj.label,
                "umls_cuis": sorted(list(node_obj.umls_cuis)),
                "mentions": node_obj.mentions,
            }
            output_data["nodes"].append(node_dict)

        for edge_obj in self.hyperedges.values():
            edge_dict = {
                "name": edge_obj.name,
                "type": edge_obj.type,
                "nodes": sorted(list(edge_obj.nodes)),
                "text_content": edge_obj.text_content,
            }
            output_data["hyperedges"].append(edge_dict)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)
            print(f"\nHypergraph successfully saved to '{filepath}'")
        except IOError as e:
            print(f"Error saving hypergraph to '{filepath}': {e}")

    def to_hypernetx_object(self):
        """Converts the internal hypergraph to a HyperNetX.Hypergraph object."""
        edge_dict = {
            edge.name: list(edge.nodes)
            for edge in self.hyperedges.values()
            if edge.nodes
        }
        if not edge_dict:
            return hnx.Hypergraph()

        H = hnx.Hypergraph(edge_dict)
        for nid, node_obj in self.nodes.items():
            if nid in H.nodes:
                H.nodes[nid].properties.update(
                    {
                        "label": node_obj.label,
                        "text": node_obj.text,
                        "mentions": len(node_obj.mentions),
                    }
                )
        for eid, edge_obj in self.hyperedges.items():
            if eid in H.edges:
                H.edges[eid].properties.update({"type": edge_obj.type})
        return H

    def visualize(self, output_filename="results/hypergraph_visualization.png"):
        """Visualizes the full hypergraph with color-coded edge types."""
        H = self.to_hypernetx_object()
        if not H.nodes:
            print("Hypergraph is empty. Skipping visualization.")
            return

        plt.figure(figsize=(24, 20))
        bipartite_graph = H.bipartite()
        pos = nx.spring_layout(bipartite_graph, k=0.3, iterations=50, seed=42)

        node_radii = {
            nid: 0.25 + (props.get("mentions", 0) * 0.05)
            for nid, props in H.nodes.properties.items()
        }
        node_labels = {
            nid: f"{props.get('text', '')}\n({props.get('mentions', 0)})"
            for nid, props in H.nodes.properties.items()
        }
        node_colors = ["skyblue" for _, props in H.nodes.properties.items()]

        edge_color_map = {
            "sentence": "lightgrey",
            "section": "lightblue",
            "document": "lightgreen",
        }
        ordered_edge_ids = list(H.edges)
        edge_facecolors = [
            edge_color_map.get(
                H.edges[eid].properties.get("type", "sentence"), "lightgrey"
            )
            for eid in ordered_edge_ids
        ]

        hnx.draw(
            H,
            pos=pos,
            with_node_labels=True,
            node_labels=node_labels,
            node_radius=node_radii,
            nodes_kwargs={
                "facecolors": node_colors,
                "edgecolors": "black",
                "linewidths": 0.5,
            },
            node_labels_kwargs={"fontsize": 9, "fontweight": "bold"},
            edges_kwargs={
                "facecolors": edge_facecolors,
                "edgecolors": "gray",
                "alpha": 0.7,
            },
        )

        plt.title("Full Medical Hypergraph (Reconstructed from Atoms)", fontsize=18)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"\nFull hypergraph visualization saved to '{output_filename}'")
        plt.close()


def load_json_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file '{filepath}': {e}")
        return None


if __name__ == "__main__":
    json_file_path = "results/hg_atoms_data_prod.json"
    data = load_json_file(json_file_path)

    if data:
        hg = AtomHypergraph()
        hg.build_from_json_data(data)
        hg.print_summary()
        hg.save_to_json("results/hypergraph_output_2.json")
        hg.visualize("results/hypergraph_visualization_2.png")
    else:
        print("Could not load data, hypergraph processing aborted.")
