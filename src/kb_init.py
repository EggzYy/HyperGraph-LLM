# file: kb_init.py
# Purpose: Creates an empty Knowledge Base required by the spacy-llm EntityLinker.
# Run this script once before running the main pipeline.
from spacy.kb import KnowledgeBase, InMemoryLookupKB
from pathlib import Path
import warnings
from spacy.vocab import Vocab
from spacy.lang.en import English


class FullyImplementedKB(KnowledgeBase):
    def __init__(self, vocab: Vocab, entity_vector_length: int):
        super().__init__(vocab, entity_vector_length)
        ...


def create_empty_kb(vocab, output_dir):
    """Creates and saves an empty knowledge base."""
    # Suppress UserWarning about empty vocab
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # vocab = nlp.vocab
        kb = InMemoryLookupKB(vocab=vocab, entity_vector_length=64)

    # This is an empty KB. To populate it, you would use kb.add_entity() and kb.add_alias()
    # For now, it serves as a placeholder for the linker component to initialize correctly.
    kb_path = output_dir / "kb"
    kb.to_disk(kb_path)
    print(f"✔ Empty KnowledgeBase created at: {kb_path.resolve()}")


if __name__ == "__main__":
    output_dir = Path("./")
    output_dir.mkdir(exist_ok=True)

    # We need a vocab to initialize the KB. A blank model is sufficient.
    nlp = English()

    create_empty_kb(nlp.vocab, output_dir)
    print("\nScript finished. You can now run the main pipeline.")
