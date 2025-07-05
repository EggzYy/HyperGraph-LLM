# file: rel_normalizer.py
"""
Lightweight post-processing component to preserve raw LLM relation outputs.

This component copies the original content of doc._.rel to doc._.rel_raw
(if not already present) so we always preserve what the LLM returned for later debugging.
"""

import logging
from spacy.language import Language
from spacy.tokens import Doc

# Register the custom extension if not already present
if not Doc.has_extension("rel_raw"):
    Doc.set_extension("rel_raw", default=None, force=True)

logger = logging.getLogger(__name__)


@Language.factory("rel_normalizer")
def create_rel_normalizer(nlp, name: str):
    """Factory function to create the rel_normalizer component."""
    return RelNormalizer(nlp, name)


class RelNormalizer:
    """
    Lightweight post-processing component that preserves raw LLM relation outputs.

    This component should be inserted immediately after the llm_rel component
    in the spaCy pipeline. It copies doc._.rel to doc._.rel_raw if not already
    present, ensuring we always preserve what the LLM originally returned.
    """

    def __init__(self, nlp, name: str):
        self.nlp = nlp
        self.name = name
        logger.info(f"Initialized {self.name} component")

    def __call__(self, doc: Doc) -> Doc:
        """
        Process the document and preserve raw relation data.

        Args:
            doc: The spaCy Doc object to process

        Returns:
            The processed Doc object with preserved raw relation data
        """
        # Only copy if rel_raw is not already present and rel exists
        if not hasattr(doc._, "rel_raw") or doc._.rel_raw is None:
            if hasattr(doc._, "rel") and doc._.rel is not None:
                # Make a copy of the raw relation data
                doc._.rel_raw = (
                    doc._.rel.copy() if hasattr(doc._.rel, "copy") else doc._.rel
                )
                logger.debug(
                    f"Preserved raw relation data for document: {len(doc._.rel_raw) if hasattr(doc._.rel_raw, '__len__') else 'N/A'} relations"
                )
            else:
                logger.debug("No relation data found to preserve")
        else:
            logger.debug("Raw relation data already present, skipping preservation")

        return doc
