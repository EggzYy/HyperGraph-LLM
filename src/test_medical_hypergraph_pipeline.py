import json
import unittest
import logging
import spacy
from unittest.mock import Mock, patch
from spacy.tokens import Doc
from medical_hypergraph_pipeline import MedicalHypergraphPipeline


class TestExtractHypergraphAtoms(unittest.TestCase):
    """Test suite for _extract_hypergraph_atoms serialization capabilities."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        # Set up logging to catch any issues
        logging.basicConfig(level=logging.WARNING)

        # Use a simple spacy model for testing
        try:
            cls.nlp = spacy.load("en_core_sci_scibert")
        except OSError:
            # Fallback to basic English model
            cls.nlp = spacy.blank("en")
            cls.nlp.add_pipe("sentencizer")

        # Mock the pipeline class with minimal setup
        cls.pipeline = Mock(spec=MedicalHypergraphPipeline)
        cls.pipeline.nlp = cls.nlp

        # Create the actual method we want to test by binding it to our mock
        cls.pipeline._extract_hypergraph_atoms = (
            MedicalHypergraphPipeline._extract_hypergraph_atoms.__get__(cls.pipeline)
        )
        cls.pipeline._create_entity_instance_atoms = lambda doc, offset=0: ([], {})

    def setUp(self):
        """Set up test-specific resources."""
        # Create a sample document with medical text
        self.sample_text = "Patient denies fever but has a headache. The patient was prescribed medication."
        self.doc = self.nlp(self.sample_text)

        # Add mock context_graph extension if not present
        if not hasattr(self.doc._, "context_graph"):
            Doc.set_extension("context_graph", default=None, force=True)

        # Mock a simple context graph structure
        mock_context_graph = Mock()
        mock_context_graph.targets = []
        mock_context_graph.modifiers = []
        mock_context_graph.edges = []
        self.doc._.context_graph = mock_context_graph

    @patch("medical_hypergraph_pipeline.normalize_relations")
    def test_extract_hypergraph_atoms_serialization(self, mock_normalize_relations):
        """Test that _extract_hypergraph_atoms returns JSON-serializable data."""
        # Mock normalize_relations to return empty list
        mock_normalize_relations.return_value = []

        # Run _extract_hypergraph_atoms on the sample document
        result = self.pipeline._extract_hypergraph_atoms(self.doc)

        # Verify the result structure contains expected keys
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertIn(
            "entity_atoms", result, "Result should contain 'entity_atoms' key"
        )
        self.assertIn(
            "sentence_relation_hints",
            result,
            "Result should contain 'sentence_relation_hints' key",
        )
        self.assertIn(
            "context_graph", result, "Result should contain 'context_graph' key"
        )

        # Verify context_graph has expected structure
        context_graph = result["context_graph"]
        self.assertIsInstance(
            context_graph, dict, "context_graph should be a dictionary"
        )
        self.assertIn(
            "targets", context_graph, "context_graph should contain 'targets' key"
        )
        self.assertIn(
            "modifiers", context_graph, "context_graph should contain 'modifiers' key"
        )
        self.assertIn(
            "edges", context_graph, "context_graph should contain 'edges' key"
        )

        # Test JSON serialization - this is the main test
        try:
            json_output = json.dumps(result)
            self.assertIsInstance(
                json_output, str, "JSON serialization should return a string"
            )

            # Verify the JSON can be parsed back
            parsed_result = json.loads(json_output)
            self.assertEqual(
                parsed_result,
                result,
                "Round-trip JSON serialization should preserve data",
            )

        except TypeError as e:
            self.fail(f"TypeError raised during JSON serialization: {e}")
        except Exception as e:
            self.fail(f"Unexpected error during JSON serialization: {e}")

    @patch("medical_hypergraph_pipeline.normalize_relations")
    def test_extract_hypergraph_atoms_with_offset(self, mock_normalize_relations):
        """Test that _extract_hypergraph_atoms works correctly with character offsets."""
        # Mock normalize_relations to return empty list
        mock_normalize_relations.return_value = []

        # Test with offset parameter
        offset = 10
        result = self.pipeline._extract_hypergraph_atoms(self.doc, offset=offset)

        # Verify the result is still JSON-serializable
        try:
            json_output = json.dumps(result)
            self.assertIsInstance(
                json_output,
                str,
                "JSON serialization with offset should return a string",
            )
        except TypeError as e:
            self.fail(f"TypeError raised during JSON serialization with offset: {e}")

    @patch("medical_hypergraph_pipeline.normalize_relations")
    def test_context_graph_structure(self, mock_normalize_relations):
        """Test that context_graph specifically has JSON-serializable structure."""
        # Mock normalize_relations to return empty list
        mock_normalize_relations.return_value = []

        result = self.pipeline._extract_hypergraph_atoms(self.doc)
        context_graph = result["context_graph"]

        # Test serialization of just the context_graph part
        try:
            json_output = json.dumps(context_graph)
            self.assertIsInstance(
                json_output,
                str,
                "context_graph JSON serialization should return a string",
            )

            # Verify the JSON contains expected keys
            self.assertIn('"targets"', json_output, "JSON should contain targets key")
            self.assertIn(
                '"modifiers"', json_output, "JSON should contain modifiers key"
            )
            self.assertIn('"edges"', json_output, "JSON should contain edges key")

        except TypeError as e:
            self.fail(f"TypeError raised during context_graph JSON serialization: {e}")


if __name__ == "__main__":
    unittest.main()
