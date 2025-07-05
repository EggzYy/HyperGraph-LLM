# file: medical_hypergraph_pipeline.py
from __future__ import annotations

import itertools
import json
import logging
import os
import pathlib
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hypernetx.classes.hypergraph as hnx
import networkx as nx
import spacy
import spacy_llm
from medspacy.context.context_rule import ConTextRule
from medspacy.section_detection import SectionRule
from medspacy.target_matcher import TargetRule
from semantic_chunker import get_chunker
from spacy.language import Language
from spacy.tokens import Doc, Span
import gc

gc.collect()
# Import custom components
import rel_normalizer  # Import to register the factory
from scispacy.linking import EntityLinker
# Ensure custom attributes are registered globally before use
for _attr in (
    "temporality",
    "experiencer",
    "section_category",
    "is_negated",
    "modifiers",
    "target_rule",
    "context_rule",
    "context_graph",
):
    if not Span.has_extension(_attr):
        Span.set_extension(_attr, default=None, force=True)
    if not Doc.has_extension(_attr):
        Doc.set_extension(_attr, default=None, force=True)

if not Span.has_extension("umls_ents"):
    Span.set_extension("umls_ents", getter=lambda span: span._.kb_ents, force=True)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s"
)
spacy_llm.logger.addHandler(logging.StreamHandler())
spacy_llm.logger.setLevel(logging.DEBUG)

# Relation processing metrics and configuration
REL_DEBUG = os.getenv("REL_DEBUG", "0").lower() in ("1", "true", "yes", "on")
rel_logger = logging.getLogger(__name__ + ".relations")
if REL_DEBUG:
    rel_logger.setLevel(logging.DEBUG)
else:
    rel_logger.setLevel(logging.INFO)


# Global metrics for relation processing
class RelationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_relations = 0
        self.successfully_parsed = 0
        self.rejected_relations = 0
        self.bad_examples = []  # List of (relation, exception_message) tuples
        self.max_bad_examples = 10  # Configurable number of bad examples to store

    def add_successful(self):
        self.successfully_parsed += 1

    def add_rejected(self, relation_data: Any, exception_msg: str):
        self.rejected_relations += 1
        if len(self.bad_examples) < self.max_bad_examples:
            self.bad_examples.append((relation_data, exception_msg))

    def log_summary(self):
        rel_logger.info("Relation Processing Summary:")
        rel_logger.info(f"  Total relations processed: {self.total_relations}")
        rel_logger.info(f"  Successfully parsed: {self.successfully_parsed}")
        rel_logger.info(f"  Rejected relations: {self.rejected_relations}")
        rel_logger.info(
            f"  Success rate: {(self.successfully_parsed/self.total_relations*100):.1f}%"
            if self.total_relations > 0
            else "  Success rate: N/A"
        )

        if self.bad_examples:
            rel_logger.info(f"  First {len(self.bad_examples)} bad examples:")
            for i, (relation, error) in enumerate(self.bad_examples, 1):
                rel_logger.info(f"    {i}. {relation!r} -> {error}")


# Global metrics instance
rel_metrics = RelationMetrics()


def get_relation_metrics_summary() -> Dict[str, Any]:
    """Get a dictionary summary of relation processing metrics."""
    return {
        "total_relations": rel_metrics.total_relations,
        "successfully_parsed": rel_metrics.successfully_parsed,
        "rejected_relations": rel_metrics.rejected_relations,
        "success_rate": (
            (rel_metrics.successfully_parsed / rel_metrics.total_relations * 100)
            if rel_metrics.total_relations > 0
            else 0.0
        ),
        "bad_examples": rel_metrics.bad_examples.copy(),
    }


def log_final_relation_metrics():
    """Log the final comprehensive summary of relation processing metrics."""
    rel_metrics.log_summary()


def reset_relation_metrics():
    """Reset the global relation metrics."""
    rel_metrics.reset()
    rel_logger.info("Relation processing metrics have been reset")


def configure_relation_metrics(max_bad_examples: int = 10):
    """Configure the relation metrics settings.

    Args:
        max_bad_examples: Maximum number of bad examples to store for logging
    """
    rel_metrics.max_bad_examples = max_bad_examples
    rel_logger.info(f"Configured relation metrics: max_bad_examples={max_bad_examples}")


def _serialize_span(span: Span) -> dict:
    """Serialize a spaCy Span to a dictionary.

    Args:
        span: The spaCy Span object to serialize

    Returns:
        Dictionary containing span text, character positions, and label
    """
    return {
        "text": span.text,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "label": span.label_ or None,
    }


def _serialize_context_object(obj) -> dict:
    """Serialize a medspaCy context object (ConTextModifier, ConTextTarget, etc.) to a dictionary.

    Args:
        obj: The medspaCy context object to serialize

    Returns:
        Dictionary containing the object's relevant information
    """
    if obj is None:
        return None

    # Handle ConTextModifier objects
    if hasattr(obj, "modifier_span") and hasattr(obj, "category"):
        return {
            "type": "modifier",
            "category": getattr(obj, "category", None),
            "span_indices": getattr(obj, "modifier_span", None),
            "direction": getattr(obj, "direction", None),
        }

    # Handle ConTextTarget objects
    if hasattr(obj, "target_span"):
        return {"type": "target", "span_indices": getattr(obj, "target_span", None)}

    # Handle Span objects
    if hasattr(obj, "start_char") and hasattr(obj, "end_char") and hasattr(obj, "text"):
        return _serialize_span(obj)

    # Fallback: try to convert to string or return basic representation
    try:
        return {"type": "unknown", "value": str(obj)}
    except Exception:
        return {"type": "unknown", "value": repr(obj)}


def _get_incidence_graph(H: hnx.Hypergraph) -> nx.Graph:
    """Helper to get the incidence graph from a HyperNetX object, supporting multiple versions."""
    if hasattr(H, "incidence_graph"):
        return H.incidence_graph
    if callable(getattr(H, "bipartite", None)):
        return H.bipartite()
    raise RuntimeError(
        "Unsupported HyperNetX version: no incidence graph accessor found."
    )


PIPELINE_INSTANCE_CONTEXT = {"current": None}


@Language.factory("dynamic_target_rules_component")
def create_dynamic_target_rules_component(nlp, name: str):
    """
    This factory creates the DynamicTargetRulesComponent. It retrieves the parent
    pipeline instance from a context and injects its specific dependencies into the component.
    """
    pipeline_instance = PIPELINE_INSTANCE_CONTEXT["current"]
    if pipeline_instance is None:
        raise ValueError("The pipeline instance context was not set correctly.")

    return DynamicTargetRulesComponent(
        rule_cache=pipeline_instance.rule_cache,
        key_for_rule_func=pipeline_instance._key_for_rule,
        nlp_instance=nlp,
    )


@Language.component("debug_context_graph")
def debug_context_graph(doc):
    # Warn if any modifier has zero edges
    if doc._.context_graph:
        lonely = [
            m
            for m in doc._.context_graph.modifiers
            if not any(m is e[1] for e in doc._.context_graph.edges)
        ]
        if lonely:
            logging.warning(
                f"{len(lonely)} modifier(s) still have no targets: "
                f"{[doc[s:e+1].text for m in lonely for s,e in [m.modifier_span]]}"
            )
    return doc


def normalize_relations(doc) -> List[Dict[str, Any]]:
    """
    Normalize doc._.rel_raw or doc._.rel into canonical list[dict] format for spaCy-LLM >= 0.7.

    Expected output format:
    [{"dep": "ENT0", "dest": "ENT1", "relation": "CAUSE"}, ...]

    Conversion rules:
    1. If element is dict – keep as is
    2. If element is str and json.loads succeeds – load it
    3. If element is pseudo-Python dict string ('{a:b, c:d}') – patch to JSON:
       • Add quotes around keys/values with regex
       • Replace single quotes with double quotes
       • Then json.loads
    4. Skip and log any element that still can't be parsed

    Args:
        doc: spaCy Doc object with ._.rel_raw or ._.rel attribute

    Returns:
        List of normalized relation dictionaries
    """
    # Reset metrics for this document processing
    doc_metrics = RelationMetrics()

    # Try rel_raw first, then fall back to rel
    raw_relations = None
    if hasattr(doc._, "rel_raw") and doc._.rel_raw:
        raw_relations = doc._.rel_raw
        rel_logger.debug("Using doc._.rel_raw for normalization")
    elif hasattr(doc._, "rel") and doc._.rel:
        raw_relations = doc._.rel
        rel_logger.debug("Using doc._.rel for normalization")

    if not raw_relations:
        rel_logger.debug("No rel_raw or rel attribute found or they're empty")
        return []

    rel_logger.debug(
        f"Starting normalization of {len(raw_relations)} raw relation elements"
    )
    normalized_relations = []

    for i, element in enumerate(raw_relations):
        doc_metrics.total_relations += 1

        try:
            # Rule 0: If element is a RelationItem (namedtuple), convert to dict
            if hasattr(element, "_asdict") or type(element).__name__ == "RelationItem":
                if hasattr(element, "_asdict"):
                    # namedtuple-like object
                    element_dict = element._asdict()
                else:
                    # fallback for RelationItem-like objects
                    element_dict = {
                        "dep": str(element.dep) if hasattr(element, "dep") else None,
                        "dest": str(element.dest) if hasattr(element, "dest") else None,
                        "relation": (
                            str(element.relation)
                            if hasattr(element, "relation")
                            else None
                        ),
                    }
                normalized_relations.append(element_dict)
                doc_metrics.add_successful()
                if REL_DEBUG:
                    rel_logger.debug(
                        f"Element {i}: RelationItem converted to dict: {element_dict}"
                    )
                continue

            # Rule 1: If element is already a dict, keep as is
            if isinstance(element, dict):
                normalized_relations.append(element)
                doc_metrics.add_successful()
                if REL_DEBUG:
                    rel_logger.debug(
                        f"Element {i}: Already a dict, keeping as-is: {element}"
                    )
                continue

            # Rule 2: If element is a string, try json.loads
            if isinstance(element, str):
                element = element.strip()
                if not element:
                    if REL_DEBUG:
                        rel_logger.debug(f"Element {i}: Empty string, skipping")
                    continue

                try:
                    # Try direct JSON parsing first
                    if REL_DEBUG:
                        rel_logger.debug(
                            f"Element {i}: Attempting direct JSON parsing for: {element!r}"
                        )
                    parsed = json.loads(element)
                    if isinstance(parsed, dict):
                        normalized_relations.append(parsed)
                        doc_metrics.add_successful()
                        if REL_DEBUG:
                            rel_logger.debug(
                                f"Element {i}: Successfully parsed with direct JSON"
                            )
                        continue
                except json.JSONDecodeError as e:
                    if REL_DEBUG:
                        rel_logger.debug(
                            f"Element {i}: Direct JSON parsing failed: {e}"
                        )
                    pass

                # Rule 3: Try to fix pseudo-Python dict strings
                try:
                    if REL_DEBUG:
                        rel_logger.debug(
                            f"Element {i}: Attempting regex-based auto-fix for: {element!r}"
                        )
                    # Fix pseudo-Python dict format: {a:b, c:d} -> {"a":"b", "c":"d"}
                    fixed_element = _fix_pseudo_python_dict(element)
                    if REL_DEBUG:
                        rel_logger.debug(
                            f"Element {i}: Auto-fixed to: {fixed_element!r}"
                        )
                    parsed = json.loads(fixed_element)
                    if isinstance(parsed, dict):
                        normalized_relations.append(parsed)
                        doc_metrics.add_successful()
                        if REL_DEBUG:
                            rel_logger.debug(
                                f"Element {i}: Successfully parsed with regex auto-fix"
                            )
                        continue
                except (json.JSONDecodeError, Exception) as e:
                    if REL_DEBUG:
                        rel_logger.debug(f"Element {i}: Regex auto-fix failed: {e}")
                    pass

            # Rule 4: Skip and log unparseable elements
            error_msg = (
                "Unable to parse as dict, JSON, or auto-fix pseudo-Python format"
            )
            doc_metrics.add_rejected(element, error_msg)
            rel_logger.warning(
                f"Element {i}: Skipping unparseable relation element: {element!r}"
            )

        except Exception as e:
            error_msg = f"Unexpected error during processing: {e}"
            doc_metrics.add_rejected(element, error_msg)
            rel_logger.warning(
                f"Element {i}: Error processing relation element: {element!r}. Error: {e}"
            )

    # Log summary for this document
    rel_logger.info(
        f"Document normalization complete: {doc_metrics.successfully_parsed}/{doc_metrics.total_relations} relations successfully normalized"
    )

    if doc_metrics.bad_examples:
        rel_logger.info("Bad examples from this document:")
        for i, (relation, error) in enumerate(doc_metrics.bad_examples, 1):
            rel_logger.info(f"  {i}. {relation!r} -> {error}")

    # Update global metrics
    rel_metrics.total_relations += doc_metrics.total_relations
    rel_metrics.successfully_parsed += doc_metrics.successfully_parsed
    rel_metrics.rejected_relations += doc_metrics.rejected_relations
    rel_metrics.bad_examples.extend(
        doc_metrics.bad_examples[
            : rel_metrics.max_bad_examples - len(rel_metrics.bad_examples)
        ]
    )

    return normalized_relations


def _fix_pseudo_python_dict(text: str) -> str:
    """
    Convert pseudo-Python dict strings to valid JSON.

    Examples:
    - '{a:b, c:d}' -> '{"a":"b", "c":"d"}'
    - "{key:'value', num:123}" -> '{"key":"value", "num":123}'
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace single quotes with double quotes
    text = text.replace("'", '"')

    # Add quotes around unquoted keys and string values
    # This regex matches key:value pairs where key or value might be unquoted
    # Pattern explanation:
    # (^|[,{]\s*) - start of string or comma/brace followed by optional whitespace
    # ([a-zA-Z_][a-zA-Z0-9_]*) - unquoted identifier (key)
    # (\s*:\s*) - colon with optional whitespace
    # ([a-zA-Z_][a-zA-Z0-9_]*) - unquoted identifier (value)
    # (?=[,}]) - followed by comma or closing brace

    # First, quote unquoted keys
    text = re.sub(r"(^|[,{]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', text)

    # Then, quote unquoted string values (but not numbers or booleans)
    text = re.sub(r"(:\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?=[,}])", r'\1"\2"', text)

    return text


class MedicalHypergraphPipeline:

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        self.base_chunker = get_chunker(
            "gpt-4.1", chunking_type="text", max_tokens=250, trim=False, overlap=0
        )
        self.chunker = self.base_chunker
        self.rule_cache: Dict[str, TargetRule] = self._load_rules_from_file()
        self.context_rule_cache: Dict[str, ConTextRule] = (
            self._load_context_rules_from_file()
        )
        PIPELINE_INSTANCE_CONTEXT["current"] = self
        self.nlp: Language = self._setup_nlp_pipeline()
        PIPELINE_INSTANCE_CONTEXT["current"] = None
        self.cid_counter = itertools.count(1)
    def chunks(self, text: str):
        """Yield (start_char, end_char, chunk_text) tuples."""
        for start_index, chunk_text in self.base_chunker.chunk_indices(text):
            end_index = start_index + len(chunk_text)
            yield start_index, end_index, chunk_text

    def _setup_nlp_pipeline(self) -> Language:
        # 1) Load the vector model first
        vec_nlp = spacy.load("en_core_sci_scibert")

        # 2) Build the rest of your pipeline from the config (no overrides kwarg)
        cfg = spacy.util.load_config(pathlib.Path("config.cfg"))
        nlp = spacy.util.load_model_from_config(cfg, auto_fill=True)
        nlp.tokenizer = vec_nlp.tokenizer
        nlp.vocab.vectors = vec_nlp.vocab.vectors
        nlp.vocab.vectors.name = vec_nlp.vocab.vectors.name
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", after="spancat")

        nlp.add_pipe("dynamic_target_rules_component", after="sentencizer")

        # 4. Target Matcher
        nlp.add_pipe(
            "medspacy_target_matcher",
            after="dynamic_target_rules_component",
            config={"rules": None},
        )
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
        initial_target_rules = list(self.rule_cache.values())
        if initial_target_rules:
            target_matcher.add(initial_target_rules)
            logging.info(
                f"Programmatically added {len(initial_target_rules)} rule(s) to the TargetMatcher."
            )

        # 5. ConText Component - Add pipe with no rules, then add them from the cache
        context_pipe = nlp.add_pipe(
            "medspacy_context", config={"rules": None}, last=True
        )
        context_pipe.target_span_groups = ["ents", "medspacy_spans"]
        context_pipe.context_graph = True  # type: ignore
        initial_context_rules = list(self.context_rule_cache.values())
        if initial_context_rules:
            context_pipe.add(initial_context_rules)
        else:
            logging.warning(
                "No ConText rules loaded. The context component may not function as expected."
            )

        nlp.add_pipe("debug_context_graph", after="medspacy_context")

        # 6. Entity Linker
        linker_config = self.config.get(
            "linker_config",
            {"resolve_abbreviations": True, "linker_name": "umls", "threshold": 0.7},
        )
        nlp.add_pipe("scispacy_linker", config=linker_config, last=True)

        nlp.initialize(lambda: [])
        logging.info(f"Pipeline ready: {nlp.pipe_names}")
        return nlp

    def _load_rules_from_file(self) -> Dict[str, TargetRule]:
        """Loads TargetRules from the JSON file specified in the config."""
        rules_path_str = self.config.get("target_rules_path")
        if not rules_path_str:
            logging.info(
                "`target_rules_path` not in config. Starting with an empty rule set."
            )
            return {}

        rules_path = pathlib.Path(rules_path_str)
        rule_cache = {}

        if not rules_path.exists():
            logging.info(
                f"No saved TargetRules found at {rules_path}—starting with an empty rule set."
            )
            return rule_cache

        try:
            with rules_path.open("r", encoding="utf8") as f:
                raw_rules_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(
                f"Could not read or parse {rules_path}: {e}. Starting fresh."
            )
            return rule_cache

        if not isinstance(raw_rules_data, list):
            logging.warning(
                f"Expected a list of rules in {rules_path}. Starting fresh."
            )
            return rule_cache

        loaded_rules = [self._coerce_saved_rule(item) for item in raw_rules_data]
        good_rules = [
            rule for rule in loaded_rules if rule and self._is_good_rule(rule)
        ]

        discarded_count = len(raw_rules_data) - len(good_rules)
        if discarded_count > 0:
            logging.warning(
                f"Discarded {discarded_count} invalid rule(s) from {rules_path}"
            )

        for rule in good_rules:
            rule_cache[self._key_for_rule(rule)] = rule

        logging.info(f"Restored {len(rule_cache)} TargetRules from {rules_path}")
        return rule_cache

    def _load_section_rules_from_file(self) -> List[SectionRule]:
        """Loads SectionRule objects from the JSON file specified in the config."""
        rules_path_str = self.config.get("section_rules_path")
        if not rules_path_str:
            logging.warning(
                "`section_rules_path` not specified in config. No section rules will be loaded."
            )
            return []

        rules_path = pathlib.Path(rules_path_str)
        if not rules_path.exists():
            logging.warning(
                f"Sectionizer rules file not found at {rules_path}. No section rules will be loaded."
            )
            return []

        try:
            with rules_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_rules = data.get("section_rules", [])
            if not isinstance(raw_rules, list):
                logging.error(
                    f"Sectionizer rules file {rules_path} must contain a list under the 'section_rules' key."
                )
                return []

            section_rules = [
                SectionRule.from_dict(rule_data) for rule_data in raw_rules
            ]
            logging.info(
                f"Loaded {len(section_rules)} Sectionizer rules from {rules_path}"
            )
            return section_rules
        except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
            logging.error(
                f"Failed to load or parse Sectionizer rules from {rules_path}: {e}"
            )
            return []

    def _load_context_rules_from_file(self) -> Dict[str, ConTextRule]:
        """Loads ConTextRule objects from JSON and populates the cache."""
        rules_path_str = self.config.get("context_rules_path")
        if not rules_path_str:
            logging.warning(
                "`context_rules_path` not specified in config. Starting with empty context rule set."
            )
            return {}

        rules_path = pathlib.Path(rules_path_str)
        if not rules_path.exists():
            logging.warning(
                f"Context rules file not found at {rules_path}. Starting fresh."
            )
            return {}

        cache = {}
        try:
            with rules_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_rules = data.get("context_rules", [])
            if not isinstance(raw_rules, list):
                logging.error(
                    f"Context rules file {rules_path} must contain a list under 'context_rules' key."
                )
                return {}

            for rule_data in raw_rules:
                rule = ConTextRule.from_dict(rule_data)
                key = self._key_for_context_rule(rule)
                cache[key] = rule

            logging.info(
                f"Loaded and cached {len(cache)} ConText rules from {rules_path}"
            )
            return cache
        except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
            logging.error(
                f"Failed to load or parse ConText rules from {rules_path}: {e}"
            )
            return {}

    def save_rules(self) -> None:
        """Serializes all valid TargetRules from the cache to a JSON file."""
        rules_path = pathlib.Path(self.config["target_rules_path"])
        valid_rules_to_save = [
            self._rule_to_dict(rule)
            for rule in self.rule_cache.values()
            if self._is_good_rule(rule)
        ]

        if not valid_rules_to_save:
            logging.info(f"No valid target rules in cache to save to {rules_path}.")
            return

        with rules_path.open("w", encoding="utf8") as f:
            json.dump(valid_rules_to_save, f, indent=2)
        logging.info(f"Saved {len(valid_rules_to_save)} TargetRule(s) to {rules_path}")

    def save_context_rules(self) -> None:
        """Serializes all ConTextRules from the cache to a JSON file."""
        rules_path_str = self.config.get("context_rules_path")
        if not rules_path_str:
            logging.error(
                "Cannot save context rules: `context_rules_path` not specified in config."
            )
            return

        rules_path = pathlib.Path(rules_path_str)
        rules_to_save = [rule.to_dict() for rule in self.context_rule_cache.values()]

        if not rules_to_save:
            logging.info(f"No context rules in cache to save to {rules_path}.")
            return

        # Wrap in the expected dictionary structure
        output_data = {"context_rules": rules_to_save}

        with rules_path.open("w", encoding="utf8") as f:
            json.dump(output_data, f, indent=2)
        logging.info(f"Saved {len(rules_to_save)} ConTextRule(s) to {rules_path}")

    @staticmethod
    def _key_for_rule(rule: TargetRule) -> str:
        """Generates a unique key for a TargetRule."""
        key_parts: List[str] = []
        if rule.pattern:
            if isinstance(rule.pattern, str):
                key_parts.append(f"pattern_regex:{rule.pattern}")
            elif isinstance(rule.pattern, list):
                key_parts.append(
                    f"pattern_tokens:{json.dumps(rule.pattern, sort_keys=True)}"
                )
        else:
            key_parts.append(f"literal_exact:{rule.literal.lower()}")
        key_parts.append(f"category:{rule.category.lower()}")
        return "|".join(key_parts)

    @staticmethod
    def _key_for_context_rule(rule: ConTextRule) -> str:
        """Generates a unique key for a ConTextRule."""
        key_parts: List[str] = []
        if rule.pattern:
            key_parts.append(
                f"pattern_tokens:{json.dumps(rule.pattern, sort_keys=True)}"
            )
        else:
            key_parts.append(f"literal_exact:{rule.literal.lower()}")
        key_parts.append(f"category:{rule.category.lower()}")
        key_parts.append(f"direction:{rule.direction.lower()}")
        return "|".join(key_parts)

    @staticmethod
    def _is_good_rule(obj: Any) -> bool:
        return (
            isinstance(obj, TargetRule)
            and obj.literal
            and obj.literal.strip()
            and obj.category
            and obj.category.strip()
        )

    @staticmethod
    def _rule_to_dict(rule: TargetRule) -> dict[str, Any]:
        return {
            "literal": rule.literal,
            "pattern": rule.pattern,
            "category": rule.category,
            "attributes": rule.attributes,
        }

    def _dict_to_rule(self, data: dict[str, Any]) -> TargetRule:
        try:
            return TargetRule(
                literal=data["literal"],
                category=data["category"],
                pattern=data.get("pattern"),
                attributes=data.get("attributes"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid dict for TargetRule: {data}. Error: {e}")

    def _coerce_saved_rule(self, obj: Any) -> Optional[TargetRule]:
        try:
            if isinstance(obj, dict):
                rule = self._dict_to_rule(obj)
                return rule if self._is_good_rule(rule) else None
        except ValueError as e:
            logging.warning(f"Skipping invalid rule data: {obj}. Error: {e}")
        return None

    def add_context_rule(
        self,
        literal: str,
        category: str,
        direction: str,
        pattern: Optional[List[Dict]] = None,
    ) -> None:
        """
        Adds a new ConTextRule to the pipeline dynamically.

        The rule is added to the cache and the active pipeline component.
        Call `save_context_rules()` to persist the new rule to the JSON file.

        Args:
            literal: The literal phrase of the rule.
            category: The category of the modifier (e.g., "NEGATED_EXISTENCE").
            direction: The direction of the rule (e.g., "FORWARD", "BACKWARD").
            pattern: An optional spaCy pattern for more complex matches.
        """
        new_rule = ConTextRule(
            literal=literal, category=category, direction=direction, pattern=pattern
        )
        rule_key = self._key_for_context_rule(new_rule)

        if rule_key in self.context_rule_cache:
            logging.info(f"ConTextRule for '{literal}' already exists. Skipping.")
            return

        # Add to cache
        self.context_rule_cache[rule_key] = new_rule

        # Add to the live component in the nlp pipeline
        try:
            context_pipe = self.nlp.get_pipe("medspacy_context")
            context_pipe.add([new_rule])
            logging.info(
                f"Dynamically added new ConTextRule: '{literal}' | {category} | {direction}"
            )
        except KeyError:
            logging.error(
                "Could not find 'medspacy_context' in the pipeline to add new rule."
            )

    # ----------------------------------------------------------------------
    # public API ------------------------------------------------------------
    # ----------------------------------------------------------------------
    def process_text(self, text: str) -> Tuple[hnx.Hypergraph, Dict[str, Any]]:
        """
        Chunk ► NLP ► atoms ► hypergraph.
        """
        aggregated = {
            "entity_atoms": [],
            "sentence_relation_hints": [],
            "context_graph": {"targets": [], "modifiers": [], "edges": []},
        }
        pathlib.Path("chunks.txt").write_text(str(self.chunks(text)))
        # Process each chunk and aggregate results
        for start, end, chunk in self.chunks(text):
            doc = self.nlp(chunk)
            atoms = self._extract_hypergraph_atoms(doc, offset=start)
            aggregated["entity_atoms"].extend(atoms["entity_atoms"])
            aggregated["sentence_relation_hints"].extend(
                atoms["sentence_relation_hints"]
            )
            for k in ("targets", "modifiers", "edges"):
                aggregated["context_graph"][k].extend(atoms["context_graph"][k])

        # Create synthetic document-level boundaries with proper offsets
        pseudo_doc_sents = []
        for start, end, chunk in self.chunks(text):
            chunk_doc = self.nlp(chunk)
            for sent in chunk_doc.sents:
                # Create a synthetic sentence object with adjusted character positions
                class SyntheticSent:
                    def __init__(self, text, start_char, end_char):
                        self.text = text
                        self.start_char = start_char
                        self.end_char = end_char

                pseudo_doc_sents.append(
                    SyntheticSent(
                        sent.text, sent.start_char + start, sent.end_char + start
                    )
                )

        pseudo_doc_paras = text.split("\n\n")

        H = self.build_hypergraph(
            aggregated, sent_bounds=pseudo_doc_sents, para_bounds=pseudo_doc_paras
        )
        return H, aggregated

    # ----------------------------------------------------------------------
    # internals -------------------------------------------------------------
    # ----------------------------------------------------------------------
    def _extract_hypergraph_atoms(self, doc: Doc, offset: int = 0) -> Dict[str, Any]:
        """Extract all hypergraph data from a document with optional character offset."""
        ent_atoms, canonical_map = self._create_entity_instance_atoms(doc, offset)
        ent_idx_to_canonical_id = {}
        for idx, ent in enumerate(doc.ents):
            surface = ent.text.lower().strip()
            best_cui = self._best_cui(ent)
            canon_key = (surface, best_cui)
            canonical_id = canonical_map.get(canon_key)
            if canonical_id:
                ent_idx_to_canonical_id[idx] = canonical_id

        # Map entity indices to their canonical IDs
        #ent_atoms = [
        #    {**atom, "canonical_id": ent_idx_to_canonical_id.get(atom["id"], None)}
        #    for atom in ent_atoms
        #]

        # Extract context graph from medspaCy
        context_graph = {"targets": [], "modifiers": [], "edges": []}
        if hasattr(doc._, "context_graph") and doc._.context_graph:
            cg = doc._.context_graph

            # Collect targets with offset adjustment
            for target in getattr(cg, "targets", []):
                if hasattr(target, "target_span"):
                    start, end = target.target_span
                    context_graph["targets"].append(
                        {
                            "text": doc[start : end + 1].text,
                            "start_char": doc[start].idx + offset,
                            "end_char": doc[end].idx + len(doc[end].text) + offset,
                            "span_indices": (start, end),
                        }
                    )

            # Collect modifiers with offset adjustment
            for modifier in getattr(cg, "modifiers", []):
                if hasattr(modifier, "modifier_span"):
                    start, end = modifier.modifier_span
                    context_graph["modifiers"].append(
                        {
                            "text": doc[start : end + 1].text,
                            "start_char": doc[start].idx + offset,
                            "end_char": doc[end].idx + len(doc[end].text) + offset,
                            "category": getattr(modifier, "category", None),
                            "span_indices": (start, end),
                        }
                    )

            # Collect edges
            for edge in getattr(cg, "edges", []):
                target = edge[0] if len(edge) > 0 else None
                modifier = edge[1] if len(edge) > 1 else None

                # Serialize context objects to dictionaries
                target = _serialize_context_object(target)
                modifier = _serialize_context_object(modifier)

                context_graph["edges"].append({"target": target, "modifier": modifier})

        # Extract sentence relation hints using normalized relations
        sentence_relation_hints = []
        normalized_relations = normalize_relations(doc)
        for relation in normalized_relations:
            try:
                # Convert relation to text-based format for offset independence
                dep_idx = int(relation.get("dep", -1))
                dest_idx = int(relation.get("dest", -1))
                if (
                    dep_idx >= 0
                    and dest_idx >= 0
                    and dep_idx < len(doc.ents)
                    and dest_idx < len(doc.ents)
                ):
                    dep_ent = doc.ents[dep_idx]
                    dest_ent = doc.ents[dest_idx]
                    dep_cid = ent_idx_to_canonical_id.get(dep_idx, f"ENT{dep_idx}")
                    dest_cid = ent_idx_to_canonical_id.get(dest_idx, f"ENT{dest_idx}")
                    sentence_relation_hints.append(
                        {
                            "dep_cid": dep_cid,
                            "dep_text": dep_ent.text,
                            "dest_cid": dest_cid,
                            "dest_text": dest_ent.text,
                            "relation": relation.get("relation", "UNKNOWN"),
                            "dep_start_char": dep_ent.start_char + offset,
                            "dep_end_char": dep_ent.end_char + offset,
                            "dest_start_char": dest_ent.start_char + offset,
                            "dest_end_char": dest_ent.end_char + offset,
                            "sentence_text": dep_ent.sent.text.replace("\n", " "),
                        }
                    )
            except (ValueError, IndexError, AttributeError) as e:
                logging.debug(f"Skip invalid relation {relation}: {e}")

        return {
            "entity_atoms": ent_atoms,
            "sentence_relation_hints": sentence_relation_hints,
            "context_graph": context_graph,
        }

    def _extract_relation_hints(self, doc: Doc) -> List[Dict[str, Any]]:
        """Pull relations from doc._.rel (set by spacy.REL.v1)."""
        hints: List[Dict[str, Any]] = []
        if Doc.has_extension("rel") and doc._.rel:
            for (head_start, head_end), (child_start, child_end), label in doc._.rel:
                head_span = doc[head_start:head_end]
                child_span = doc[child_start:child_end]
                hints.append(
                    {
                        "sentence": head_span.sent.text,
                        "head": head_span.text,
                        "child": child_span.text,
                        "label": label,
                    }
                )
        return hints

    # ---------- entity atoms ----------------------------------------------
    def _create_entity_instance_atoms(
        self, doc: Doc, offset: int = 0
    ) -> Tuple[List[Dict], Dict]:
        instance_atoms: List[Dict] = []
        canonical_map: Dict[Tuple[str, Optional[str]], str] = {}
        #cid_counter = itertools.count(1)

        all_ents = list(doc.ents) + list(doc.spans.get("spancat", []))
        seen = set()

        for span in sorted(all_ents, key=lambda s: (s.start_char, s.end_char)):
            if (span.start_char, span.end_char) in seen:
                continue
            seen.add((span.start_char, span.end_char))

            surface = span.text.lower().strip()
            best_cui = self._best_cui(span)
            canon_key = (surface, best_cui)
            canonical_id = canonical_map.setdefault(
                canon_key, f"ENT{next(self.cid_counter)}"
            )

            modifiers = (
                [
                    mod.category
                    for mod in (span._.modifiers or [])
                    if hasattr(mod, "category")
                ]
                if getattr(span._, "modifiers", None)
                else []
            )

            section_title = (
                getattr(span._, "section_category", None)
                or getattr(span._, "section_label", None)
                or "UNCATEGORISED"
            )

            instance_atoms.append(
                {
                    "instance_id": f"inst_{len(instance_atoms) + 1}",
                    "canonical_id": canonical_id,
                    "text": span.text,
                    "label": span.label_,
                    "start_char": span.start_char + offset,
                    "end_char": span.end_char + offset,
                    "sentence_text": span.sent.text.replace("\n", " "),
                    "section_title": section_title,
                    "umls_linking": self._get_umls_details(span),
                    "contextual_modifiers": modifiers,
                }
            )

        return instance_atoms, canonical_map

    @staticmethod
    def _best_cui(span: Span) -> Optional[str]:
        if span._.get("kb_ents"):
            return span._.kb_ents[0][0]
        if span._.get("umls_ents"):
            return span._.umls_ents[0][0]
        return None

    @staticmethod
    def _get_umls_details(ent: Span) -> List[Dict]:
        """
        Works for **medspaCy / scispaCy** UMLS linker *and* generic spaCy-LLM
        KB linker – it simply echoes whatever’s in `ent._.kb_ents`.
        """
        if not ent._.kb_ents:
            return []
        return [{"cui": cui, "score": round(score, 3)} for cui, score in ent._.kb_ents]

    @staticmethod
    def _merge_into_global(aggregated: Dict[str, Any], atoms: Dict[str, Any]) -> None:
        """Merge chunk atoms into the global aggregated structure."""
        aggregated["entity_atoms"].extend(atoms["entity_atoms"])
        aggregated["sentence_relation_hints"].extend(atoms["sentence_relation_hints"])
        for k in ("targets", "modifiers", "edges"):
            aggregated["context_graph"][k].extend(atoms["context_graph"][k])

    # ---------- hypergraph build ------------------------------------------
    @staticmethod
    def build_hypergraph(
        data: Dict[str, Any], doc: Doc = None, sent_bounds=None, para_bounds=None
    ) -> hnx.Hypergraph:
        nodes: Dict[str, Dict] = {}
        edges: Dict[str, set] = defaultdict(set)
        eattrs: Dict[str, Dict] = {}

        # nodes -----------------------------------------------------------
        for inst in data["entity_atoms"]:
            cid = inst["canonical_id"]
            nodes.setdefault(
                cid,
                {
                    "text": inst["text"],
                    "label": inst["label"],
                    "umls": inst["umls_linking"],
                },
            )

        # sentence-level edges -------------------------------------------
        if sent_bounds:
            for i, sent in enumerate(sent_bounds, 1):
                # For synthetic sentences, we need to calculate boundaries differently
                if hasattr(sent, "start_char"):
                    sent_start, sent_end = sent.start_char, sent.end_char
                    sent_text = sent.text.replace("\n", " ")
                else:
                    # Fallback for other sent representations
                    continue

                member_cids = {
                    inst["canonical_id"]
                    for inst in data["entity_atoms"]
                    if sent_start <= inst["start_char"] < sent_end
                }
                if member_cids:
                    eid = f"sent_{i}"
                    edges[eid] = member_cids
                    eattrs[eid] = {
                        "level": "sentence",
                        "certainty": 1.0,
                        "text": sent_text,
                    }
        elif doc:
            # Fallback to original behavior when doc is provided
            for i, sent in enumerate(doc.sents, 1):
                member_cids = {
                    inst["canonical_id"]
                    for inst in data["entity_atoms"]
                    if sent.start_char <= inst["start_char"] < sent.end_char
                }
                if member_cids:
                    eid = f"sent_{i}"
                    edges[eid] = member_cids
                    eattrs[eid] = {
                        "level": "sentence",
                        "certainty": 1.0,
                        "text": sent.text.replace("\n", " "),
                    }

        # paragraph edges (very simple) ----------------------------------
        if para_bounds:
            paragraphs = [p for p in para_bounds if p.strip()]
            pos = 0
            original_text = "\n\n".join(paragraphs)  # Reconstruct for position finding
            for i, para in enumerate(paragraphs, 1):
                start = original_text.find(para, pos)
                end = start + len(para)
                pos = end + 2  # Account for \n\n separator
                member_cids = {
                    inst["canonical_id"]
                    for inst in data["entity_atoms"]
                    if start <= inst["start_char"] < end
                }
                if member_cids:
                    eid = f"para_{i}"
                    edges[eid] = member_cids
                    eattrs[eid] = {
                        "level": "section",
                        "certainty": 1.0,
                        "text": f"Paragraph {i}",
                    }
        elif doc:
            # Fallback to original behavior when doc is provided
            paragraphs = [p for p in doc.text.split("\n\n") if p.strip()]
            pos = 0
            for i, para in enumerate(paragraphs, 1):
                start = doc.text.find(para, pos)
                end = start + len(para)
                pos = end
                member_cids = {
                    inst["canonical_id"]
                    for inst in data["entity_atoms"]
                    if start <= inst["start_char"] < end
                }
                if member_cids:
                    eid = f"para_{i}"
                    edges[eid] = member_cids
                    eattrs[eid] = {
                        "level": "section",
                        "certainty": 1.0,
                        "text": f"Paragraph {i}",
                    }

        # lonely nodes ----------------------------------------------------
        used = {n for s in edges.values() for n in s}
        for cid in nodes:
            if cid not in used:
                edges[f"loop_{cid}"] = {cid}
                eattrs[f"loop_{cid}"] = {"level": "loop_span", "certainty": 1.0}

        # construct HNX ---------------------------------------------------
        H = hnx.Hypergraph(edges)
        for cid, attrs in nodes.items():
            H.nodes[cid].attrs = attrs

        inc_g = _get_incidence_graph(H)
        for eid, attrs in eattrs.items():
            H.edges[eid].attrs = attrs
            for n in H.edges[eid].elements:
                inc_g.edges[(n, eid)].update(attrs)

        # propagate certainty from edges to nodes
        for cid in H.nodes:
            # Check if the node exists in incidence_dict before accessing it
            if cid in H.incidence_dict:
                edge_certs = [eattrs[eid]["certainty"] for eid in H.incidence_dict[cid]]
                H.nodes[cid].attrs["certainty"] = min(edge_certs) if edge_certs else 1.0
            else:
                # Node exists but has no edges, set default certainty
                H.nodes[cid].attrs["certainty"] = 1.0
        return H

    # ---------- serialisation --------------------------------------------
    @staticmethod
    def save_hypergraph(H: hnx.Hypergraph, path: Union[str, pathlib.Path]) -> None:
        inc = _get_incidence_graph(H).copy()
        nx.set_node_attributes(inc, False, "is_hyperedge")
        for n in H.nodes:
            if H.nodes[n].attrs:
                inc.nodes[n]["node_attrs"] = dict(H.nodes[n].attrs)
        for e in H.edges:
            inc.nodes[e]["is_hyperedge"] = True
            if H.edges[e].attrs:
                inc.nodes[e]["edge_attrs"] = dict(H.edges[e].attrs)
        nx.write_gpickle(inc, str(path))
        logging.info(f"Hypergraph saved → {path}")

    @staticmethod
    def load_hypergraph(path: Union[str, pathlib.Path]) -> hnx.Hypergraph:
        inc = nx.read_gpickle(str(path))
        if hasattr(hnx.Hypergraph, "from_bipartite_graph"):
            H = hnx.Hypergraph.from_bipartite_graph(inc)
        elif hasattr(hnx.Hypergraph, "from_incidence_graph"):
            H = hnx.Hypergraph.from_incidence_graph(inc)
        else:
            edge_members = defaultdict(set)
            for u, v in inc.edges:
                if inc.nodes[u].get("is_hyperedge"):
                    edge_members[u].add(v)
                elif inc.nodes[v].get("is_hyperedge"):
                    edge_members[v].add(u)
            H = hnx.Hypergraph(edge_members)

        for edge_id in H.edges:
            edge_obj = H.edges[edge_id]
            attrs = inc.nodes[edge_id].get("edge_attrs")
            if attrs:
                if edge_obj.attrs is None:
                    edge_obj.attrs = {}
                edge_obj.attrs.update(attrs)

        for node_id in H.nodes:
            node_obj = H.nodes[node_id]
            attrs = inc.nodes[node_id].get("node_attrs")
            if attrs:
                if node_obj.attrs is None:
                    node_obj.attrs = {}
                node_obj.attrs.update(attrs)

        logging.info(f"Hypergraph loaded ← {path}")
        return H


class DynamicTargetRulesComponent:
    """A spaCy pipeline component to dynamically add TargetRules from NER entities."""

    def __init__(
        self,
        rule_cache: Dict[str, TargetRule],
        key_for_rule_func: Callable[[TargetRule], str],
        nlp_instance: Language,
    ):
        self.rule_cache = rule_cache
        self.key_for_rule = key_for_rule_func
        self.nlp = nlp_instance

    def __call__(self, doc: Doc) -> Doc:
        """
        Identifies new potential rules from NER entities and adds them to the
        TargetMatcher for the current `doc` processing.
        """
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        newly_added_rules = []

        for ent in doc.ents:
            literal = ent.text.strip()
            if not literal:
                continue

            new_rule = TargetRule(literal, ent.label_)
            new_key = self.key_for_rule(new_rule)

            if new_key not in self.rule_cache:
                self.rule_cache[new_key] = new_rule
                newly_added_rules.append(new_rule)

        if newly_added_rules:
            target_matcher.add(newly_added_rules)
            logging.info(
                f"Dynamically added {len(newly_added_rules)} new rule(s) to TargetMatcher."
            )

        return doc
