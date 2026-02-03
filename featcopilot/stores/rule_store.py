"""Persistent storage for transform rules.

Provides JSON-file based storage for saving, loading, and searching
reusable transform rules.
"""

import json
import os
from pathlib import Path
from typing import Optional

from featcopilot.core.transform_rule import TransformRule
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class TransformRuleStore:
    """
    Persistent storage for transform rules.

    Stores rules in a JSON file for reuse across sessions and datasets.
    Supports searching by tags, description similarity, and column patterns.

    Parameters
    ----------
    path : str, optional
        Path to the JSON file for storage. Defaults to ~/.featcopilot/rules.json

    Examples
    --------
    >>> store = TransformRuleStore()
    >>> store.save_rule(rule)
    >>> matching = store.find_matching_rules(columns=["price", "quantity"])
    >>> all_rules = store.list_rules()
    """

    DEFAULT_PATH = "~/.featcopilot/rules.json"

    def __init__(self, path: Optional[str] = None):
        self.path = Path(os.path.expanduser(path or self.DEFAULT_PATH))
        self._rules: dict[str, TransformRule] = {}
        self._ensure_directory()
        self._load()

    def _ensure_directory(self) -> None:
        """Ensure the storage directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load rules from storage file."""
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                    self._rules = {rule_id: TransformRule.from_dict(rule_data) for rule_id, rule_data in data.items()}
                logger.debug(f"Loaded {len(self._rules)} rules from {self.path}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load rules from {self.path}: {e}")
                self._rules = {}
        else:
            self._rules = {}

    def _save(self) -> None:
        """Save rules to storage file."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                data = {rule_id: rule.to_dict() for rule_id, rule in self._rules.items()}
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._rules)} rules to {self.path}")
        except OSError as e:
            logger.error(f"Failed to save rules to {self.path}: {e}")
            raise

    def save_rule(self, rule: TransformRule) -> str:
        """
        Save a rule to the store.

        Parameters
        ----------
        rule : TransformRule
            The rule to save

        Returns
        -------
        str
            The rule's ID
        """
        self._rules[rule.id] = rule
        self._save()
        logger.info(f"Saved rule '{rule.name}' with ID {rule.id}")
        return rule.id

    def get_rule(self, rule_id: str) -> Optional[TransformRule]:
        """
        Get a rule by ID.

        Parameters
        ----------
        rule_id : str
            The rule's ID

        Returns
        -------
        TransformRule or None
            The rule if found, None otherwise
        """
        return self._rules.get(rule_id)

    def get_rule_by_name(self, name: str) -> Optional[TransformRule]:
        """
        Get a rule by name.

        Parameters
        ----------
        name : str
            The rule's name

        Returns
        -------
        TransformRule or None
            The first rule matching the name, None if not found
        """
        for rule in self._rules.values():
            if rule.name == name:
                return rule
        return None

    def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a rule by ID.

        Parameters
        ----------
        rule_id : str
            The rule's ID

        Returns
        -------
        bool
            True if deleted, False if not found
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            self._save()
            logger.info(f"Deleted rule {rule_id}")
            return True
        return False

    def list_rules(self, tags: Optional[list[str]] = None) -> list[TransformRule]:
        """
        List all rules, optionally filtered by tags.

        Parameters
        ----------
        tags : list[str], optional
            Filter rules that have all specified tags

        Returns
        -------
        list[TransformRule]
            List of matching rules
        """
        rules = list(self._rules.values())

        if tags:
            rules = [r for r in rules if all(t in r.tags for t in tags)]

        return rules

    def find_matching_rules(
        self,
        columns: Optional[list[str]] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_usage: int = 0,
    ) -> list[tuple[TransformRule, dict[str, str]]]:
        """
        Find rules that can be applied to the given context.

        Parameters
        ----------
        columns : list[str], optional
            Available column names to match against
        description : str, optional
            Description to search for (keyword matching)
        tags : list[str], optional
            Required tags
        min_usage : int, default=0
            Minimum usage count

        Returns
        -------
        list[tuple[TransformRule, dict]]
            List of (rule, column_mapping) tuples for applicable rules,
            sorted by usage count (most used first)
        """
        results: list[tuple[TransformRule, dict[str, str]]] = []

        for rule in self._rules.values():
            # Filter by usage count
            if rule.usage_count < min_usage:
                continue

            # Filter by tags
            if tags and not all(t in rule.tags for t in tags):
                continue

            # Filter by description keywords
            if description:
                keywords = description.lower().split()
                rule_text = f"{rule.name} {rule.description}".lower()
                if not any(kw in rule_text for kw in keywords):
                    continue

            # Check column compatibility
            mapping: dict[str, str] = {}
            if columns:
                matches, mapping = rule.matches_columns(columns)
                if not matches:
                    continue

            results.append((rule, mapping))

        # Sort by usage count (descending)
        results.sort(key=lambda x: x[0].usage_count, reverse=True)

        return results

    def search_by_description(self, query: str, limit: int = 10) -> list[TransformRule]:
        """
        Search rules by description similarity.

        Parameters
        ----------
        query : str
            Search query
        limit : int, default=10
            Maximum number of results

        Returns
        -------
        list[TransformRule]
            Matching rules sorted by relevance
        """
        query_words = set(query.lower().split())
        scored_rules: list[tuple[float, TransformRule]] = []

        for rule in self._rules.values():
            rule_words = set(f"{rule.name} {rule.description}".lower().split())

            # Simple word overlap scoring
            overlap = len(query_words & rule_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored_rules.append((score, rule))

        # Sort by score descending
        scored_rules.sort(key=lambda x: x[0], reverse=True)

        return [rule for _, rule in scored_rules[:limit]]

    def import_rules(self, path: str, merge: bool = True) -> int:
        """
        Import rules from another JSON file.

        Parameters
        ----------
        path : str
            Path to import from
        merge : bool, default=True
            If True, merge with existing rules. If False, replace all.

        Returns
        -------
        int
            Number of rules imported
        """
        import_path = Path(os.path.expanduser(path))

        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")

        with open(import_path, encoding="utf-8") as f:
            data = json.load(f)

        if not merge:
            self._rules = {}

        count = 0
        for _rule_id, rule_data in data.items():
            rule = TransformRule.from_dict(rule_data)
            self._rules[rule.id] = rule
            count += 1

        self._save()
        logger.info(f"Imported {count} rules from {path}")

        return count

    def export_rules(self, path: str, tags: Optional[list[str]] = None) -> int:
        """
        Export rules to a JSON file.

        Parameters
        ----------
        path : str
            Path to export to
        tags : list[str], optional
            Only export rules with these tags

        Returns
        -------
        int
            Number of rules exported
        """
        export_path = Path(os.path.expanduser(path))
        export_path.parent.mkdir(parents=True, exist_ok=True)

        rules_to_export = self.list_rules(tags=tags)

        with open(export_path, "w", encoding="utf-8") as f:
            data = {r.id: r.to_dict() for r in rules_to_export}
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(rules_to_export)} rules to {path}")

        return len(rules_to_export)

    def clear(self) -> None:
        """Remove all rules from the store."""
        self._rules = {}
        self._save()
        logger.info("Cleared all rules")

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, rule_id: str) -> bool:
        return rule_id in self._rules

    def __iter__(self):
        return iter(self._rules.values())
