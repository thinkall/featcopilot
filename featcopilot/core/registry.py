"""Feature registry for tracking and managing features."""

from typing import Callable, Optional

from featcopilot.core.feature import Feature, FeatureOrigin


class FeatureRegistry:
    """
    Global registry for feature definitions and generators.

    Provides registration and lookup of:
    - Feature transformation functions
    - Feature generator classes
    - Custom feature definitions
    """

    _instance: Optional["FeatureRegistry"] = None
    _transformations: dict[str, Callable] = {}
    _generators: dict[str, type] = {}

    def __new__(cls) -> "FeatureRegistry":
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_default_transformations()
        return cls._instance

    def _init_default_transformations(self) -> None:
        """Initialize default transformation functions."""
        import numpy as np

        self._transformations = {
            "log": lambda x: np.log1p(np.abs(x)),
            "log10": lambda x: np.log10(np.abs(x) + 1),
            "sqrt": lambda x: np.sqrt(np.abs(x)),
            "square": lambda x: x**2,
            "cube": lambda x: x**3,
            "reciprocal": lambda x: 1 / (x + 1e-8),
            "abs": lambda x: np.abs(x),
            "sign": lambda x: np.sign(x),
            "exp": lambda x: np.exp(np.clip(x, -50, 50)),
            "sin": lambda x: np.sin(x),
            "cos": lambda x: np.cos(x),
            "tanh": lambda x: np.tanh(x),
        }

    def register_transformation(self, name: str, func: Callable) -> None:
        """
        Register a transformation function.

        Parameters
        ----------
        name : str
            Name of transformation
        func : callable
            Function that takes array and returns transformed array
        """
        self._transformations[name] = func

    def get_transformation(self, name: str) -> Optional[Callable]:
        """Get a registered transformation by name."""
        return self._transformations.get(name)

    def list_transformations(self) -> list[str]:
        """List all registered transformation names."""
        return list(self._transformations.keys())

    def register_generator(self, name: str, generator_class: type) -> None:
        """
        Register a feature generator class.

        Parameters
        ----------
        name : str
            Name of generator
        generator_class : type
            Class that generates features
        """
        self._generators[name] = generator_class

    def get_generator(self, name: str) -> Optional[type]:
        """Get a registered generator by name."""
        return self._generators.get(name)

    def list_generators(self) -> list[str]:
        """List all registered generator names."""
        return list(self._generators.keys())

    def create_feature(self, name: str, transformation: str, source_columns: list[str], **kwargs) -> Feature:
        """
        Create a feature using a registered transformation.

        Parameters
        ----------
        name : str
            Feature name
        transformation : str
            Name of registered transformation
        source_columns : list
            Columns used in transformation
        **kwargs : dict
            Additional feature attributes

        Returns
        -------
        Feature
            Created feature object
        """
        func = self.get_transformation(transformation)
        if func is None:
            raise ValueError(f"Unknown transformation: {transformation}")

        # Generate code string for the transformation
        code = f"result = {transformation}(df['{source_columns[0]}'])"

        return Feature(
            name=name,
            source_columns=source_columns,
            transformation=transformation,
            code=code,
            origin=FeatureOrigin.POLYNOMIAL,
            **kwargs,
        )


# Global registry instance
registry = FeatureRegistry()
