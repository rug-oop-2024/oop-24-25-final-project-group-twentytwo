from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Custom exception raised when a path is not found."""

    def __init__(self, path: str) -> None:
        """
        Initialize the NotFoundError exception.

        Args:
            path (str): The path that was not found. This will be included
                        in the error message and stored for further access.

        Example:
            raise NotFoundError("/some/nonexistent/path")
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class for storage backends.

    Defines the methods for saving, loading, deleting, and listing data in a
    storage system.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """Save data to a given path.

        Args:
            data (bytes): The data to save.
            path (str): The path to save the data.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Load data from a given path.

        Args:
            path (str): The path to load data from.

        Returns:
            bytes: The loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete data at a given path.

        Args:
            path (str): The path to delete the data.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """List all paths under a given path.

        Args:
            path (str): The path to list.

        Returns:
            List[str]: A list of paths.
        """
        pass


class LocalStorage(Storage):
    """An implementation of the Storage class for local file system storage.

    Provides methods to save, load, delete, and list files
    in a specified base directory.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """Initialize the LocalStorage instance with a base path.

        Args:
            base_path (str): The base directory for storing files.
                                Defaults to './assets'.
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """Save data to the local storage.

        Args:
            data (bytes): The data to save.
            key (str): The key (path) where the data will be stored.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """Load data from the local storage.

        Args:
            key (str): The key (path) of the data to load.

        Returns:
            bytes: The loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """Delete data from the local storage.

        Args:
            key (str): The key (path) of the data to delete.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """List all files in the directory specified by the prefix.

        Args:
            prefix (str): The prefix (path) to search for files.

        Returns:
            List[str]: A list of file paths relative to the base directory.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [
            os.path.relpath(p, self._base_path)
            for p in keys
            if os.path.isfile(p)
        ]

    def _assert_path_exists(self, path: str) -> None:
        """Ensure that a path exists.

        Args:
            path (str): The path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """Join the base path with the given path.

        Args:
            path (str): The path to join with the base path.

        Returns:
            str: The full path.
        """
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
