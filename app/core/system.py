from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """
    Handles the registration, retrieval, listing, and deletion of artifacts
    in the system. It uses a database for metadata storage and a storage
    system for saving artifact data.
    """

    def __init__(self, database: Database, storage: Storage) -> None:
        """
        Initialize the ArtifactRegistry with a database and storage backend.

        Args:
            database (Database): Database instance to manage metadata.
            storage (Storage): Storage instance to manage artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register an artifact by saving its data to storage and
        its metadata to the database.

        Args:
            artifact (Artifact): The artifact to register.
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)

        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)
        print(
            f"Artifact metadata for {artifact.name} saved \
            with ID {artifact.id}, version: {artifact.version}"
        )

    def list(self, type: str = None) -> List[Artifact]:
        """
        List all artifacts or those of a specific type.

        Args:
            type (str): The type of artifacts to filter by. Defaults to None.

        Returns:
            List[Artifact]: A list of artifacts matching the criteria.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieve a specific artifact by its ID.

        Args:
            artifact_id (str): The unique identifier of the artifact.

        Returns:
            Artifact: The requested artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete an artifact and its associated metadata.

        Args:
            artifact_id (str): The unique identifier of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)
        print(f"Deleted artifact {data['name']} with ID {artifact_id}")


class AutoMLSystem:
    """
    Singleton class for managing the AutoML system.
    """

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize the AutoML system with storage and database.

        Args:
            storage (LocalStorage): The storage instance for artifact data.
            database (Database): The database instance for metadata management.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Retrieve the singleton instance of the AutoMLSystem. If none exists,
        it will be created.

        Returns:
            AutoMLSystem: The singleton instance of the system.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo")),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Access the artifact registry.

        Returns:
            ArtifactRegistry: The registry instance.
        """
        return self._registry
