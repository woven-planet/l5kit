import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union


L5KIT_DATA_FOLDER_ENV_KEY = "L5KIT_DATA_FOLDER"


class DataManager(ABC):
    @abstractmethod
    def require(self, key: str) -> str:
        pass


class LocalDataManager(DataManager):
    """LocalDataManager allows you to require data to be present in the subpath of a specific folder.

    Example:
    Your data folder is set to ``"/tmp/my-data-folder"``, and you
    call ``local_data_manager.require("path/to/image.jpg")``, it would check if
    ``"/tmp/my-data-folder/path/to/image.jpg"`` exists, and if so return that complete path
    (``"/tmp/my-data-folder/path/to/image.jpg"``), otherwise it raises an error.

    In order of precedence, the local data folder is set by
      1. Passing in the path to the constructor of ``LocalDataManager``
      2. Setting the ``L5KIT_DATA_FOLDER`` environment variable.
    """

    def __init__(self, local_data_folder: Optional[Union[str, Path]] = None):
        if local_data_folder is None:
            if L5KIT_DATA_FOLDER_ENV_KEY in os.environ:
                local_data_folder = os.environ[L5KIT_DATA_FOLDER_ENV_KEY]
            else:
                raise ValueError(
                    f"{L5KIT_DATA_FOLDER_ENV_KEY} has not been set and you passed None to this call."
                    "either set the env variable or pass a valid path to this call"
                )

        self.root_folder = Path(local_data_folder)

    def require(self, key: str) -> str:
        """Require checks whether the file with the given key is present in the local data folder, if it is not it
        raises an error. Returns the path to the file otherwise.

        Arguments:
            key (str): Path from the data folder where the file or folder should be present.

        Returns:
            str -- Filepath including the data folder where required key is present.
        """
        local_path = self.root_folder / key
        local_path_str = str(local_path)

        if local_path.exists():
            return local_path_str
        else:
            raise FileNotFoundError(f"{key} is not present in local data folder {self.root_folder}")
