"""Stream one episode at a time to HDF5; publish only completed episodes."""
import json
from pathlib import Path

import h5py
import numpy as np


class EpisodeWriter:
    def __init__(self, directory, metadata, success_only=True):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata, self.success_only = metadata, success_only
        self.file, self.path, self.count = None, None, 0

    def start(self, instruction, calibration):
        if self.file is not None:
            raise RuntimeError("Finish or abort the current episode before starting another")
        self.count += 1
        # Exclusive creation prevents accidental replacement of an earlier collection.
        self.path = self.root / f"episode_{self.count:06d}.h5.partial"
        self.file = h5py.File(self.path, "x")
        self.file.attrs.update(instruction=instruction, metadata=json.dumps(self.metadata), fps=30)
        for key, value in calibration.items():
            self.file.attrs[key] = value

    def append(self, **fields):
        if self.file is None or not fields:
            raise RuntimeError("Start an episode before appending its frames")
        for key, value in fields.items():
            value = np.asarray(value)
            if key not in self.file:
                self.file.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                    chunks=(1, *value.shape), dtype=value.dtype, compression="gzip", compression_opts=1)
            dataset = self.file[key]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = value

    def finish(self, success):
        if self.file is None:
            raise RuntimeError("There is no active episode to finish")
        self.file.attrs["success"] = bool(success)
        self.file.close()
        self.file = None
        if success or not self.success_only:
            self.path.rename(self.path.with_suffix(""))
        else:
            self.path.unlink()

    def abort(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            # Preserve .partial to make interrupted capture identifiable.
