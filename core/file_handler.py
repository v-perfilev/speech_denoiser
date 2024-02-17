import os
import random

import sys


class FileHandler:
    total_limit = sys.maxsize
    limit_per_folder = sys.maxsize

    def __init__(self, total_limit=sys.maxsize, limit_per_folder=sys.maxsize):
        self.total_limit = total_limit
        self.limit_per_folder = limit_per_folder

    def get_file_paths(self, base_paths, file_format, limit=None):
        file_paths = []

        for base_path in base_paths:
            for root, dirs, files in os.walk(base_path):
                files_processed = 0
                for file_name in files:
                    if file_name.endswith(f'.{file_format}'):
                        file_path = os.path.join(root, file_name)
                        file_paths.append(file_path)
                        files_processed += 1
                    if files_processed >= self.limit_per_folder:
                        break

        random.shuffle(file_paths)
        return file_paths[:limit if limit is not None else self.total_limit]
