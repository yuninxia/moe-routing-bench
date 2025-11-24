#!/usr/bin/env python
from __future__ import annotations

import os

from download_dataset import BASE_DIR, download_dataset

if __name__ == "__main__":
    output_path = os.path.join(BASE_DIR, "raw", "social_i_qa_train.json")
    download_dataset("social_i_qa", "train", output_path)
