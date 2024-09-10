# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
import uuid

from iopath.common.file_io import PathManager as _PathManager

PathManager = _PathManager()


def save_to_jsonl(data, filename, write_mode="w"):
    with open(filename, write_mode, encoding="utf8") as file:
        for item in data:
            json_str = json.dumps(item)
            file.write(json_str + "\n")


def mark_done(path, version_string=None):
    """
    Mark this path as prebuilt.

    Marks the path as done by adding a '.built' file with the current timestamp
    plus a version description string if specified.

    :param str path:
        The file path to mark as built.

    :param str version_string:
        The version of this dataset.
    """
    with PathManager.open(os.path.join(path, ".built"), "w") as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write("\n" + version_string)


def built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.

    If a version_string is provided, this has to match, or the version is regarded as
    not built.
    """
    if version_string:
        fname = os.path.join(path, ".built")
        if not PathManager.exists(fname):
            return False
        else:
            with PathManager.open(fname, "r") as read:
                text = read.read().split("\n")
            return len(text) > 1 and text[1] == version_string
    else:
        return PathManager.exists(os.path.join(path, ".built"))


def map_str_to_uuid(str_to_map: str):
    """Generate a hash for a given string using v5 UUID
    In this case the md5 of str_to_map is computed, then some bits are truncated (for uuid namespace) and then
    the uuid is created given the md5 as the input name.

    Usage example: mapping prompts or src+tgt sequences in the dataset to a unique id.

    Args:
        str_to_map (str): string to map to an ID

    Returns:
        str: UUID converted to string
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str_to_map))
