#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024/11/03

@author: alain

Bake file

"""

from pathlib import Path
import struct
import numpy as np
import pickle
import zlib

import bpy

mx = (1 << 31) - 1

def pack_long(a):
    b = np.append(a & mx, a >> 31).tolist()
    return struct.pack(f"{len(b)}i", *b)

def unpack_long(n, packed):
    a = np.array(struct.unpack(f"{2*n}i", packed))
    return (a[:n] | (a[n:] << 31))

# =============================================================================================================================
# BakeFile class

class BakeFile:

    FILE_VERSION = 1
    MAX = 2000

    def __init__(self, name):
        """ > Initialize a bake file

        A bake file is a list of dictionaries used to restore a frame:

        ``` python
        bfile = BakeFile("test")

        array = [1, 2, 3]
        value = 3.14

        index = 10 # Frame number

        # Save frame data

        bfile.write(index, array=array, value=value)

        # Load saved data

        data = bfile.read(index)
        print(data['array'])
        # [1, 2, 3]
        print(data['value'])
        # 3.14
        ```

        Two sets of data are saved:
        - animation data : data per frame used to anime the objects, e.g. : vertices position
        - state data : data necessary to compute the next frame, e.g. objects speed

        > [!IMPORTANT]
        > The files are created in the temp folder defined in Blender. If this folder is not defined,
        > the temp file is created in the current working directory.

        > [!CAUTION]
        > The bake files are not deleted.

        > [!NOTE]
        > If the way to save a frame changes, the bake file is not valid anymore.

        Arguments
        ---------
        - name (str) : bake file name
        """

        # ----- Temp file names

        folder = self.get_temp_folder()

        self.file_name       = folder / f"{name}"
        self.anim_file_name  = folder / f"{name}.anim"
        self.state_file_name = folder / f"{name}.state"

        # ----- State frame

        self.state_frame = None

        # ----- Load the header

        header = self.read_header(self.file_name)
        if header is None:
            self.clear()

        else:
            self.offsets     = header['offsets']
            self.sizes       = header['sizes']
            self.state_frame = header['state_frame']

    # ====================================================================================================
    # Temp folder

    @classmethod
    def get_temp_folder(cls):

        try:
            folder_name = bpy.context.scene.npblender_temp_folder
        except:
            folder_name = bpy.context.preferences.filepaths.temporary_directory

        if folder_name == "":
            return Path.cwd()
        else:
            return Path(folder_name)

    # ====================================================================================================
    # Read file header

    @classmethod
    def read_header(cls, file_name):
        """ > read file header
        """

        anim_file_name  = str(file_name) + '.anim'
        state_file_name = str(file_name) + '.state'

        if not Path(anim_file_name).exists():
            return None

        with open(anim_file_name, 'rb') as f:

            file_version = struct.unpack('i', f.read(4))[0]
            if file_version != cls.FILE_VERSION:
                return None

            state_frame = struct.unpack('i', f.read(4))[0]
            if state_frame < 0:
                state_frame = None

            n = struct.unpack('i', f.read(4))[0]
            offsets = unpack_long(n, f.read(n*8))
            sizes   = list(struct.unpack(f'{n}i', f.read(n*4)))

        return {'offsets': offsets, 'sizes': sizes, 'state_frame': state_frame}

    # ====================================================================================================
    # Write file header

    def write_header(self, f):
        """ > Write file header.

        The structure of the file header is the following
        - int32 : file structure version, currently = 1
        - int32 : state frame (-1 for None)
        - int32 : number n of entries
        - 2*n*int32 : data block 'offsets' on twice 64 bits integers
        - n*int32 : data block 'sizes'

        Arguments
        ---------
        - f (file) : a file open in mode b
        """

        f.seek(0)
        f.write(struct.pack('i', self.FILE_VERSION))

        state_frame = -1 if self.state_frame is None else self.state_frame
        f.write(struct.pack('i', state_frame))

        n = len(self.offsets)
        f.write(struct.pack('i', n))

        f.write(pack_long(self.offsets))
        f.write(struct.pack(f'{n}i', *self.sizes))

    # ====================================================================================================
    # Clear file content

    def clear(self):
        """ > Clear file content

        Rewrite the file with empty entries
        """
        self.offsets     = np.zeros(self.MAX, int)
        self.sizes       = [0]*self.MAX
        self.state_frame = None
        with open(self.anim_file_name, 'wb') as f:
            self.write_header(f)

    # ====================================================================================================
    # Write data block

    def write(self, index, data, state):
        """ > Write data block

        > [!NOTE]
        > Data is compressed

        Arguments
        ---------
        - index (int) : frame number
        - data (any) : data block to be saved
        """

        if index >= len(self.offsets):
            return

        if (self.state_frame is not None) and (index != self.state_frame + 1):
            raise Exception(f"BakeFile algorithm error> Sorry frame to write {index} should be equal to state frame {self.state_frame} + 1.")

        if self.offsets[index] != 0:
            raise Exception(f"BakeFile algorithm error> BakeFile already contains an entry for index {index}!")

        # ----- Write animation data

        self.state_frame = index

        c = zlib.compress(pickle.dumps(data))

        with open(self.anim_file_name, 'r+b') as f:

            # Write animation data
            f.seek(0, 2)
            self.offsets[index] = f.tell()
            self.sizes[index]   = len(c)
            f.write(c)

            # Rewrite the header before closing
            self.write_header(f)

        # ----- Write state data

        with open(self.state_file_name, 'wb') as f:
            pickle.dump(state, f)

    # ====================================================================================================
    # Read data block

    def read(self, index):
        """ > Read data block

        Arguments
        ---------
        - index (int) : block index to read

        Returns
        -------
        - data : data saved with <#write> method, None if no entry is empty.
        """

        if index >= len(self.offsets):
            return None

        offset, size = self.offsets[index], self.sizes[index]
        if offset == 0:
            return None

        with open(self.anim_file_name, 'rb') as f:
            f.seek(offset)
            return pickle.loads(zlib.decompress(f.read(size)))

    # ====================================================================================================
    # Read state data

    def read_state(self):
        """ > Read state from file
        """
        if Path(self.state_file_name).exists():
            with open(self.state_file_name, 'rb') as f:
                return pickle.load(f)
        else:
            return None

    # ====================================================================================================
    # Management

    @classmethod
    def bake_files(cls):
        folder = cls.get_temp_folder()
        return [file for file in folder.iterdir() if file.suffix == '.anim']

    @classmethod
    def delete_bake_files(cls, *names):
        folder    = cls.get_temp_folder()
        to_delete = []
        all_files = len(names) == 0 or (len(names) == 1 and names[0] == "")

        for file in folder.iterdir():
            if file.suffix == '.anim' or file.suffix == '.state':
                if all_files or file.stem in names:
                    to_delete.append(file)

        for file in to_delete:
            file.unlink()

        print(f"Delete bake file: {len(to_delete)} files deleted")

        return len(to_delete)
