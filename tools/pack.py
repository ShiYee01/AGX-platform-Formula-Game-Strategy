"""Pack a submission with Python, root design.json and numeric NPZ only.

打包提交物：只含 Python、根目录 design.json 和数值 NPZ。Requires NumPy.
"""
import argparse
import io
from pathlib import Path
import stat
import zipfile

import numpy as np

MIB = 1024 ** 2


def pack(root, output):
    """Write the kit's agent and maps with bounded entries, sizes and tensors.

    Args:
        root: Extracted kit directory.
        output: Submission ZIP destination.
    """
    files = {p.name: p.read_bytes() for p in sorted((root/'agent').iterdir())
             if p.suffix == '.py' or p.name == 'design.json'}
    files.update({'weights/'+p.name: p.read_bytes() for p in sorted((root/'data').glob('*.npz'))})
    if 'agent.py' not in files or not 1 <= len(files) <= 64:
        raise ValueError('Need agent.py and at most 64 entries / 需要 agent.py，最多 64 个条目')
    source = weights = tensors = 0
    for name, raw in files.items():
        if name.endswith('.py') or name == 'design.json':
            raw.decode('utf-8')
            if len(raw) > 256*1024:
                raise ValueError('Source file exceeds 256 KiB')
            source += len(raw)
        else:
            weights += len(raw)
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                if len(arrays.files) > 64:
                    raise ValueError('More than 64 tensors')
                for key in arrays.files:
                    arr = arrays[key]
                    if arr.dtype.kind not in 'biuf' or arr.dtype.fields or arr.dtype.subdtype:
                        raise ValueError('Only plain numeric arrays are accepted / 只接受普通数值数组')
                    tensors += arr.nbytes
    if source > MIB or weights > 64*MIB or tensors > 64*MIB or sum(map(len, files.values())) > 128*MIB:
        raise ValueError('Submission size limit')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    if output.stat().st_size > 64*MIB:
        output.unlink()
        raise ValueError('ZIP exceeds 64 MiB')
    print(f'{output}: {len(files)} entries; {output.stat().st_size} bytes; {tensors} tensor bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('submission.zip'))
    args = parser.parse_args()
    pack(Path(__file__).resolve().parents[1], args.output.resolve())
