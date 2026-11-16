"""Read all map fields from numeric NPZ without pickle / 安全读取全部地图字段。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(path):
    """Recover the JSON-equivalent names, geometry, units and coordinate fields.

    Args:
        path: A kit mapdata NPZ path.
    """
    with np.load(path, allow_pickle=False) as data:
        structure=json.loads(data['structure_utf8'].tobytes().decode('utf-8'))
        pools={kind:data[kind] for kind in 'ifb'}
    def unpack(node):
        tag=node[0]
        if tag=='n':
            return None
        if tag=='s':
            return node[1]
        if tag=='d':
            return {key:unpack(value) for key,value in node[1]}
        if tag=='l':
            return [unpack(value) for value in node[1]]
        _,kind,offset,shape=node
        count=math.prod(shape) if shape else 1
        return pools[kind][offset:offset+count].reshape(shape).tolist()
    return unpack(structure)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('npz',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=load(args.npz)
    if args.output:
        args.output.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(schema_version=result['schema_version'],
                         name=result.get('name',result.get('display_name')),fields=list(result)),ensure_ascii=False))
