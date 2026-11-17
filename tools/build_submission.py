"""Build the current two-file racing submission, without local tools or logs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate design key: '+key)
        result[key]=value
    return result


def reject_constant(value):
    raise ValueError('Non-finite JSON value: '+value)


def build(output,manifest_path,agent_dir=None):
    agent_dir = agent_dir or ROOT/'agent'
    files={name:(agent_dir/name).read_bytes() for name in ('agent.py','design.json')}
    ast.parse(files['agent.py'].decode('utf-8'))
    design=json.loads(files['design.json'],object_pairs_hook=unique_object,parse_constant=reject_constant)
    if 'design' not in design or set(design)-{'design','rain_design'}:
        raise ValueError('Expected design and optional rain_design')
    if any(len(raw)>256*1024 for raw in files.values()) or sum(map(len,files.values()))>1024*1024:
        raise ValueError('Source size limit')
    if len(files['design.json'])>8320:
        raise ValueError('Design envelope size limit')
    for book in design.values():
        if len(json.dumps(book,ensure_ascii=False,separators=(',',':')).encode())>4096:
            raise ValueError('Design book size limit')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,raw in files.items():
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;archive.writestr(info,raw)
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist()==list(files) and archive.testzip() is None
        assert all(archive.read(name)==raw for name,raw in files.items())
    manifest=dict(zip=output.name,zip_bytes=output.stat().st_size,
                  zip_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  files={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in files.items()},
                  validation='Local syntax, JSON, size, archive CRC and byte equality only; platform admission pending.')
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--agent-dir',type=Path,help='Standalone agent folder; defaults to agent/')
    args=parser.parse_args();build(args.output,args.manifest,args.agent_dir)
