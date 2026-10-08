"""The manifoldx command. `manifoldx export` runs an app and writes its scene to glTF instead of a window.

    manifoldx export app.py scene.glb [--strict] [-- <app args>]
    manifoldx export -m package.module scene.glb [--strict] [-- <app args>]
"""

import argparse
import runpy
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app_args = []
    if "--" in argv:
        cut = argv.index("--")
        argv, app_args = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(prog="manifoldx")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="run an app and write its scene to a .glb instead of opening a window")
    ex.add_argument("-m", dest="module", help="run this module as __main__, like python -m")
    ex.add_argument("paths", nargs="+", metavar="[app.py] out.glb")
    ex.add_argument("--strict", action="store_true", help="exit with code 1 and write nothing if anything is left out")
    args = parser.parse_args(argv)
    if len(args.paths) != (1 if args.module else 2):
        parser.error("give app.py and out.glb, or -m module and out.glb")
    return export(args.module, None if args.module else args.paths[0], args.paths[-1], args.strict, app_args)


def export(module, script, output, strict, app_args):
    from manifoldx.engine import Engine
    from manifoldx.gltf import ExportIncomplete

    exported = []

    def run(engine):
        exported.append(engine.export_gltf(output, strict=strict))

    original, saved_argv = Engine.run, sys.argv
    Engine.run = run
    sys.argv = [module or script, *app_args]
    try:
        if module:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(script, run_name="__main__")
    except ExportIncomplete:
        return 1
    finally:
        Engine.run, sys.argv = original, saved_argv
    if not exported:
        print(f"manifoldx export: {module or script} finished without calling engine.run(); nothing written",
              file=sys.stderr)
        return 2
    return 0
