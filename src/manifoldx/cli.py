"""The manifoldx command.

`manifoldx run` runs an app and reloads it in the same window whenever its source changes.
`manifoldx export` runs an app and writes its scene to glTF instead of a window.

    manifoldx run app.py [-- <app args>]
    manifoldx run -m package.module [-- <app args>]
    manifoldx export app.py scene.glb [--strict] [-- <app args>]
    manifoldx export -m package.module scene.glb [--strict] [-- <app args>]
"""

import argparse
import os
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
    run = sub.add_parser("run", help="run an app and reload it in the same window when its source changes")
    run.add_argument("-m", dest="module", help="run this module as __main__, like python -m")
    run.add_argument("script", nargs="?", metavar="app.py")
    args = parser.parse_args(argv)
    if args.command == "run":
        if (args.module is None) == (args.script is None):
            parser.error("give app.py or -m module")
        from manifoldx.hot import HotReloader

        HotReloader(script=args.script, module=args.module, argv=app_args).run()
        return 0
    if len(args.paths) != (1 if args.module else 2):
        parser.error("give app.py and out.glb, or -m module and out.glb")
    return export(args.module, None if args.module else args.paths[0], args.paths[-1], args.strict, app_args)


def export(module, script, output, strict, app_args):
    from manifoldx.engine import Engine
    from manifoldx.gltf import ExportIncomplete

    exported = []

    def run(engine):
        exported.append(engine.export_gltf(output, strict=strict))

    original, saved_argv, saved_path = Engine.run, sys.argv, list(sys.path)
    Engine.run = run
    sys.argv = [module or script, *app_args]
    # What python itself puts first on sys.path: the script's folder, or the working directory for -m.
    sys.path.insert(0, os.getcwd() if module else os.path.dirname(os.path.abspath(script)))
    try:
        if module:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(script, run_name="__main__")
    except ExportIncomplete:
        return 1
    finally:
        Engine.run, sys.argv = original, saved_argv
        sys.path[:] = saved_path
    if not exported:
        print(f"manifoldx export: {module or script} finished without calling engine.run(); nothing written",
              file=sys.stderr)
        return 2
    return 0
