"""Open a run's trajectory in rerun:  python3 scripts/view_log.py runs/<id>"""
import sys

from vesper.viz import view_run

view_run(sys.argv[1])
