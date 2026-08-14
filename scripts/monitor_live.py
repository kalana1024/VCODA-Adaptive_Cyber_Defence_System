import argparse
from vcoda.capture.live import monitor_live

if __name__ == "__main__":
 p=argparse.ArgumentParser(); p.add_argument("--interface",required=True); a=p.parse_args(); monitor_live(a.interface)
