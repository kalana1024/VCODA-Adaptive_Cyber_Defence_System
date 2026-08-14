import argparse
from pprint import pprint
from vcoda.capture.pcap import analyse_pcap

if __name__ == "__main__":
 p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",default="outputs/pcap_analysis.json"); a=p.parse_args(); pprint(analyse_pcap(a.input,a.output))
