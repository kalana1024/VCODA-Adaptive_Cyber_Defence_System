import argparse
from pprint import pprint
from vcoda.models.deep import train_deep

if __name__ == "__main__":
 p=argparse.ArgumentParser(); p.add_argument("--architecture",default="mlp"); p.add_argument("--task",default="binary"); a=p.parse_args(); pprint(train_deep(a.architecture,task=a.task))
