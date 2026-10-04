"""Classify a recorded video as burglary, normal or vandalism."""
import argparse
import json
from pathlib import Path

import torch

from trace.crime import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkpoint', default='runs/crime_baseline/best.pt')
    parser.add_argument('--encoder', default='models/r3d_18-b3b3357e.pth')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', help='Optional new JSON file')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = predict(args.source, args.checkpoint, args.encoder, args.device)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
