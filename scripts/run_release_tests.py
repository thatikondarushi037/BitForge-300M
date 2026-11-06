"""Run actual unittest suites and save counts; expensive tests are opt-in."""
import argparse
import io
import json
from pathlib import Path
import time
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-full',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output already exists')
    root=Path(__file__).resolve().parents[1]
    rows=[]
    for name,directory in [('ordinary',root/'tests')]+([('full-model',root/'tests/expensive')] if args.include_full else []):
        stream=io.StringIO()
        suite=unittest.TestLoader().discover(str(directory))
        start=time.perf_counter()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        row={'suite':name,'run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
             'failed':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'seconds':time.perf_counter()-start,'log':stream.getvalue(),
             'failures':[{'test':str(test),'traceback':trace} for test,trace in result.failures+result.errors],
             'skips':[{'test':str(test),'reason':reason} for test,reason in result.skipped]}
        rows.append(row)
        print(stream.getvalue(),end='')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:json.dump({'suites':rows},output,indent=2)
    return int(any(r['failed'] or r['errors'] for r in rows))


if __name__=='__main__':raise SystemExit(main())
