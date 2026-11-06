"""Offline application: python -m bitforge --help."""
import argparse
import json
import pickle
from pathlib import Path
import sys

import torch

from .inference import benchmark_model, generate, load_local_checkpoint, freeze_quantized_projections, execution_preflight
from .execution import autocast_context


def main(argv=None):
    parser = argparse.ArgumentParser(description='BitForge offline ternary Transformer research CLI')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('generate', 'model-info', 'benchmark', 'verify-checkpoint', 'demo'):
        sub = commands.add_parser(name)
        sub.add_argument('--checkpoint', type=Path, required=True)
        sub.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
        sub.add_argument('--threads', type=int, default=4)
        sub.add_argument('--precision', default='fp32', choices=('fp32', 'bf16'))
        sub.add_argument('--json', action='store_true', help='machine-readable output')
        sub.add_argument('--verbose', action='store_true')
        if name in ('generate', 'demo'):
            sub.add_argument('--prompt', default='The technician will ')
            sub.add_argument('--max-new-tokens', type=int, default=32)
            sub.add_argument('--temperature', type=float, default=0.)
            sub.add_argument('--top-k', type=int, default=0)
            sub.add_argument('--top-p', type=float, default=1.)
            sub.add_argument('--seed', type=int, default=42)
        if name in ('benchmark', 'verify-checkpoint'):
            sub.add_argument('--sequence-length', type=int, default=8)
        if name == 'benchmark':
            sub.add_argument('--warmup', type=int, default=2)
            sub.add_argument('--repeats', type=int, default=5)
            sub.add_argument('--generation-tokens', type=int, default=8)
        if name == 'verify-checkpoint':
            sub.add_argument('--reference', type=Path, help='corresponding local dense/packed checkpoint')
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.threads <= 32:
            raise ValueError('threads must be 1..32')
        torch.set_num_threads(args.threads)
        model, metadata = load_local_checkpoint(args.checkpoint, args.device)
        if args.command in ('generate', 'demo'):
            result = generate(model, args.prompt, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                              seed=args.seed, precision=args.precision)
            result['model'] = metadata
            if args.command == 'demo':
                result['notice'] = ('Research demonstration; synthetic-data training does not establish general language ability. '
                                    'The 297M architecture is untrained; byte-restricted generation is only a smoke test.')
            if args.json:
                print(json.dumps(result, ensure_ascii=True))
            else:
                if args.verbose or args.command == 'demo':
                    print(json.dumps(metadata, indent=2))
                if 'notice' in result:
                    print(result['notice'])
                # Escape controls while keeping ordinary text readable on Windows.
                text = ''.join(c if c.isprintable() or c == '\n' else repr(c)[1:-1] for c in result['text'])
                print(text.encode(sys.stdout.encoding or 'utf-8', errors='backslashreplace').decode(sys.stdout.encoding or 'utf-8'))
        elif args.command == 'model-info':
            print(json.dumps(metadata, indent=2))
        elif args.command == 'benchmark':
            result = benchmark_model(model, sequence_length=args.sequence_length, warmup=args.warmup,
                                     repeats=args.repeats, precision=args.precision, generation_tokens=args.generation_tokens)
            result['model'] = metadata
            print(json.dumps(result, indent=2))
        else:
            if not 1 <= args.sequence_length <= model.config.max_seq_len:
                raise ValueError('invalid verification context')
            execution_preflight(model, args.sequence_length)
            x = torch.arange(args.sequence_length, device=args.device)[None, :] % min(256, model.config.vocab_size)
            with torch.inference_mode(), autocast_context(args.device, args.precision):
                actual = model(x).float().cpu()
            if not torch.isfinite(actual).all():
                raise ValueError('checkpoint produces nonfinite logits')
            result = {'verified': True, 'shape': list(actual.shape), 'model': metadata}
            if args.reference:
                # Release the first full model before loading the reference.
                del model
                import gc
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                source_device = metadata.get('source_quantization_device', args.device)
                reference, _ = load_local_checkpoint(args.reference, source_device)
                reference = freeze_quantized_projections(reference).to(args.device)
                with torch.inference_mode(), autocast_context(args.device, args.precision):
                    expected = reference(x).float().cpu()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                result['max_absolute_error'] = (actual-expected).abs().max().item()
            print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, AssertionError, EOFError, pickle.UnpicklingError) as error:
        print(f'bitforge: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
