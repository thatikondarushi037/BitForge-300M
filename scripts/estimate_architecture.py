"""Report architecture arithmetic only: no model construction or CUDA calls."""

import json

from bitforge.config import architecture_summary, target_300m_config, tiny_config


if __name__ == '__main__':
    print(json.dumps({'tiny': architecture_summary(tiny_config()),
                      'target_300m': architecture_summary(target_300m_config())}, indent=2))
