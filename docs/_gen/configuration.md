# Configuration

Every analysis constant lives in one typed, validated object. The defaults are
the frozen dataclasses in [`secom.config`][secom.config], and
`config/default.yaml` mirrors them exactly (a test keeps the two identical), so
the YAML file documents the analysis and the dataclasses are the schema.

```bash
secom run                       # reads config/default.yaml when it exists
secom run --config my.yaml      # rerun under different choices
```

A custom file only needs the keys it changes. Unknown keys and wrong types are
errors, not silent fallbacks, so a typo in a threshold cannot quietly revert to
the default. The decision numbers in the comments (D2, D5, ...) refer to the
[decision log](decisions.md).

## Core settings

```yaml title="config/default.yaml"
{{DEFAULT_YAML}}
```

## Stage sections

A stage that needs its own settings registers a section under a top-level YAML
key instead of editing `secom.config`, so stages developed in parallel never
conflict there:

```python
from dataclasses import dataclass

from secom.config import register_section


@register_section("mspc")
@dataclass(frozen=True)
class MSPCConfig:
    #: cumulative baseline variance the retained components must explain
    variance_target: float = 0.80
```

The stage reads it with `ctx.config.section(MSPCConfig)`. When the YAML file has
no block for the key, the dataclass defaults apply, so existing config files keep
working. Sections register on import, and `load_config` imports every stage
module first. Put the section in the stage module itself, or in a module the
stage imports.

The sections registered today, read from source when this site was built:

{{SECTIONS}}
