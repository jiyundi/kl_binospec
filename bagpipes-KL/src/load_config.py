import yaml
import re


def load_config(filename="./config.yaml", z_spec=0):
    with open(filename, "r", encoding="utf-8") as file2:
        fit_instructions = yaml.safe_load(file2)
    fit_instructions = convert_ranges(fit_instructions)
    fit_instructions['redshift'] = z_spec
    return fit_instructions

def convert_ranges(obj):
    """
    递归转换所有形如 '(x, y)' 的字符串为数值元组 (float(x), float(y))
    """
    if isinstance(obj, dict):
        return {k: convert_ranges(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ranges(v) for v in obj]
    elif isinstance(obj, str):
        match = re.fullmatch(r'\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)', obj)
        if match:
            return (float(match.group(1)), float(match.group(2)))
        return obj
    else:
        return obj
  
