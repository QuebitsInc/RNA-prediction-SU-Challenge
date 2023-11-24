def flatten(o):
    for item in o:
        if isinstance(o, dict):
            yield o[item]
            continue
        elif isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def dict_to(x, device='cuda'):
    return {k: x[k].to(device) for k in x}


def to_device(x, device='cuda'):
    return tuple(dict_to(e, device) for e in x)
