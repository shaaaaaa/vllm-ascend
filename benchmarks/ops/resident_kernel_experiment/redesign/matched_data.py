# SPDX-License-Identifier: Apache-2.0
"""CPU-only input contract for the matched registered-host comparison."""
import hashlib
import torch


def validate_trace(trace):
    initial, steps = trace['initial_tokens'], trace['steps']
    if initial.device.type != 'cpu' or steps.device.type != 'cpu':
        raise ValueError('trace must be CPU tensors')
    if initial.dtype != torch.int32 or steps.dtype != torch.int32 or steps.ndim != 4:
        raise ValueError('int32 initial_tokens[R,N] and steps[S,R,Q,K] required')
    s, r, q, k = steps.shape
    if min(s, r, k) < 1 or q not in (1, 2) or initial.shape != (r, q*k):
        raise ValueError('trace geometry mismatch')
    for name in ('boundary', 'lengths'):
        if trace[name].dtype != torch.int32 or trace[name].shape != (r,):
            raise ValueError(f'{name} must be int32[R]')
    boundary, lengths = trace['boundary'], trace['lengths']
    if torch.any(boundary < q*k) or torch.any(lengths < boundary) or torch.any(lengths > 1048576):
        raise ValueError('prefix must fit initial cache and lengths must cover prefix')
    ready = trace['initial_ready']
    if ready.dtype != torch.bool or ready.shape != initial.shape:
        raise ValueError('initial_ready must be bool[R,N]')
    for request in range(r):
        if initial[request].unique().numel() != q*k:
            raise ValueError('initial resident tokens must be unique')
        if torch.any(initial[request] < 0) or torch.any(initial[request] >= boundary[request]):
            raise ValueError('initial tokens must belong to the prefix')
        selected = steps[:, request]
        if torch.any(selected < -1) or torch.any(selected >= lengths[request]):
            raise ValueError('selections must be -1 padding or below the valid length')
    return trace


def make_trace(requests=8, steps=4, queries=2, topk=2048, prefix=131072,
               tail=128, overlap=1024, hit_rate=.9, scenario='rank_shift', seed=7):
    if not 0 <= overlap <= topk or not 0 <= hit_rate <= 1 or scenario not in ('stable', 'rank_shift', 'permuted', 'cold'):
        raise ValueError('invalid workload controls')
    rng = torch.Generator().manual_seed(seed)
    initial = torch.stack([torch.randperm(prefix, generator=rng)[:queries*topk] for _ in range(requests)]).int()
    current = initial.reshape(requests, queries, topk).clone()
    stream = []
    for _ in range(steps):
        if scenario == 'rank_shift':
            current = current.roll(2, -1)
        elif scenario == 'permuted':
            current = torch.stack([torch.stack([row[torch.randperm(topk, generator=rng)] for row in req]) for req in current])
        replace = torch.rand(current.shape, generator=rng) > hit_rate
        current = torch.where(replace, torch.randint(prefix + tail, current.shape, generator=rng).int(), current)
        if queries == 2:
            current[:, 1, :overlap] = current[:, 0, :overlap]
        stream.append(current.clone())
    return validate_trace(dict(initial_tokens=initial, steps=torch.stack(stream),
                               initial_ready=torch.full_like(initial, scenario != 'cold', dtype=torch.bool),
                               boundary=torch.full((requests,), prefix, dtype=torch.int32),
                               lengths=torch.full((requests,), prefix+tail, dtype=torch.int32)))


def trace_digest(trace):
    digest = hashlib.sha256()
    for name in sorted(trace):
        tensor = trace[name].contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def payload(tokens, request, width, plane):
    """Deterministic, exactly representable BF16 values without a dense HBM source."""
    values = (tokens.long()[..., None] * 1103515245 + torch.arange(width) * 12345
              + request * 2654435761 + plane * 97531)
    return (((values ^ (values >> 16)) % 256).float() / 128).to(torch.bfloat16)


def initial_slots(trace, block_size=128):
    r, n = trace['initial_tokens'].shape
    if n % block_size:
        raise ValueError('resident capacity must be block aligned')
    pages = torch.stack([torch.randperm(n // block_size, generator=torch.Generator().manual_seed(101+i)) for i in range(r)]).int()
    logical = torch.arange(n)
    destination = (pages[:, logical // block_size].long() * block_size + logical % block_size).contiguous()
    positions = trace['initial_ready'].long().cumsum(-1).sub(1).clamp_min(0)
    source = destination.gather(1, positions)
    return pages, source, destination
