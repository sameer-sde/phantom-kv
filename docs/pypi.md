# Phantom: deterministic testing for LLM-serving KV caches

Phantom finds **cross-request data leaks and KV-cache corruption** in LLM
inference engines, exactly, on a laptop CPU, with no GPU and no model
weights. It has been tested on vLLM and LMCache.

Phantom is a model with no weights. To the serving engine it looks like a
real LLM: it is scheduled, batched, preempted, prefix-cached and offloaded.
But instead of K/V vectors it writes a **provenance tag** (writer, position,
prefix chain hash) into every KV slot, and checks every slot it reads. Two
slots are interchangeable exactly when their chain hashes match, which is
the rule prefix caching is allowed to exploit, so correct sharing never
raises an alarm. One seed replays a whole run event for event, and failures
shrink automatically to a handful of events.

## Install and try

```bash
pip install phantom-kv

# A correct engine: expect "no violations found"
phantomkv fuzz --seeds 1000

# A planted bug (free blocks on abort while KV writes are in flight): caught
phantomkv fuzz --seeds 1000 --plant abort-free-race

# Shrink a failing run to a minimal reproduction (53 events -> 6)
phantomkv shrink --seed 0 --plant abort-free-race
```

The simulator needs nothing but Python 3.10+. Running Phantom inside real
vLLM and LMCache uses the scripts in the repository.

## Results

- **Real bugs:** 2 in LMCache, found independently (a fast server restart
  is never noticed by vLLM; aborted requests leak read locks for 300 s).
  Both were already reported upstream; Phantom's A/B tests show both open
  fixes work.
- **Planted bugs:** caught in every class tried, in vLLM's block pool,
  between two vLLM engines, and in LMCache's own code.
- **False alarms:** 0 on unmodified vLLM and LMCache; every seed replays
  identically.
- **Unplanted code:** no bugs found in vLLM's two CPU offloading systems.

## Links

- Repository, with diagrams and full results:
  https://github.com/sameer-sde/phantom-kv
- Blog post: https://github.com/sameer-sde/phantom-kv/blob/main/docs/blog/phantom.md
- Paper (PDF): https://github.com/sameer-sde/phantom-kv/blob/main/docs/paper/phantom.pdf
- Changelog: https://github.com/sameer-sde/phantom-kv/blob/main/CHANGELOG.md

MIT licensed.
