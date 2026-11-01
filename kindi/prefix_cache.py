"""Bounded host cache of complete hybrid-state snapshots. Sampling RNG is request-local."""
from collections import OrderedDict


def snapshot_bytes(snapshot):
    total=sum(array.nbytes for kind in ('states','histories','keys','values') for array in snapshot[kind].values())
    return total+(snapshot['logits'].nbytes if snapshot['logits'] is not None else 0)+len(snapshot['tokens'])*32+1024


def engine_snapshot_bytes(engine):
    """Same accounting as snapshot_bytes, using metadata before any host copy."""
    n=int(engine.position)
    if n!=engine.position or n<0 or n!=len(engine.tokens):raise ValueError('invalid checkpoint position/token count')
    total=sum(array.nbytes for kind in ('states','histories') for array in getattr(engine,kind).values())
    for kind in ('keys','values'):
        for array in getattr(engine,kind).values():
            capacity=array.shape[0]
            if capacity<1 or n>capacity or array.nbytes%capacity:
                raise ValueError('invalid checkpoint KV capacity')
            total+=n*(array.nbytes//capacity)
    return total+(engine.last_logits.nbytes if engine.last_logits is not None else 0)+len(engine.tokens)*32+1024


class PrefixCache(object):
    def __init__(self,budget_bytes,max_entries=8):
        self.budget=max(0,int(budget_bytes));self.max_entries=max(0,int(max_entries));self.entries=OrderedDict();self.used=0

    def put(self,snapshot):
        size=snapshot_bytes(snapshot)
        if not self.max_entries or size>self.budget:return False
        key=(snapshot['fingerprint'],tuple(snapshot['tokens']))
        if key in self.entries:self.used-=self.entries.pop(key)[1]
        self.entries[key]=(snapshot,size);self.used+=size
        while self.used>self.budget or len(self.entries)>self.max_entries:
            _,(_,amount)=self.entries.popitem(last=False);self.used-=amount
        return True

    def capture(self,engine):
        # Do not copy an inadmissible checkpoint just to reject it in put().
        if not self.max_entries or not self.budget or engine_snapshot_bytes(engine)>self.budget:return False
        return self.put(engine.snapshot())

    def best(self,tokens,fingerprint):
        tokens=tuple(tokens);best=None;length=-1
        for key,(snapshot,_) in self.entries.items():
            prefix=key[1]
            if key[0]==fingerprint and len(prefix)>length and len(prefix)<=len(tokens) and tokens[:len(prefix)]==prefix:
                best=key;length=len(prefix)
        if best is None:return None
        value=self.entries.pop(best);self.entries[best]=value
        return value[0]

    def clear(self):self.entries.clear();self.used=0
