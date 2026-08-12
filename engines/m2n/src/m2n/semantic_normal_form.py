from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from .canonical_ir import CanonicalIR, CNode, Dim, validate_canonical


def _dim(d:Dim):
    return tuple((k,v.numerator,v.denominator) for k,v in d.powers)

def _param(v:Any):
    if isinstance(v,Dim): return ('dim',_dim(v))
    if isinstance(v,Fraction): return ('frac',v.numerator,v.denominator)
    if isinstance(v,(tuple,list)):
        return tuple(_param(x) for x in v)
    if isinstance(v,dict):
        return tuple(sorted((str(k),_param(x)) for k,x in v.items() if k not in {'share_key'}))
    if isinstance(v,float): return ('float',repr(float(v)))
    return v

@dataclass
class _Ctx:
    share_groups:dict[str,int]
    next_share:int=0
    def share(self,key:str)->int:
        if key not in self.share_groups:
            self.share_groups[key]=self.next_share; self.next_share+=1
        return self.share_groups[key]

def _neg(e):
    if e[0]=='neg': return e[1]
    if e[0]=='const_zero': return e
    return ('neg',e)

def _add_terms(e):
    return list(e[1]) if e[0]=='add' else [e]

def _add(a,b):
    terms=_add_terms(a)+_add_terms(b)
    terms=[t for t in terms if t[0] != 'const_zero']
    # cancel exact t + (-t) pairs conservatively
    pool=[]
    for t in terms:
        inv=_neg(t)
        if inv in pool:
            pool.remove(inv)
        else:
            pool.append(t)
    if not pool:return ('const_zero',)
    if len(pool)==1:return pool[0]
    return ('add',tuple(sorted(pool,key=repr)))

def _node_expr(n:CNode,xs:list,ctx:_Ctx):
    p=n.params
    if n.op=='add': return _add(xs[0],xs[1])
    if n.op=='subtract': return _add(xs[0],_neg(xs[1]))
    if n.op=='negate': return _neg(xs[0])
    if n.op=='zero_like': return ('const_zero',)
    if n.op=='positive_gain':
        pp=tuple(sorted((k,_param(v)) for k,v in p.items() if k not in {'init'}))
        # init does not alter architecture semantics for a trainable slot; fixed value does.
        if not p.get('learnable',False):
            pp=tuple(sorted((k,_param(v)) for k,v in p.items()))
        return ('positive_gain',pp,xs[0])
    if n.op=='local_neural':
        key=str(p.get('share_key') or n.id)
        share=ctx.share(key)
        pp=tuple(sorted((k,_param(v)) for k,v in p.items() if k!='share_key'))
        return ('local_neural',share,pp,xs[0])
    if n.op=='sqrt_pos_diff_scaled':
        pp=tuple(sorted((k,_param(v)) for k,v in p.items()))
        return ('sqrt_pos_diff_scaled',pp,*xs)
    if n.op=='euler_state_update':
        pp=tuple(sorted((k,_param(v)) for k,v in p.items()))
        return ('euler_state_update',pp,xs[0],xs[1])
    raise KeyError(n.op)

def semantic_form(ir:CanonicalIR):
    validate_canonical(ir)
    env={v.name:('input',v.name,_dim(v.dim)) for v in ir.inputs}
    ctx=_Ctx({})
    for n in ir.nodes:
        env[n.id]=_node_expr(n,[env[k] for k in n.inputs],ctx)
    expected=tuple((k,_dim(d)) for k,d in ir.expected_output_dims)
    # outputs retain declared interface names; internal node IDs disappear.
    outs=tuple((name,env[name]) for name in ir.outputs)
    return ('cmir-semantic-v1',tuple((v.name,_dim(v.dim)) for v in ir.inputs),outs,expected)

def semantic_signature(ir:CanonicalIR)->str:
    payload=repr(semantic_form(ir)).encode()
    return hashlib.sha256(payload).hexdigest()
