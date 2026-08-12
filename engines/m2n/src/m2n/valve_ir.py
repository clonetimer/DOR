from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction

from .canonical_ir import Dim, DIMLESS, VarSpec, CNode, CanonicalIR, compile_canonical


@dataclass(frozen=True)
class FourWayValveIR:
    """Typed four-edge directional-valve metering map.

    This IR is deliberately local and static: it maps measured/instrumented spool
    position and four port pressures to the four metering-edge flows.  It does not
    model spool actuator dynamics, line inertance, chamber compliance or pump
    dynamics.  Those remain separate mechanism slots.

    ``law`` is a finite K2 choice:
      - ``linear_dp``: q ~ opening * dp
      - ``sqrt_orifice``: q ~ opening * signed_sqrt(dp)

    Edge overlap values are fixed engineering declarations for a candidate; edge
    gains are K1-positive parameters.  A common known leakage-area flow is retained
    in both candidates so K2 comparison isolates pressure-law structure.
    """
    spool_dim: Dim
    pressure_dim: Dim
    flow_dim: Dim
    law: str = 'sqrt_orifice'
    max_spool: float = 4.0e-4
    overlaps: tuple[float,float,float,float] = (0.0,0.0,0.0,0.0)  # PA,PB,AT,BT
    gain_inits: tuple[float,float,float,float] = (1.0e-8,1.0e-8,1.0e-8,1.0e-8)
    leakage_coeff: float = 0.0
    metadata: tuple[tuple[str,str], ...] = ()


def validate_four_way_valve_ir(ir: FourWayValveIR) -> bool:
    if ir.law not in ('linear_dp','sqrt_orifice'):
        raise ValueError(f'unsupported four-way valve law {ir.law!r}')
    if ir.max_spool <= 0:
        raise ValueError('max_spool must be positive')
    if len(ir.overlaps)!=4 or len(ir.gain_inits)!=4:
        raise ValueError('four edge overlaps/gains are required')
    if any(x < 0 for x in ir.overlaps):
        raise ValueError('this application IR expects non-negative overlaps')
    if any(x <= 0 for x in ir.gain_inits):
        raise ValueError('gain initializations must be positive')
    if ir.leakage_coeff < 0:
        raise ValueError('leakage_coeff must be non-negative')
    return True


def lower_four_way_valve_ir(ir: FourWayValveIR) -> CanonicalIR:
    validate_four_way_valve_ir(ir)
    L=ir.spool_dim; P=ir.pressure_dim; Q=ir.flow_dim
    inputs=(VarSpec('spool',L),VarSpec('pP',P),VarSpec('pT',P),VarSpec('pA',P),VarSpec('pB',P))
    nodes=[]
    # Positive / negative metering directions.
    nodes.append(CNode('spool_pos','clamp',('spool',),{'min_value':0.0,'max_value':float(ir.max_spool)}))
    nodes.append(CNode('spool_neg_raw','negate',('spool',),{}))
    nodes.append(CNode('spool_neg','clamp',('spool_neg_raw',),{'min_value':0.0,'max_value':float(ir.max_spool)}))

    edge_defs=(
        ('pa','spool_pos','pP','pA',ir.overlaps[0],ir.gain_inits[0]),
        ('pb','spool_neg','pP','pB',ir.overlaps[1],ir.gain_inits[1]),
        ('at','spool_neg','pA','pT',ir.overlaps[2],ir.gain_inits[2]),
        ('bt','spool_pos','pB','pT',ir.overlaps[3],ir.gain_inits[3]),
    )
    outputs=[]; expected=[]
    for eid,opening,up,dn,overlap,ginit in edge_defs:
        ov=f'{eid}__overlap'
        nodes.append(CNode(ov,'signed_constant_like',(opening,),{'learnable':False,'value':float(overlap),'output_dim':L}))
        raw=f'{eid}__opening_raw'
        nodes.append(CNode(raw,'subtract',(opening,ov),{}))
        eff=f'{eid}__opening'
        nodes.append(CNode(eff,'clamp',(raw,),{'min_value':0.0,'max_value':float(ir.max_spool)}))

        if ir.law=='linear_dp':
            dp=f'{eid}__dp'; nodes.append(CNode(dp,'subtract',(up,dn),{}))
            feat=f'{eid}__meter_feat'; nodes.append(CNode(feat,'multiply',(eff,dp),{}))
            gain_dim=Q/(L*P)
        else:
            feat=f'{eid}__meter_feat'
            nodes.append(CNode(feat,'signed_sqrt_diff_scaled',(up,dn,eff),{
                'coeff':1.0,'coeff_dim':DIMLESS,'sqrt_eps':0.0}))
            gain_dim=Q/(L*(P**Fraction(1,2)))
        meter=f'{eid}__meter_q'
        nodes.append(CNode(meter,'positive_gain',(feat,),{
            'learnable':True,'init':float(ginit),'gain_dim':gain_dim,'eps':1e-16}))

        if ir.leakage_coeff>0:
            leak=f'{eid}__leak_q'
            nodes.append(CNode(leak,'signed_sqrt_diff_scaled',(up,dn),{
                'coeff':float(ir.leakage_coeff),'coeff_dim':Q/(P**Fraction(1,2)),'sqrt_eps':0.0}))
            out=f'q_{eid}'; nodes.append(CNode(out,'add',(meter,leak),{}))
        else:
            out=f'q_{eid}'
            nodes.append(CNode(out,'positive_gain',(meter,),{'learnable':False,'value':1.0,'gain_dim':DIMLESS}))
        outputs.append(out); expected.append((out,Q))
    return CanonicalIR(inputs,tuple(nodes),tuple(outputs),tuple(expected),
                       (('source_ir','four_way_valve'),('law',ir.law))+tuple(ir.metadata))


def compile_four_way_valve_ir(ir: FourWayValveIR):
    cir=lower_four_way_valve_ir(ir)
    return compile_canonical(cir),cir
