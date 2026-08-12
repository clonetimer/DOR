from __future__ import annotations
from dataclasses import dataclass
from .canonical_ir import (
    Dim, TIME, VarSpec, CNode, CanonicalIR,
    CanonicalStateWrapper, compile_canonical, validate_canonical,
)


@dataclass(frozen=True)
class CoordinateSpec:
    position: str
    velocity: str
    position_dim: Dim

    @property
    def velocity_dim(self) -> Dim:
        return self.position_dim / TIME

    @property
    def acceleration_dim(self) -> Dim:
        return self.position_dim / (TIME ** 2)


@dataclass(frozen=True)
class InputSpec:
    name: str
    dim: Dim


@dataclass(frozen=True)
class DriveSpec:
    id: str
    input: str
    gain_dim: Dim
    init: float = 1.0
    law: str = 'gain'  # gain | local_neural
    hidden: int = 1
    share_key: str | None = None


@dataclass(frozen=True)
class ConstantLoadSpec:
    """Signed constant term in the acceleration balance (gravity/bias/preload)."""
    id: str = 'load'
    init: float = 0.0


@dataclass(frozen=True)
class ResistanceSpec:
    """Local signed constitutive resistance in the acceleration balance.

    ``viscous``:          a_r = c_v * v
    ``viscous_coulomb``: a_r = c_v * v + c_c * sign(v)
    ``asymmetric_viscous_coulomb``: direction-specific viscous/Coulomb resistance
    ``stribeck``:         viscous + Coulomb/breakaway Stribeck resistance
    ``local_neural``:     a_r = sign(v)*positive_NN(|v|,sign(v)), dissipative

    The outer mechanics topology is always q_dot=v, v_dot=drive-resistance.
    This keeps K3 local: only the constitutive resistance is neuralized.
    """
    id: str
    law: str  # viscous | viscous_coulomb | asymmetric_viscous_coulomb | local_neural
    viscous_gain_init: float = 0.1
    coulomb_gain_init: float = 0.1
    viscous_pos_init: float = 0.1
    viscous_neg_init: float = 0.1
    coulomb_pos_init: float = 0.1
    coulomb_neg_init: float = 0.1
    breakaway_delta_init: float = 0.5
    stribeck_velocity_init: float = 0.03
    smoothing_velocity_init: float = 0.001
    hidden: int = 1
    share_key: str | None = None


@dataclass(frozen=True)
class SecondOrderMechanicsIR:
    coordinate: CoordinateSpec
    inputs: tuple[InputSpec, ...]
    drive: DriveSpec
    resistance: ResistanceSpec
    dt: float
    constant_load: ConstantLoadSpec | None = None
    metadata: tuple[tuple[str, str], ...] = ()


def validate_second_order_ir(ir: SecondOrderMechanicsIR) -> bool:
    if ir.dt <= 0:
        raise ValueError('dt must be positive')
    names = {x.name for x in ir.inputs}
    if len(names) != len(ir.inputs):
        raise ValueError('duplicate input names')
    if ir.coordinate.position == ir.coordinate.velocity or ir.coordinate.position in names or ir.coordinate.velocity in names:
        raise ValueError('duplicate state/input names')
    if ir.drive.input not in names:
        raise ValueError('drive references unavailable input')
    if ir.drive.law not in ('gain','local_neural'):
        raise ValueError(f'unsupported drive law {ir.drive.law}')
    idim = next(x.dim for x in ir.inputs if x.name == ir.drive.input)
    if idim * ir.drive.gain_dim != ir.coordinate.acceleration_dim:
        raise ValueError('drive dimension mismatch')
    if ir.resistance.law not in ('viscous', 'viscous_coulomb', 'asymmetric_viscous_coulomb', 'stribeck', 'local_neural'):
        raise ValueError(f'unsupported resistance law {ir.resistance.law}')
    return True


def lower_second_order_ir(ir: SecondOrderMechanicsIR):
    validate_second_order_ir(ir)
    q = ir.coordinate.position
    v = ir.coordinate.velocity
    qdim = ir.coordinate.position_dim
    vdim = ir.coordinate.velocity_dim
    adim = ir.coordinate.acceleration_dim
    inp = tuple([VarSpec(q, qdim), VarSpec(v, vdim)] + [VarSpec(x.name, x.dim) for x in ir.inputs])
    nodes: list[CNode] = []

    drive = f'{ir.drive.id}__accel'
    if ir.drive.law == 'gain':
        nodes.append(CNode(drive, 'positive_gain', (ir.drive.input,), {
            'learnable': True,
            'init': float(ir.drive.init),
            'gain_dim': ir.drive.gain_dim,
            'eps': 1e-8,
        }))
    else:
        nodes.append(CNode(drive, 'signed_gain_local_neural', (ir.drive.input,), {
            'output_dim': adim,
            'hidden': int(ir.drive.hidden),
            'share_key': ir.drive.share_key,
        }))

    r = ir.resistance
    if r.law == 'viscous':
        resist = f'{r.id}__viscous'
        nodes.append(CNode(resist, 'positive_gain', (v,), {
            'learnable': True,
            'init': float(r.viscous_gain_init),
            'gain_dim': adim / vdim,
            'eps': 1e-8,
        }))
    elif r.law == 'viscous_coulomb':
        visc = f'{r.id}__viscous'
        sgn = f'{r.id}__sign'
        coul = f'{r.id}__coulomb'
        resist = f'{r.id}__total'
        nodes.append(CNode(visc, 'positive_gain', (v,), {
            'learnable': True,
            'init': float(r.viscous_gain_init),
            'gain_dim': adim / vdim,
            'eps': 1e-8,
        }))
        nodes.append(CNode(sgn, 'sign', (v,), {}))
        nodes.append(CNode(coul, 'positive_gain', (sgn,), {
            'learnable': True,
            'init': float(r.coulomb_gain_init),
            'gain_dim': adim,
            'eps': 1e-8,
        }))
        nodes.append(CNode(resist, 'add', (visc, coul), {}))
    elif r.law == 'asymmetric_viscous_coulomb':
        # Decompose velocity/sign into positive and negative signed parts using
        # generic algebra; no device-specific conditional operator is introduced.
        av=f'{r.id}__abs_v'; vp0=f'{r.id}__v_plus_sum'; vn0=f'{r.id}__v_minus_sum'
        vp=f'{r.id}__v_plus'; vn=f'{r.id}__v_minus'
        sgn=f'{r.id}__sign'; asgn=f'{r.id}__abs_sign'
        ip0=f'{r.id}__i_plus_sum'; im0=f'{r.id}__i_minus_sum'; ip=f'{r.id}__i_plus'; im=f'{r.id}__i_minus'
        nodes += [
            CNode(av,'abs',(v,),{}), CNode(vp0,'add',(v,av),{}), CNode(vn0,'subtract',(v,av),{}),
            CNode(vp,'positive_gain',(vp0,),{'learnable':False,'value':0.5,'gain_dim':Dim()}),
            CNode(vn,'positive_gain',(vn0,),{'learnable':False,'value':0.5,'gain_dim':Dim()}),
            CNode(sgn,'sign',(v,),{}), CNode(asgn,'abs',(sgn,),{}),
            CNode(ip0,'add',(sgn,asgn),{}), CNode(im0,'subtract',(sgn,asgn),{}),
            CNode(ip,'positive_gain',(ip0,),{'learnable':False,'value':0.5,'gain_dim':Dim()}),
            CNode(im,'positive_gain',(im0,),{'learnable':False,'value':0.5,'gain_dim':Dim()}),
        ]
        vp_r=f'{r.id}__visc_pos'; vn_r=f'{r.id}__visc_neg'; cp=f'{r.id}__coul_pos'; cm=f'{r.id}__coul_neg'
        for nid,src,init,gdim in [
            (vp_r,vp,r.viscous_pos_init,adim/vdim),(vn_r,vn,r.viscous_neg_init,adim/vdim),
            (cp,ip,r.coulomb_pos_init,adim),(cm,im,r.coulomb_neg_init,adim)]:
            nodes.append(CNode(nid,'positive_gain',(src,),{'learnable':True,'init':float(init),'gain_dim':gdim,'eps':1e-8}))
        vis=f'{r.id}__visc_total'; cou=f'{r.id}__coul_total'; resist=f'{r.id}__total'
        nodes += [CNode(vis,'add',(vp_r,vn_r),{}),CNode(cou,'add',(cp,cm),{}),CNode(resist,'add',(vis,cou),{})]
    elif r.law == 'stribeck':
        resist=f'{r.id}__stribeck'
        nodes.append(CNode(resist,'stribeck_resistance',(v,),{
            'output_dim':adim,
            'viscous_init':float(r.viscous_gain_init),
            'coulomb_init':float(r.coulomb_gain_init),
            'breakaway_delta_init':float(r.breakaway_delta_init),
            'stribeck_velocity_init':float(r.stribeck_velocity_init),
            'smoothing_velocity_init':float(r.smoothing_velocity_init),
            'share_key':r.share_key,
        }))
    else:
        resist = f'{r.id}__local'
        nodes.append(CNode(resist, 'dissipative_local_neural', (v,), {
            'output_dim': adim,
            'hidden': int(r.hidden),
            'share_key': r.share_key,
        }))

    total_resist=resist
    if ir.constant_load is not None:
        load=f'{ir.constant_load.id}__constant'
        nodes.append(CNode(load,'signed_constant_like',(v,),{
            'learnable':True,'init':float(ir.constant_load.init),'output_dim':adim,
        }))
        total_resist=f'{r.id}__with_load'
        nodes.append(CNode(total_resist,'add',(resist,load),{}))
    acc = f'{v}__rate'
    nodes.append(CNode(acc, 'subtract', (drive, total_resist), {}))
    qrate = f'{q}__rate'
    nodes.append(CNode(qrate, 'positive_gain', (v,), {
        'learnable': False,
        'value': 1.0,
        'gain_dim': Dim(),
    }))
    qnext = f'{q}__next'
    vnext = f'{v}__next'
    nodes.append(CNode(qnext, 'euler_state_update', (q, qrate), {'dt': float(ir.dt), 'dt_dim': TIME}))
    nodes.append(CNode(vnext, 'euler_state_update', (v, acc), {'dt': float(ir.dt), 'dt_dim': TIME}))

    cir = CanonicalIR(
        inputs=inp,
        nodes=tuple(nodes),
        outputs=(qnext, vnext),
        expected_output_dims=((qnext, qdim), (vnext, vdim)),
        metadata=(('source_ir', 'second_order_mechanics'), ('dt', repr(float(ir.dt)))) + tuple(ir.metadata),
    )
    validate_canonical(cir)
    return cir, (q, v), (qrate, acc)


def compile_second_order_ir(ir: SecondOrderMechanicsIR):
    cir, names, rate_nodes = lower_second_order_ir(ir)
    return CanonicalStateWrapper(compile_canonical(cir), names, rate_nodes), cir
