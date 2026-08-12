#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISGN Operator Library —— 从数学运算符升级为工程系统原语。

对应文档第 6 章：
  6.1 Algebraic      : Add / Sub / Mul / Div
  6.2 Temporal       : FilteredDerivative / LeakyIntegrator / Delay
  6.3 Stable 1st-order: Lag  (K/(tau*s+1), tau=softplus)
  6.4 2nd-order osc. : Oscillator (K wn^2/(s^2+2 z wn s + wn^2))
  6.5 Engineering NL : Saturation / DeadZone
  6.6 Balance        : 守恒/平衡残差结构模板

所有动态原语采用稳定参数化（softplus），使“稳定性”成为结构硬偏置
（文档 7.1 / 9.3）：极点实部恒为负，网络无法违反。

量纲系统（文档第 8 章）：以 [M,L,T,I,Theta] 维度向量做连接前的类型检查，
大量物理上荒谬的候选连接在训练前即被剪除。
"""
import torch
import torch.nn as nn

EPS = 1e-6

# ----------------------------------------------------------------------------
# Typed Grammar：四层原语体系（文档 §6.1 重设计）
# 原语按语义层组织，而非扁平候选列表；Balance 属 Equation-semantics 层，
# 是结构模板而非普通可搜索节点。
# ----------------------------------------------------------------------------
TYPED_GRAMMAR = {
    "Algebraic":   ["Add", "Sub", "Mul", "Div"],          # safe division
    "Dynamic":     ["Integrator", "FirstOrderLag", "SecondOrderOscillator",
                    "Delay", "FilteredDerivative"],
    "Nonlinear":   ["Saturation", "DeadZone"],
    "Equation":    ["Balance"],                            # 方程模板，非候选节点
}
# 安全除法：避免数值爆炸（文档 §7.4）
def safe_div(a, b, eps=EPS):
    return a / (eps + torch.abs(b))


# ----------------------------------------------------------------------------
# 维度系统 [M, L, T, I, Theta]
# ----------------------------------------------------------------------------
class Dim:
    VECS = {
        "none":        [0, 0, 0, 0, 0],
        "temperature": [0, 0, 0, 0, 1],
        "pressure":    [1, -1, -2, 0, 0],
        "flow":        [1, 0, -1, 0, 0],   # mass flow M/T
        "volume_flow": [0, 3, -1, 0, 0],   # volumetric flow L^3/T
        "level":       [0, 1, 0, 0, 0],
        "velocity":    [0, 1, -1, 0, 0],
        "angle":       [0, 0, 0, 0, 0],     # 弧度无量纲
        "speed":       [0, 0, -1, 0, 0],    # 角速度 rad/s = 1/T（弧度无量纲）
        "torque":      [1, 2, -2, 0, 0],
        "force":       [1, 1, -2, 0, 0],
        "power":       [1, 2, -3, 0, 0],
        "voltage":     [1, 2, -3, -1, 0],   # M L^2 T^-3 I^-1（原 -1 错放在 Θ 位）
        "acceleration":[0, 1, -2, 0, 0],   # L/T^2 振动加速度
        "mass":        [1, 0, 0, 0, 0],     # 质量 M
        "acc":         [0, 1, -2, 0, 0],   # 同上（CWRU 三通道简写）
        "current":     [0, 0, 0, 1, 0],    # 电流 I
        "displacement":[0, 1, 0, 0, 0],    # 位移 L
    }
    TIME = [0, 0, 1, 0, 0]

    @classmethod
    def get(cls, name):
        return list(cls.VECS.get(name, [0, 0, 0, 0, 0]))

    @classmethod
    def add(cls, a, b):
        return [x + y for x, y in zip(a, b)]

    @classmethod
    def sub(cls, a, b):
        return [x - y for x, y in zip(a, b)]

    @classmethod
    def eq(cls, a, b):
        return a == b


def op_dim(name, d1, d2):
    """返回算子输出维度向量；若组合物理非法返回 None（被结构性剪除）。"""
    if name in ("Add", "Sub", "Balance"):
        return d1 if Dim.eq(d1, d2) else None   # 加减/守恒要求同量纲
    if name == "Mul":
        return Dim.add(d1, d2)
    if name == "Div":
        return Dim.sub(d1, d2)
    if name == "Derivative":
        # d/dt 自动减一个时间维度
        return Dim.sub(d1, Dim.TIME)
    if name == "Integrator":
        # ∫ dt 自动加一个时间维度
        return Dim.add(d1, Dim.TIME)
    # Lag / Oscillator / Delay / Sat / DeadZone：相当于滤波/缩放，维度不变
    return list(d1)


# ----------------------------------------------------------------------------
# 算子基类：每个算子是一个“单元”，沿时间展开
#   forward(x1, x2, state, dt) -> (out[B], new_state)
# 非时序算子 state 为 None
# ----------------------------------------------------------------------------
class Operator(nn.Module):
    arity = 2          # 默认双入；单入算子把 x2 置 0
    temporal = False

    def __init__(self):
        super().__init__()

    def step(self, x1, x2, state, dt):
        raise NotImplementedError


class Add(Operator):
    arity = 2

    def step(self, x1, x2, state, dt):
        return x1 + x2, None


class Sub(Operator):
    arity = 2

    def step(self, x1, x2, state, dt):
        return x1 - x2, None


class Mul(Operator):
    arity = 2

    def step(self, x1, x2, state, dt):
        return x1 * x2, None


class Div(Operator):
    """Safe division：x1/(eps+|x2|)，避免数值爆炸（文档 §7.4）。"""
    arity = 2

    def step(self, x1, x2, state, dt):
        return safe_div(x1, x2), None


class Derivative(Operator):
    """带滚降的可学习微分器 G_D(s)=s/(tau_D s+1)（文档 6.2）。"""
    arity = 1
    temporal = True

    def __init__(self):
        super().__init__()
        self.ltau = nn.Parameter(torch.zeros(1))

    def step(self, x1, x2, state, dt):
        tau = torch.nn.functional.softplus(self.ltau) + EPS
        a = torch.exp(-dt / tau)                        # 低通系数（torch.exp 保持 autograd）
        prev_x, prev_y = (x1, torch.zeros_like(x1)) if state is None else state
        raw = (x1 - prev_x) / dt                        # 差分
        y = a * prev_y + (1.0 - a) * raw                # 滚降
        return y, (x1, y)


# FilteredDerivative 即 Derivative（带滚降可学习微分器），保留别名以契合文档命名
# 必须在 class Derivative 定义之后才能绑定
FilteredDerivative = Derivative


class Integrator(Operator):
    """泄漏积分（文档 6.2），避免偏置长期累积。"""
    arity = 1
    temporal = True

    def __init__(self):
        super().__init__()
        self.ltau = nn.Parameter(torch.zeros(1))

    def step(self, x1, x2, state, dt):
        tau = torch.nn.functional.softplus(self.ltau) + EPS
        a = torch.exp(-dt / tau.clamp(min=dt))
        prev = torch.zeros_like(x1) if state is None else state
        # 零阶保持离散：y_{t+1} = e^{-dt/τ} y_t + τ(1-e^{-dt/τ}) x_t
        # （原实现误用 dt 代替 τ(1-a)，非标准 leaky integrator）
        y = a * prev + (1.0 - a) * tau * x1
        return y, y


class Delay(Operator):
    """传输/测量/控制延迟（文档 6.2）。

    可微分数延迟：延迟量 d = softplus(ldelay) ∈ [0, max_delay] 为连续值，
    输出在离散延迟 lo=floor(d) 与 hi=lo+1 的缓冲值之间做线性插值，
    插值权重 w = d - lo 随 ldelay 连续可导 —— 因此 ∂L/∂ldelay 存在
    （原实现用 int(round(...)) 离散化，梯度链在延迟处断开）。
    """
    arity = 1
    temporal = True

    def __init__(self, max_delay=8):
        super().__init__()
        self.max_delay = max_delay
        self.ldelay = nn.Parameter(torch.tensor(2.0))

    def step(self, x1, x2, state, dt):
        d = torch.clamp(torch.nn.functional.softplus(self.ldelay), 0.0, float(self.max_delay))
        lo = torch.floor(d)
        hi = lo + 1.0
        w = d - lo                                  # 连续插值权重，随 d 可导
        buf = state if state is not None else [torch.zeros_like(x1)] * (self.max_delay + 1)
        buf = buf[1:] + [x1]
        n = len(buf)
        ilo = int(lo.item())
        ihi = int(hi.item())
        out_lo = buf[-(ilo + 1)] if ilo + 1 <= n else buf[0]
        out_hi = buf[-(ihi + 1)] if ihi + 1 <= n else buf[0]
        out = (1.0 - w) * out_lo + w * out_hi
        return out, buf


class Lag(Operator):
    """稳定一阶惯性 G(s)=K/(tau s+1)，tau=softplus（文档 6.3）。"""
    arity = 1
    temporal = True

    def __init__(self):
        super().__init__()
        self.ltau = nn.Parameter(torch.zeros(1))
        self.lK = nn.Parameter(torch.zeros(1))

    def step(self, x1, x2, state, dt):
        tau = torch.nn.functional.softplus(self.ltau) + EPS
        K = torch.nn.functional.softplus(self.lK) + EPS
        a = torch.exp(-dt / tau.clamp(min=dt))
        prev = torch.zeros_like(x1) if state is None else state
        y = a * prev + (1.0 - a) * K * x1
        return y, y


class Oscillator(Operator):
    """二阶振荡模态 K wn^2/(s^2+2 z wn s + wn^2)（文档 6.4）。
    参数可解释为固有频率 wn、阻尼 z、增益 K（故障表现为窗口级漂移）。"""
    arity = 1
    temporal = True

    def __init__(self):
        super().__init__()
        self.lwn = nn.Parameter(torch.zeros(1))
        self.lz = nn.Parameter(torch.zeros(1))
        self.lK = nn.Parameter(torch.zeros(1))

    def step(self, x1, x2, state, dt):
        wn = torch.nn.functional.softplus(self.lwn) + EPS
        z = torch.nn.functional.softplus(self.lz) + EPS
        K = torch.nn.functional.softplus(self.lK) + EPS
        y0, y1 = (torch.zeros_like(x1), torch.zeros_like(x1)) if state is None else state
        # RK2 (Heun's method) for the 2nd-order ODE:
        #   y0' = y1
        #   y1' = K*wn^2*x - 2*z*wn*y1 - wn^2*y0
        k1_0 = y1
        k1_1 = K * wn * wn * x1 - 2.0 * z * wn * y1 - wn * wn * y0
        k2_0 = y1 + dt * k1_1
        k2_1 = K * wn * wn * x1 - 2.0 * z * wn * (y1 + dt * k1_1) - wn * wn * (y0 + dt * k1_0)
        ny0 = y0 + dt * 0.5 * (k1_0 + k2_0)
        ny1 = y1 + dt * 0.5 * (k1_1 + k2_1)
        return ny0, (ny0, ny1)


class Saturation(Operator):
    """饱和非线性（文档 6.5）。"""
    arity = 1

    def __init__(self):
        super().__init__()
        self.lb = nn.Parameter(torch.tensor(0.5))

    def step(self, x1, x2, state, dt):
        b = torch.nn.functional.softplus(self.lb) + EPS
        return torch.clamp(x1, -b, b), None


class DeadZone(Operator):
    """死区非线性（文档 6.5）。"""
    arity = 1

    def __init__(self):
        super().__init__()
        self.lc = nn.Parameter(torch.tensor(0.2))

    def step(self, x1, x2, state, dt):
        c = torch.nn.functional.softplus(self.lc) + EPS
        return torch.where(torch.abs(x1) <= c, torch.zeros_like(x1), x1 - torch.sign(x1) * c), None


class Balance(Operator):
    """守恒/平衡残差结构模板（文档 6.6）。
    out = in1 - in2，物理含义：守恒关系的“不平衡量”。"""
    arity = 2

    def step(self, x1, x2, state, dt):
        return x1 - x2, None


# 注册表（顺序即 one-hot 索引）
OPERATOR_REGISTRY = [
    ("Add", Add), ("Sub", Sub), ("Mul", Div and Mul), ("Div", Div),
    ("Derivative", Derivative), ("Integrator", Integrator), ("Delay", Delay),
    ("Lag", Lag), ("Oscillator", Oscillator),
    ("Saturation", Saturation), ("DeadZone", DeadZone), ("Balance", Balance),
]
# 修正：上面 Mul 写法易混，显式列出
OPERATOR_REGISTRY = [
    ("Add", Add), ("Sub", Sub), ("Mul", Mul), ("Div", Div),
    ("Derivative", Derivative), ("Integrator", Integrator), ("Delay", Delay),
    ("Lag", Lag), ("Oscillator", Oscillator),
    ("Saturation", Saturation), ("DeadZone", DeadZone), ("Balance", Balance),
]
OPERATOR_NAMES = [n for n, _ in OPERATOR_REGISTRY]
OPERATOR_CLASSES = [c for _, c in OPERATOR_REGISTRY]
K_OPS = len(OPERATOR_NAMES)
