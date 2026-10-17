#!/usr/bin/env python3
"""全文统一的方法配色与深度渐变(2026-10-11 组会口径)。

四类方法互不混用,各占一个色系:

    ECHO                   红   base #D64550
    EAGLE                  蓝   base #3478B8
    MTP                    灰   base #6B7280
    MTP warm-start EAGLE   品红 base #C34C98

同一方法内部按 MTP 深度排序:深度越小颜色越浅,越大越深。色阶取组会给定的四档锚点;
深度档位多于四档时在锚点之间线性插值,不会凭空多出"方法色"。

色号在**全局**配置表上解析,不按单张图里实际画了几条曲线重排。这样同一个配置
(比如 ECHO-1.5)在 Figure 1a、Figure 1b、Figure 3a、Figure 5 里永远是同一个颜色。

两处为可读性做的微调(组会允许"微调色值,不改色系"):
  * MTP 只有 MTP-1 / MTP-3 两档,取锚点 1 和 2(#9AA1AA / #6B7280)而不是 0 和 3。
    锚点 0(#C7CBD1)在白底上做线条太淡,锚点 3(#4B5563)又接近正文黑。
  * ECHO-2(7) 落在锚点 2 和 3 之间,插值得 #C03644。

用法:
    import method_colors as MC
    MC.color('ECHO-1.5')      -> '#E8757E'
    MC.BASE['EAGLE']          -> '#3478B8'
    python3 method_colors.py  # 打印全部配置的色号
"""

RAMPS = {
    'ECHO':                 ['#F2A1A7', '#E8757E', '#D64550', '#A92738'],
    'EAGLE':                ['#A9C7E8', '#6CA1D4', '#3478B8', '#1F4F7C'],
    'MTP':                  ['#C7CBD1', '#9AA1AA', '#6B7280', '#4B5563'],
    'MTP warm-start EAGLE': ['#E6A9CF', '#D97DB8', '#C34C98', '#8E2F6C'],
}
BASE = {m: r[2] for m, r in RAMPS.items()}

# 配置 -> (方法, 用于渐变的深度值)。深度值取各配置的 MTP 训练深度:ECHO-1.5 从 {1,2,3}
# 采样、ECHO-2(7) 从 {1..7} 采样,标签里的数字就是其平均深度,直接用它排序。
CONFIGS = {
    'ECHO-1':        ('ECHO', 1.00),
    'ECHO-1.5':      ('ECHO', 1.50),
    'ECHO-1.75(5)':  ('ECHO', 1.75),
    'ECHO-2(7)':     ('ECHO', 2.00),
    'ECHO-3':        ('ECHO', 3.00),
    'MTP-1':         ('MTP', 1.00),
    'MTP-3':         ('MTP', 3.00),
    'EAGLE3-cold':   ('EAGLE', 1.00),
    'EAGLE3-warm':   ('MTP warm-start EAGLE', 1.00),
}
# 同一配置的 sft 版与其 pretrain 版同色;serving 图里的 arm 名也映射到对应配置
ALIAS = {'ECHO-1 sft': 'ECHO-1', 'ECHO-1.5 sft': 'ECHO-1.5', 'ECHO-2(7) sft': 'ECHO-2(7)',
         'ECHO-MTP': 'ECHO-1.5', 'ECHO-MTP (K=7)': 'ECHO-2(7)',
         'EAGLE3 warm from DS-MTP (own target)': 'EAGLE3-warm'}

# 多档方法在 [0, 3] 的锚点坐标上取这些位置;单档方法直接用 BASE
SLOTS = {'ECHO': {1.00: 0.0, 1.50: 1.0, 1.75: 2.0, 2.00: 2.5, 3.00: 3.0},
         'MTP': {1.00: 1.0, 3.00: 2.0}}


def _hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb2hex(c):
    return '#%02X%02X%02X' % tuple(int(round(v)) for v in c)


def _sample(ramp, pos):
    """pos 是 [0, len(ramp)-1] 上的实数坐标,在相邻锚点之间线性插值。"""
    lo = int(pos)
    if lo >= len(ramp) - 1:
        return ramp[-1]
    f = pos - lo
    a, b = _hex2rgb(ramp[lo]), _hex2rgb(ramp[lo + 1])
    return _rgb2hex([a[i] + (b[i] - a[i]) * f for i in range(3)])


def method(label):
    """配置标签 -> 四类方法之一。未登记的标签返回 None。"""
    lab = ALIAS.get(label, label)
    return CONFIGS[lab][0] if lab in CONFIGS else None


def color(label):
    """配置标签 -> 十六进制色号。未登记的标签返回 None。"""
    lab = ALIAS.get(label, label)
    if lab not in CONFIGS:
        return None
    meth, depth = CONFIGS[lab]
    slots = SLOTS.get(meth)
    return BASE[meth] if slots is None else _sample(RAMPS[meth], slots[depth])


def table():
    """按方法分组列出全部配置的色号,供核对。"""
    out = {}
    for lab, (meth, depth) in sorted(CONFIGS.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        out.setdefault(meth, []).append((lab, depth, color(lab)))
    return out


if __name__ == '__main__':
    for meth, rows in table().items():
        print(f'{meth}  (base {BASE[meth]})')
        for lab, depth, col in rows:
            print(f'    {lab:<14} depth {depth:<5} {col}')
