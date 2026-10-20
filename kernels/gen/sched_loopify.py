#!/usr/bin/env python3
"""Grid-stride loop around a SHIPPED decode cubin, keeping its schedule.

The decode launch has grid.y = the moe_align table length, but a tier only
has its live pairs in it; the empty ones load their desc and EXIT (~0.5 us
w2 / ~0.8 us w4 per pair per launch on GB10). With the host compacting each
tier's live pairs to the front of its table (patch 14), a small fixed grid.y
walks y = ctaid.y, ctaid.y + y_stride, ... and stops at the first empty
entry or at n_pairs.

Putting the loop into the generator and letting cubit schedule the result
changed the scoreboard allocation of the whole K-march and cost ~20%. This
tool starts from the shipped cubin's disassembly instead
(`cubit disassemble X.cubin > X.dis`), freezes every control code of the
original instructions ([B..:R.:W.:Y:S..] = hand_sched) and adds only the
loop instructions, with explicit, conservative codes:

  prelude    S2UR UR8 = ctaid.y, LDCU UR9 = y_stride (SB5)
  pair_loop: <original body>; each `S2R Ry, SR_CTAID.Y` becomes
             `MOV Ry, UR8` (waits SB5, own stall >= 6, no scoreboard);
             before the prologue's first use of Ry: LDC R0 = n_pairs (SB5),
             Ry >= n_pairs -> EXIT (R0 is dead there: the next original
             instruction writes it; the predicate needs S14, as the original
             m_rows check has - S06 read a stale P4)
  back-edge  BAR.SYNC waiting on ALL scoreboards (drains loads/STG reads;
             orders the split-K smem reuse), UR8 += UR9, BRA pair_loop

ABI: two u32 params after n_kb128 - n_pairs (table length) and y_stride
(= the launch grid.y); n_pairs = y_stride = grid.y is the one-pair-per-CTA
behaviour, so uncompacted tables stay valid. The pair index lives in UR8:
no extra vector registers, same regcount.

usage: sched_loopify.py X.dis X.sass OUT.sass
  X.sass = the shipped SASS of the same kernel (for .entry/.reg/.param/.shared)
Assemble OUT.sass as described in kernels/cubins-sm120-loop/README.md.
"""
import re
import sys


def ctrl(sched):
    wait = (sched >> 11) & 0x3f
    rb = (sched >> 8) & 7
    wb = (sched >> 5) & 7
    y = (sched >> 4) & 1
    st = sched & 0xf
    b = "".join(str(i) if wait >> i & 1 else "-" for i in range(6))
    return (f"[B{b}:R{'-' if rb == 7 else rb}:W{'-' if wb == 7 else wb}:"
            f"{'Y' if y else '-'}:S{st:02d}]")


def cc(wait=(), st=6, wb=None):
    b = "".join(str(i) if i in wait else "-" for i in range(6))
    return f"[B{b}:R-:W{'-' if wb is None else wb}:-:S{st:02d}]"


def main(dis, header, out):
    hdr = []
    for s in open(header).read().splitlines():
        if re.match(r"^\.entry|^\s+\.(reg|param|shared)", s):
            hdr.append(s)
            if s.strip() == ".param u32 n_kb128":
                hdr += ["    .param u32 n_pairs", "    .param u32 y_stride"]
    assert sum(".param" in s for s in hdr) == 7, "header must end with the loop ABI"
    body = []
    for s in open(dis).read().splitlines():
        m = re.match(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*/\* @sched 0x([0-9a-f]+) \*/ ;$", s)
        if m:
            body.append((m.group(2), int(m.group(3), 16)))
    assert body, "no instructions parsed"
    # drop the trailing self-branch trap after the final EXIT
    while body[-1][0].startswith("BRA"):
        body.pop()
    assert body[-1][0] == "EXIT", body[-1]
    res = list(hdr)
    res += [f"    {cc(st=1, wb=5)} S2UR UR8, SR_CTAID.Y ;",
            f"    {cc(st=1, wb=5)} LDCU UR9, c[0x0][0x39c] ;",
            "pair_loop:"]
    pend = None          # register waiting for the bound check
    n_ctaid = 0
    for i, (txt, sch) in enumerate(body):
        if pend and re.search(rf"\b{pend}\b", txt):
            assert re.match(r"\w+(\.\w+)* R0,", txt), f"R0 not dead at: {txt}"
            res.append(f"    {cc(st=1, wb=5)} LDC R0, c[0x0][0x398] ;")
            # predicate -> @P EXIT needs the long ALU->predicate latency: the
            # original m_rows check uses S14 (0x007ee), S06 read a stale P4
            res.append(f"    {cc(wait=(5,), st=14)} ISETP.GE.U32.AND P4, PT, {pend}, R0, PT ;")
            res.append(f"    {cc(st=5)} @P4 EXIT ;")
            pend = None
        if i == len(body) - 1:      # final EXIT -> back-edge
            res += [f"    {cc(wait=range(6), st=6)} BAR.SYNC.DEFER_BLOCKING 0x0 ;",
                    f"    {cc(st=14)} UIADD3 UR8, UPT, UPT, UR8, UR9, URZ ;",
                    f"    {cc(st=5)} BRA pair_loop ;"]
            break
        m = re.match(r"S2R (R\d+), SR_CTAID\.Y$", txt)
        if m:
            c = ctrl(sch)
            wait = [i for i in range(6) if (sch >> 11) & 0x3f & (1 << i)] + [5]
            res.append(f"    {cc(wait=wait, st=max(6, sch & 0xf))} MOV {m.group(1)}, UR8 ;")
            if n_ctaid == 0:
                pend = m.group(1)
            n_ctaid += 1
            continue
        res.append(f"    {ctrl(sch)} {txt} ;")
    assert n_ctaid >= 1 and pend is None
    open(out, "w").write("\n".join(res) + "\n")
    print(f"wrote {out}: {len(body)} original + loop, ctaid.y reads {n_ctaid}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
