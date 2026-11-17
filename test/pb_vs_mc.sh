#!/bin/bash
#
# The trapped-electron cloud of gromacs-sf against gromacs-mc, bit for bit.
#
# sfelectrons.c is gromacs-mc's PB cloud (mc-md 1eb798f, mcionize.c) with
# renames.  With the same structure, the same fixed charges, the same seeded
# cloud, the pulse off and gromacs-mc's own processes off (charge transfer,
# collisions, field ionization), the two must agree in every output: .gro,
# .trr, .edr, energy_ledger.txt, screening_log.txt, pbg_box.txt (data lines)
# and pb_log.txt (without its wall-time column).
#
# gromacs-mc needs a way to start from a given cloud, which it does not have.
# test/mc_pb_seed.patch adds GMX_MCMD_PB_SEED="N E_kin" (18 lines, first
# step only).  Apply it to a COPY of mc-md and build that, e.g.
#   rsync -a --exclude build ~/software/mc-md/gromacs-4.5.4-MCMD /scratch/mcseed/
#   cd /scratch/mcseed && patch -p1 < test/mc_pb_seed.patch
#   mkdir gromacs-4.5.4-MCMD/build && cd gromacs-4.5.4-MCMD/build
#   cmake .. -DCMAKE_BUILD_TYPE=Release -DGMX_FFT_LIBRARY=fftw3 -DBUILD_SHARED_LIBS=OFF
#   make -j8 mdrun grompp
# and pass --mc-bindir .../build/src/kernel (with share/top next to it, or
# set MC_GMXLIB).
#
# Checked 2026-11-02: identical at -nt 1 (400 steps), -nt 4 -pd (2000 steps;
# and 20 fs at 10 as, 7 grid rebuilds, 6 box resizes, 479 ions outside the
# box), and on an 8970-atom protein at -nt 6 -pd.
#
# Usage:
#   test/pb_vs_mc.sh --mc-bindir DIR [--sf-bindir DIR] [--nt N] [--steps N]
#                    [--dt PS] [--keep]
#
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TEST="$REPO/test"
SFBIN="$REPO/gromacs-4.5.4-SFMD/build/src/kernel"
MCBIN=""
NT=4
STEPS=400
DT=0.000002
KEEP=0
N_SEED=4000
E_SEED=120000

while [ $# -gt 0 ]; do
    case "$1" in
        --mc-bindir) MCBIN="$2"; shift 2 ;;
        --sf-bindir) SFBIN="$2"; shift 2 ;;
        --nt)        NT="$2";    shift 2 ;;
        --steps)     STEPS="$2"; shift 2 ;;
        --dt)        DT="$2";    shift 2 ;;
        --keep)      KEEP=1;     shift ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$MCBIN" ] || { echo "--mc-bindir is required (a gromacs-mc build with test/mc_pb_seed.patch)" >&2; exit 2; }

SF_GMXLIB="${SF_GMXLIB:-$REPO/gromacs-4.5.4-SFMD/share/top}"
MC_GMXLIB="${MC_GMXLIB:-$(cd "$MCBIN/../../.." 2>/dev/null && pwd)/share/top}"
PD=""; [ "$NT" -gt 1 ] && PD="-pd"

WORK="$(mktemp -d)"
trap '[ "$KEEP" = 1 ] && echo "kept $WORK" || rm -rf "$WORK"' EXIT
echo "workdir $WORK; sf $SFBIN; mc $MCBIN; -nt $NT $PD, $STEPS steps at dt $DT ps"

base="$WORK/base.mdp"
grep -v "^sfmd-" "$TEST/lysozyme/exp.mdp" | sed -e "s/^nsteps .*/nsteps = $STEPS/" -e "s/^dt .*/dt = $DT/" > "$base"
for d in sf mc; do
    mkdir -p "$WORK/$d/simulation_output"
    cp "$TEST/lysozyme/conf.gro" "$TEST/lysozyme/topol.top" "$TEST/lysozyme/posre.itp" "$WORK/$d/"
    cp "$TEST/invariance/lyso_charges.txt" "$WORK/$d/charges.txt"
    cp "$base" "$WORK/$d/exp.mdp"
done
cat >> "$WORK/sf/exp.mdp" <<'EOF'
sfmd-pulse-peak-time = 2.0
sfmd-pulse-fwhm = 2.0
sfmd-pulse-energy = 0
sfmd-pulse-focal-diameter = 30000
sfmd-pulse-wavelength = 850
sfmd-electrons = pbgrid
; the field-free cloud: the physics defaults (recommended model since 2026-11-11) off
sfmd-laser-barrier = 0
sfmd-laser-heating = 0
sfmd-heating-shape = sphere
sfmd-heating-surface = 0
sfmd-collisions = 0
sfmd-te-min = 0
EOF
cat >> "$WORK/mc/exp.mdp" <<'EOF'
mcmd-charge-transfer = 0
mcmd-pulse-peak-time = 2.0
mcmd-pulse-fwhm = 2.0
mcmd-pulse-photons = 0
mcmd-pulse-focal-diameter = 100
mcmd-pulse-photon-energy = 1000
mcmd-initial-charges = 1
mcmd-electrons = pbgrid
mcmd-collisions = 0
mcmd-field-ionization = 0
mcmd-seed = 1
EOF

( cd "$WORK/sf" && GMXLIB="$SF_GMXLIB" "$SFBIN/grompp" -f exp.mdp -o run.tpr -po o.mdp -maxwarn 30 > grompp.log 2>&1 \
  && GMXLIB="$SF_GMXLIB" GMX_SFMD_SEED=1 GMX_SFMD_PB_SEED="charges.txt $N_SEED $E_SEED" \
     "$SFBIN/mdrun" -deffnm run -nt "$NT" $PD -ionize > mdrun.log 2>&1 ) || { echo "FAIL: sf run, see $WORK/sf"; KEEP=1; exit 1; } &
( cd "$WORK/mc" && GMXLIB="$MC_GMXLIB" "$MCBIN/grompp" -f exp.mdp -o run.tpr -po o.mdp -maxwarn 30 > grompp.log 2>&1 \
  && GMXLIB="$MC_GMXLIB" GMX_MCMD_PB_SEED="$N_SEED $E_SEED" \
     "$MCBIN/mdrun" -deffnm run -nt "$NT" $PD -ionize > mdrun.log 2>&1 ) || { echo "FAIL: mc run, see $WORK/mc"; KEEP=1; exit 1; } &
wait
grep -q "seeded N" "$WORK/mc/mdrun.log" || { echo "FAIL: the gromacs-mc build has no GMX_MCMD_PB_SEED (apply test/mc_pb_seed.patch)"; KEEP=1; exit 1; }

FAILED=0
same() { if cmp -s "$2" "$3"; then echo "    PASS  $1"; else echo "    FAIL  $1 differs"; FAILED=1; fi; }
S="$WORK/sf"; M="$WORK/mc"
for f in run.gro run.trr run.edr; do same "$f" "$S/$f" "$M/$f"; done
for f in energy_ledger.txt screening_log.txt; do same "$f" "$S/simulation_output/$f" "$M/simulation_output/$f"; done
same "pbg_box.txt (data)" <(tail -n +2 "$S/simulation_output/pbg_box.txt") <(tail -n +2 "$M/simulation_output/pbg_box.txt")
same "pb_log.txt (no wall time)" <(cut -d' ' -f1-9,11- "$S/simulation_output/pb_log.txt") <(cut -d' ' -f1-9,11- "$M/simulation_output/pb_log.txt")
echo "    $(tail -1 "$S/simulation_output/pb_log.txt" | awk '{printf "last solve: t %s ps, N %s, T_e %s eV, regrids %s", $1, $2, $4, $19}')"
echo
if [ "$FAILED" = 0 ]; then echo "OK"; else echo "FAILURES"; KEEP=1; fi
exit $FAILED
