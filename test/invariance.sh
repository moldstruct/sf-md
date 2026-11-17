#!/bin/bash
#
# Bitwise invariance test for gromacs-sf: does a change leave the existing
# model exactly as it was?
#
# Runs short cases with the ADK seed pinned and compares every output file
# bit for bit: the final .gro, the .trr and .edr, and everything the ionize
# module writes to simulation_output/.  Meant for changes that must not alter
# behaviour, e.g. adding the trapped-electron cloud with sfmd-electrons = none.
#
# Without the cloud (sfmd-electrons = none, unchanged since sf-md 818af10 apart
# from the bondfree energy fix and the transition-log barrier):
#   water_nt1      360-atom TIP3P cluster, -nt 1 -ionize
#   water_pd4      same, -nt 4 -pd (per-rank RNG streams, charge re-sync)
#   water_plain    same tpr without -ionize: stock GROMACS path, no output dir
#   water_charges  fixed charges from water_charges.txt, pulse energy 0,
#                  -nt 4 -pd: the altered force field without ionization, and
#                  the sfmd-initial-charges reader
#   lyso_nt1       lysozyme (1960 atoms, CHARMM: Urey-Bradley, dihedrals,
#                  CMAP, 1-4), ~5500 events in 3 fs, -nt 1 -ionize
#   lyso_pd4       same at -nt 4 -pd
#   water_pd4_stop water, 12000 steps at -nt 4 -pd with autostop at
#                  E_kin/E_tot = 0.87, which fires at ~step 10500 (autostop
#                  never fires before step 10000): the master-decided,
#                  broadcast stop and the final write after it
#
# With the cloud (sfmd-electrons = pbgrid, the field-free port of gromacs-mc):
#   lyso_pb_seed   fixed charges (invariance/lyso_charges.txt, sum 5546) and a
#                  seeded cloud of 4000 electrons at 120 keV
#                  (GMX_SFMD_PB_SEED), pulse off, 400 steps at -nt 4 -pd: the
#                  grid, solve, forces, ledger and evaporation without ADK.
#                  This is the case that was checked bit for bit against
#                  gromacs-mc (test/pb_vs_mc.sh); keep it that way
#   lyso_pb_adk    the lysozyme pulse with the cloud, 600 steps at -nt 4 -pd:
#                  ADK on the master, escape or trapping at birth, the
#                  sequential ledger, the solves during the onset burst
#   lyso_pb_laser  the same with sfmd-laser-barrier = 1: escape at birth
#                  against the lowered level, the spill after the solves,
#                  evaporation above the lowered level
#   lyso_pb_laser2 lyso_pb_laser with sfmd-laser-barrier = 2: spill and
#                  evaporation against the instantaneous field |E(t)|
#   lyso_pb_heat   the same with sfmd-laser-heating = 1 as well: the
#                  inverse-bremsstrahlung heating of the cloud
#   lyso_pb_shape  lyso_pb_heat with sfmd-heating-shape = ellipsoid: the
#                  depolarization factors of the sample's ellipsoid
#   lyso_pb_surf   lyso_pb_shape with sfmd-heating-surface = 1: surface
#                  damping of the cloud's plasmon in the heating
#   lyso_pb_coll   the same with sfmd-collisions = 1 as well: electron-impact
#                  ionization by the cloud (collisional_log.txt)
#   lyso_pb_rec    the same with sfmd-coll-recombination = 1 as well: three-body
#                  recombination (q_after < q_before in collisional_log.txt)
#   lyso_pb_temin  lyso_pb_rec with sfmd-te-min = 50 eV, above the cloud's
#                  T_e until the heating takes over (~0.2 fs): solve,
#                  evaporation and recombination at the floor, no E_kin clamp
#   water_pd4_dirstop  water_pd4 with sfmd-autostop = 2 (ion directions
#                  settled, 1 fs checks, 10 fs after the peak at the earliest):
#                  the stop decided on the master and broadcast
#   lyso_pb_default  sfmd-electrons = pbgrid alone: the defaults are the
#                  recommended model (barrier 1, heating with ellipsoid and
#                  surface damping, collisions, te-min 1; no recombination).
#                  The other lyso_pb_* cases set the pre-2026-11-11 defaults
#                  (all off) first, so they test what they did before.
#   lyso_pb_lr     lyso_pb_adk with a 1 nm cutoff and sfmd-pb-long-range = 1:
#                  the ion field beyond the cutoff from the grid (pbg_lr.txt)
#   lyso_pb_pol    lyso_pb_default with sfmd-laser-polarization = z: the
#                  field in the solve, the saddle level, q E on the ions
#                  (laser_log.txt has the field and the dipole)
# A reference build without the cloud cannot run these two: with
# --ref-bindir pointing at one, pass --cases to leave them out.
#
# pb_log.txt is compared without its wall-time column.
#
# transition_log.txt is compared after sorting: under -pd the ranks append to
# it in whatever order they get there, so only its content is reproducible.
# .log and .cpt hold timestamps and host names and are not compared.
#
# Besides the bitwise comparison, each case checks what holds on any build:
# ionizing cases ionize at all and sum(charges) = number of events; the
# fixed-charge case has no events and keeps its input charge; the plain case
# writes no simulation_output/; the autostop case stops before nsteps and still
# writes its final .gro; the cloud cases never clamp the cloud's kinetic energy
# (E_clamped = 0), and in the ADK case every event either escaped or was
# trapped (or evaporated afterwards).
#
# Two modes:
#   default        compare with the stored checksums in test/invariance/.
#                  Those are specific to the build and machine that made
#                  them (see the header of reference.md5).
#   --ref-bindir   run a reference build too and compare the two directly;
#                  independent of the machine.  Each build uses its own
#                  grompp, so a tpr version change is fine.
#
# Usage:
#   test/invariance.sh [--bindir DIR] [--ref-bindir DIR] [--update]
#                      [--cases "water_nt1 lyso_nt1"] [--keep]
#
# --bindir defaults to the in-tree build (build/src/kernel).  --update writes
# the stored checksums from --bindir; only do that from a frozen reference
# build, never to make a failing change pass.
#
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TEST="$REPO/test"
BINDIR="$REPO/gromacs-4.5.4-SFMD/build/src/kernel"
REFBIN=""
UPDATE=0
KEEP=0
SEED=12345
ALL_CASES="water_nt1 water_pd4 water_plain water_charges water_pd4_stop water_pd4_dirstop water_pd4_sched lyso_nt1 lyso_pd4 lyso_pb_seed lyso_pb_adk lyso_pb_laser lyso_pb_laser2 lyso_pb_heat lyso_pb_shape lyso_pb_surf lyso_pb_coll lyso_pb_rec lyso_pb_temin lyso_pb_lr lyso_pb_sched lyso_pb_default lyso_pb_pol"
CASES="$ALL_CASES"
MANIFEST="$TEST/invariance/reference.md5"

while [ $# -gt 0 ]; do
    case "$1" in
        --bindir)     BINDIR="$2"; shift 2 ;;
        --ref-bindir) REFBIN="$2"; shift 2 ;;
        --update)     UPDATE=1;    shift ;;
        --cases)      CASES="$2";  shift 2 ;;
        --keep)       KEEP=1;      shift ;;
        -h|--help)    sed -n '2,69p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ "$UPDATE" = 1 ] && [ -n "$REFBIN" ]; then
    echo "--update and --ref-bindir don't go together" >&2; exit 2
fi

for d in "$BINDIR" $REFBIN; do
    for exe in grompp mdrun; do
        if [ ! -x "$d/$exe" ]; then
            echo "FAIL: $exe not found in $d" >&2; exit 2
        fi
    done
done

export GMXLIB="${GMXLIB:-$REPO/gromacs-4.5.4-SFMD/share/top}"
export GMX_SFMD_SEED="$SEED"

WORK="$(mktemp -d)"
trap '[ "$KEEP" = 1 ] && echo "kept $WORK" || rm -rf "$WORK"' EXIT

echo "workdir  $WORK"
echo "binaries $BINDIR"
[ -n "$REFBIN" ] && echo "against  $REFBIN"
echo "seed     $SEED"
echo

FAILED=0
fail() { echo "    FAIL  $*"; FAILED=1; KEEP=1; }

# set_key FILE KEY VALUE: replace the line for KEY, or append it.
set_key() {
    if grep -qE "^[[:space:]]*$2[[:space:]]*=" "$1"; then
        sed -i -E "s|^[[:space:]]*$2[[:space:]]*=.*|$2 = $3|" "$1"
    else
        echo "$2 = $3" >> "$1"
    fi
}

case_setup() {   # sets SYS, MDRUN_ARGS, KEYS, PBSEED for case $1
    KEYS=""
    PBSEED=""
    case "$1" in
        water_nt1)     SYS=water; MDRUN_ARGS="-nt 1 -ionize" ;;
        water_pd4)     SYS=water; MDRUN_ARGS="-nt 4 -pd -ionize" ;;
        water_plain)   SYS=water; MDRUN_ARGS="-nt 1" ;;
        water_charges) SYS=water; MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="sfmd-pulse-energy=0 sfmd-initial-charges=1" ;;
        lyso_nt1)      SYS=lyso;  MDRUN_ARGS="-nt 1 -ionize" ;;
        water_pd4_stop) SYS=water; MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=12000 sfmd-autostop=1 sfmd-autostop-energy-ratio=0.87" ;;
        water_pd4_sched) SYS=water; MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=4000 sfmd-schedule=6:0.015:0.2,9:0.03:0.2" ;;
        water_pd4_dirstop) SYS=water; MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=12000 sfmd-autostop=2 sfmd-autostop-interval=1 sfmd-autostop-min-time=10 sfmd-autostop-angle=2" ;;
        lyso_pd4)      SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize" ;;
        lyso_pb_seed)  SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=400 sfmd-pulse-energy=0 sfmd-electrons=pbgrid"
                       PBSEED="charges.txt 4000 120000" ;;
        lyso_pb_adk)   SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid" ;;
        lyso_pb_laser) SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-barrier=1" ;;
        lyso_pb_laser2) SYS=lyso; MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-barrier=2" ;;
        lyso_pb_heat)  SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-barrier=1 sfmd-laser-heating=1" ;;
        lyso_pb_shape) SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-barrier=1 sfmd-laser-heating=1 sfmd-heating-shape=ellipsoid" ;;
        lyso_pb_surf)  SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-barrier=1 sfmd-laser-heating=1 sfmd-heating-shape=ellipsoid sfmd-heating-surface=1" ;;
        lyso_pb_coll)  SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-barrier=1 sfmd-laser-heating=1 sfmd-collisions=1" ;;
        lyso_pb_rec)   SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-barrier=1 sfmd-laser-heating=1 sfmd-collisions=1 sfmd-coll-recombination=1" ;;
        lyso_pb_temin) SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-barrier=1 sfmd-laser-heating=1 sfmd-collisions=1 sfmd-coll-recombination=1 sfmd-te-min=50" ;;
        lyso_pb_lr)    SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-pb-long-range=1 rcoulomb=1.0 rvdw=1.0 rlist=1.0" ;;
        lyso_pb_sched) SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=4000 sfmd-electrons=pbgrid sfmd-laser-barrier=1 sfmd-laser-heating=1 sfmd-heating-shape=ellipsoid sfmd-heating-surface=1 sfmd-collisions=1 sfmd-coll-recombination=2 sfmd-te-min=1 sfmd-pb-long-range=1 rcoulomb=1.0 rvdw=1.0 rlist=1.0 sfmd-schedule=5:0.01:0.5" ;;
        lyso_pb_default) SYS=lyso; MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid" ;;
        lyso_pb_pol)   SYS=lyso;  MDRUN_ARGS="-nt 4 -pd -ionize"
                       KEYS="nsteps=600 sfmd-electrons=pbgrid sfmd-laser-polarization=z" ;;
        *) echo "unknown case $1" >&2; exit 2 ;;
    esac
    # The cloud's physics defaults became the recommended model (2026-11-11).
    # The cases written before that keep testing what they did: the old
    # defaults first, then the case's own keys (which win).
    case "$1" in
        lyso_pb_default|lyso_pb_pol) ;;
        lyso_pb_*) KEYS="sfmd-laser-barrier=0 sfmd-laser-heating=0 sfmd-heating-shape=sphere sfmd-heating-surface=0 sfmd-collisions=0 sfmd-te-min=0 $KEYS" ;;
    esac
}

# run_case CASE BINDIR DIR: run one case into DIR; returns nonzero on failure.
run_case() {
    local c="$1" bin="$2" dir="$3" src kv
    case_setup "$c"
    mkdir -p "$dir"
    if [ "$SYS" = water ]; then src="$TEST"; else src="$TEST/lysozyme"; fi
    cp "$src/conf.gro" "$src/topol.top" "$src/exp.mdp" "$dir/"
    [ -f "$src/posre.itp" ] && cp "$src/posre.itp" "$dir/"
    [ "$c" = water_charges ] && cp "$TEST/invariance/water_charges.txt" "$dir/charges.txt"
    [ "$c" = lyso_pb_seed ] && cp "$TEST/invariance/lyso_charges.txt" "$dir/charges.txt"
    for kv in $KEYS; do set_key "$dir/exp.mdp" "${kv%%=*}" "${kv#*=}"; done
    (
        cd "$dir" || exit 1
        "$bin/grompp" -f exp.mdp -c conf.gro -p topol.top -o run.tpr \
            -po out.mdp -maxwarn 30 > grompp.log 2>&1 || { echo "grompp failed"; exit 1; }
        case "$MDRUN_ARGS" in *-ionize*) mkdir -p simulation_output ;; esac
        # set only for the seeded case: an empty value is an error in mdrun
        if [ -n "$PBSEED" ]; then export GMX_SFMD_PB_SEED="$PBSEED"; else unset GMX_SFMD_PB_SEED; fi
        "$bin/mdrun" -deffnm run -s run.tpr $MDRUN_ARGS > mdrun.log 2>&1 \
            || { echo "mdrun failed"; exit 1; }
    )
}

# checksums CASE DIR: print "<case> <file> <md5>" for every compared file.
checksums() {
    local c="$1" dir="$2" f n
    for f in run.gro run.trr run.edr; do
        echo "$c $f $(md5sum < "$dir/$f" | cut -d' ' -f1)"
    done
    if [ -d "$dir/simulation_output" ]; then
        for f in "$dir"/simulation_output/*; do
            n="${f##*/}"
            if [ "$n" = transition_log.txt ]; then
                echo "$c simulation_output/$n(sorted) $(LC_ALL=C sort "$f" | md5sum | cut -d' ' -f1)"
            elif [ "$n" = pb_log.txt ]; then
                echo "$c simulation_output/$n(no-wall) $(cut -d' ' -f1-9,11- "$f" | md5sum | cut -d' ' -f1)"
            else
                echo "$c simulation_output/$n $(md5sum < "$f" | cut -d' ' -f1)"
            fi
        done
    fi
}

# sanity CASE DIR: build-independent checks.
sanity() {
    local c="$1" dir="$2" so="$2/simulation_output" ev q q0 last
    case "$c" in
        water_plain)
            [ -d "$so" ] && fail "$c: simulation_output/ written without -ionize"
            return ;;
    esac
    if [ ! -f "$so/charges.txt" ]; then fail "$c: no charges.txt"; return; fi
    ev=$(wc -l < "$so/transition_log.txt")
    # impact ionizations are events too (sfmd-collisions)
    # (recombinations, q_after < q_before, count -1)
    [ -f "$so/collisional_log.txt" ] && ev=$((ev + $(awk '{n += ($5 > $4) ? 1 : -1} END{print n+0}' "$so/collisional_log.txt")))
    q=$(awk '{s+=$2} END{print s+0}' "$so/charges.txt")
    if [ "$c" = water_charges ] || [ "$c" = lyso_pb_seed ]; then
        if [ "$c" = water_charges ]; then
            q0=$(awk '{s+=$2} END{print s+0}' "$TEST/invariance/water_charges.txt")
        else
            q0=$(awk '{s+=$2} END{print s+0}' "$TEST/invariance/lyso_charges.txt")
        fi
        [ "$ev" -eq 0 ] || fail "$c: $ev ionization events with the pulse off"
        [ "$q" = "$q0" ] || fail "$c: charge $q, input had $q0"
    else
        [ "$ev" -gt 0 ] || fail "$c: no ionization - the test would prove nothing"
        [ "$q" = "$ev" ] || fail "$c: sum(charges) = $q but $ev events"
    fi
    if [ "$c" = water_pd4_stop ]; then
        # last step printed in the energy blocks of run.log
        last=$(awk 'f{print $1; f=0} /^ *Step *Time/{f=1}' "$dir/run.log" | tail -1)
        grep -q "Simulation terminated" "$dir/mdrun.log" \
            && [ "$last" -gt 10000 ] && [ "$last" -lt 12000 ] \
            || fail "$c: autostop did not fire (last step $last)"
    fi
    if [ "$c" = water_pd4_dirstop ]; then
        # stops at the first check 10 fs after the peak (3 fs): 13.99 fs (step 4663, dt 3 as)
        last=$(awk 'f{print $1; f=0} /^ *Step *Time/{f=1}' "$dir/run.log" | tail -1)
        grep -q "Simulation terminated: ion directions settled" "$dir/mdrun.log" \
            && [ "$last" -lt 12000 ] && [ -s "$dir/simulation_output/autostop_dirs.txt" ] \
            || fail "$c: direction autostop did not fire (last step $last)"
    fi
    case "$c" in
        lyso_pb_*)
            local clamp esc trap evap
            [ -f "$so/pb_log.txt" ] || { fail "$c: no pb_log.txt"; return; }
            clamp=$(tail -1 "$so/pb_log.txt" | awk '{print $16}')
            awk -v e="$clamp" 'BEGIN{exit !(e == 0)}' || fail "$c: E_clamped = $clamp eV"
            if [ "$c" = lyso_pb_default ]; then
                # sfmd-electrons = pbgrid alone: the recommended model is on
                grep -q "laser barrier on" "$dir/run.log" && grep -q "laser heating on" "$dir/run.log" \
                    && grep -q "temperature floor 1 eV" "$dir/run.log" && [ -s "$so/collisional_log.txt" ] \
                    && ! grep -q "three-body recombination on" "$dir/run.log" \
                    || fail "$c: the default cloud is not barrier + heating + collisions + te-min 1, no recombination"
            fi
            if [ "$c" = lyso_pb_pol ]; then
                # the field acted: the polarization header, the dipole column nonzero
                grep -q "laser polarization z: polarization work" "$dir/run.log" \
                    && grep -q "^# sfmd-laser-polarization z" "$so/laser_log.txt" \
                    && awk '!/^#/ && NF == 22 && $20 != 0 {n++} END{exit !(n > 0)}' "$so/laser_log.txt" \
                    || fail "$c: the polarized field never acted on the cloud - not tested"
            fi
            if [ "$c" = lyso_pb_lr ]; then
                [ -s "$so/pbg_lr.txt" ] || fail "$c: no pbg_lr.txt - the long range never ran"
                grep -q "long-range ion field beyond rcoulomb = 1 nm from the grid: work -\?[1-9]" "$dir/run.log" \
                    || fail "$c: the long-range field did no work - not tested"
            fi
            if [ "$c" = lyso_pb_temin ]; then
                grep -q "below sfmd-te-min = 50" "$dir/run.log" || fail "$c: the floor never acted - not tested"
                awk '$5 < $4' "$so/collisional_log.txt" | grep -q . || fail "$c: no recombination - not tested"
            fi
            if [ "$c" = lyso_pb_rec ]; then
                awk '$5 < $4' "$so/collisional_log.txt" | grep -q . || fail "$c: no recombination - not tested"
            fi
            if [ "$c" = lyso_pb_coll ]; then
                [ -s "$so/collisional_log.txt" ] || fail "$c: no impact ionization - not tested"
            fi
            if [ "$c" = lyso_pb_heat ] || [ "$c" = lyso_pb_shape ] || [ "$c" = lyso_pb_surf ]; then
                grep -q "laser heating [1-9]" "$dir/run.log" || fail "$c: no laser heating - not tested"
            fi
            if [ "$c" = lyso_pb_laser ] || [ "$c" = lyso_pb_laser2 ]; then
                grep -q "spilled [1-9]" "$dir/run.log" || fail "$c: nothing spilled - the barrier was not tested"
            fi
            if [ "$c" = lyso_pb_adk ]; then
                read -r esc trap evap < <(sed -n 's/.*escaped at birth \([0-9]*\), trapped \([0-9]*\), evaporated \([0-9]*\).*/\1 \2 \3/p' "$dir/run.log")
                [ -n "$esc" ] && [ $((esc + trap)) -eq "$ev" ] \
                    || fail "$c: escaped $esc + trapped $trap != $ev events"
                [ "${trap:-0}" -gt 0 ] || fail "$c: nothing trapped - the cloud was not tested"
            fi
            ;;
    esac
    printf '    %-14s events %-6s charge %s\n' "$c" "$ev" "$q"
}

GOT="$WORK/got.md5"; : > "$GOT"
REFSUMS="$WORK/ref.md5"; : > "$REFSUMS"

echo "--- running"
for c in $CASES; do
    if ! out=$(run_case "$c" "$BINDIR" "$WORK/new/$c"); then
        fail "$c: $out (see $WORK/new/$c)"; continue
    fi
    sanity "$c" "$WORK/new/$c"
    checksums "$c" "$WORK/new/$c" >> "$GOT"
    if [ -n "$REFBIN" ]; then
        if ! out=$(run_case "$c" "$REFBIN" "$WORK/ref/$c"); then
            fail "$c (reference build): $out"; continue
        fi
        checksums "$c" "$WORK/ref/$c" >> "$REFSUMS"
    fi
done

if [ "$UPDATE" = 1 ]; then
    if [ "$FAILED" != 0 ]; then echo; echo "not updating: sanity checks failed"; exit 1; fi
    mkdir -p "$TEST/invariance"
    {
        echo "# test/invariance.sh reference checksums, seed $SEED"
        echo "# build: $BINDIR"
        echo "# host:  $(hostname), $(date -I)"
        [ -f "$BINDIR/../SOURCE.txt" ] && sed 's/^/# /' "$BINDIR/../SOURCE.txt"
        cat "$GOT"
    } > "$MANIFEST"
    echo; echo "wrote $MANIFEST ($(wc -l < "$GOT") files)"
    exit 0
fi

echo "--- bitwise comparison"
if [ -n "$REFBIN" ]; then
    EXPECT="$REFSUMS"
else
    if [ ! -f "$MANIFEST" ]; then echo "no $MANIFEST; make it with --update"; exit 2; fi
    EXPECT="$WORK/expect.md5"
    grep -v '^#' "$MANIFEST" | awk -v cs=" $CASES " 'index(cs, " "$1" ")' > "$EXPECT"
fi
for c in $CASES; do
    d=$(diff <(awk -v c="$c" '$1==c' "$EXPECT" | sort) <(awk -v c="$c" '$1==c' "$GOT" | sort))
    if [ -z "$d" ]; then
        echo "    PASS  $c ($(awk -v c="$c" '$1==c' "$GOT" | wc -l) files identical)"
    else
        fail "$c differs:"
        echo "$d" | sed 's/^/          /'
    fi
done

echo
if [ "$FAILED" = 0 ]; then echo "OK"; else echo "FAILURES"; fi
exit $FAILED
