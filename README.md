# Hybrid Strong-Field / Molecular Dynamics (SF/MD)

This is a strong-field ionization Molecular Dynamics (SF/MD) model that is part of the **MOLDSTRUCT** toolbox. It can be used to simulate laser-induced ionization in the optical regieme based on ADK theory and the resulting Coulomb explosion of molecular systems, from small clusters up to whole proteins.
The code is based on a modified version of GROMACS ([webpage](https://www.gromacs.org/)), with functionality close to normal GROMACS plus additional parameters (an `-ionize` module) that drive ionization from a laser pulse.
The model is developed by the Biophysics group at Uppsala University.

# Manual

This is a brief manual that will cover how to install, use the model and evaluate output.
Basic knowledge of GROMACS is assumed, check out [GROMACS webpage](https://www.gromacs.org/) for more information.

**The full reference manual is `manual/manual.pdf`** (sources in `manual/ch/`, build with
`tectonic manual.tex` or pdflatex twice): the physics of the pulse, ADK, the force field, the electron cloud, the
laser on the cloud and the collisions, every option and output file, the numerics, parallel runs and what the model
has been tested on.

## Contents

- Installation
- List of input parameters
- Trapped electrons (the electron cloud)
- Running a simulation
- Output
- Example
- Testing
- Limitations

## Installation

The installation process is the exact same as for a normal GROMACS installation.

1. Download the zip of the repository and extract it.
2. Place the `gromacs-4.5.4-SFMD` folder somewhere appropriate.
   This is only the installation files, you will choose later on where you want to install the software.
3. Go into `gromacs-4.5.4-SFMD` and create a new directory called `build` (or whatever you want).
4. Go into the newly created directory.
5. Run `cmake ../ -DCMAKE_INSTALL_PREFIX=/path/to/install/location`, where `/path/to/install/location` is replaced with where the software will be installed.
6. Run `make install`.
7. If everything worked out, the software should now be installed at your specified location.

If running into problems, try to search the issue as there are many GROMACS resources out there that can help with the install process.

## List of input parameters

The model's parameters are set in the `.mdp` file like any other MD
parameter. They are all named `sfmd-*` and take effect only when `mdrun` is
given `-ionize`; without that flag the run is unmodified GROMACS 4.5.4
whatever the `.mdp` says.

**Note the time unit.** `sfmd-pulse-peak-time` and `sfmd-pulse-fwhm` are in
**femtoseconds**, unlike `dt` and `tinit`, which stay in GROMACS'
picoseconds, and the width is the **FWHM** rather than the standard
deviation. Pulses here are tens of fs long, and the templates were already
converting FWHM to sigma by hand. Output files still use the GROMACS time in
ps.

```
sfmd-pulse-peak-time        (default = 0)    Center of the Gaussian pulse [fs]
sfmd-pulse-fwhm             (default = 0)    Pulse duration, full width at
                                             half maximum [fs]
sfmd-pulse-energy           (default = 0)    Pulse energy [J]
sfmd-pulse-focal-diameter   (default = 0)    Diameter of the focal spot [nm]
sfmd-pulse-wavelength       (default = 0)    Laser wavelength [nm]
sfmd-autostop               (default = 0)    1: stop once E_kin/E_tot exceeds
                                             sfmd-autostop-energy-ratio;
                                             2: once the ion directions have
                                             settled (below)
sfmd-autostop-energy-ratio  (default = 0.99) That threshold (autostop 1)
sfmd-autostop-angle         (default = 2)    autostop 2: stop when the median
                                             heavy-ion direction turn still
                                             to come is below this (degrees)
sfmd-autostop-interval      (default = 5)    autostop 2: fs between checks
sfmd-autostop-min-time      (default = 60)   autostop 2: no stop before this
                                             many fs after the pulse peak
sfmd-schedule               (default = "")   t:dt:pb ... (fs): from time t on,
                                             MD step dt and PB interval pb
                                             (gromacs-mc's mcmd-schedule); the
                                             first change after the pulse
                                             (peak + 3 sigma); nsteps x dt
                                             stays the end time
sfmd-initial-charges        (default = 0)    Read starting charges from
                                             "charges.txt" in the working
                                             directory: two columns, atom
                                             index and charge. Used for
                                             restart chaining.
sfmd-detailed-output        (default = 0)    Write the per-step analysis
                                             output (see below)
sfmd-charge-output-stride   (default = 50)   Steps between frames in
                                             charges_over_time.bin
```

The `.tpr` format version was raised from 73 to **75** so a stock GROMACS tool would read an SFMD `.tpr` and silently
misparse everything after `userint4`. 74 is deliberately skipped: the sibling
gromacs-mc build uses it for its own, different layout, and sharing the
number would leave the two MolDStruct codes misreading each other's files.
**Existing `.tpr` files must be regenerated with the new `grompp`**; `.tpr`
files from stock 4.5.4 still read.

The electron-cloud options below moved the version on to **175–185** (175
the cloud, 176 the laser barrier, 177 the heating, 178 impact ionization, 179
recombination, 180 the heating shape, 181 surface damping, 182 the temperature
floor, 183 the direction autostop, 184 the schedule, 185 the polarization). gromacs-mc uses 74
upwards, so this build refuses 74 and 76–174 instead of misreading them; 75 and
175–184 still read, with the newer options off.


### Reproducible runs

The ADK tunnelling draws are seeded from `/dev/random`, so every run is an
independent realisation - which is what you want for ensemble statistics, and
why two runs of the byte-identical system diverge almost immediately once
ionization starts. Setting the environment variable `GMX_SFMD_SEED` pins the
seed instead:

```
GMX_SFMD_SEED=12345 mdrun -deffnm explode -nt 4 -pd -ionize
```

Each rank still gets its own stream (the rank index is mixed into the seed),
so a pinned run is reproducible **for a given rank count** - the same seed at
`-nt 1` and `-nt 4` are different realisations, because the ranks own
different atoms. The seed actually used is printed to the `.log`.

This is what makes the ionization path testable: without it there is no way
to tell a real change in behaviour from the RNG landing differently.


## Trapped electrons (the electron cloud)

By default (`sfmd-electrons = none`) every electron ADK frees leaves the
sample at once, so the MD sees bare ion charges. In a large sample that is
wrong at moderate intensity: once the net charge is high, new electrons cannot
climb out of the ion potential and stay trapped, screening the ions.
`sfmd-electrons = pbgrid` adds that trapped population as a cloud: the
Poisson-Boltzmann model of gromacs-mc (`mcmd-electrons = pbgrid`), ported text
for text and checked bit for bit against it (`test/pb_vs_mc.sh`).

How it works, in short:

- **The cloud** is a self-consistent electron density at one temperature T_e
  on a grid around the sample, solved every `sfmd-pb-interval` fs (Newton, FFT
  convolution, adaptive grid and box). Its field acts on the ions (screening),
  its energy is kept in a ledger with the ions.
- **At birth** an ADK electron (born at rest at its atom) escapes if the depth
  of the potential at its atom is below its energy, and joins the cloud
  otherwise.
- **Evaporation**: the hot tail of the cloud above the escape level leaves
  (`sfmd-evaporation`).
- **The laser on the cloud** (all cycle-averaged, using the envelope field
  E0(t)):
  - `sfmd-laser-barrier`: the field lowers the escape level of the net charge Q
    by dW = 2 sqrt(k Q e E0). Newborn electrons above it escape, and cloud cells
    above it spill out. With `= 1` the envelope E0 is used at every solve,
    whatever the phase of the field. With `= 2` the spill and the evaporation
    level follow the instantaneous field 2 sqrt(k Q e |E(t)|), which breathes
    twice per optical cycle (2.8 fs at 850 nm). The solves, ~14 per cycle at
    0.2 fs, resolve that. The escape at birth keeps E0, since ADK electrons are
    born near the field crests.
    In the lysozyme scans mode 2 gave the same charges, ion energies (within
    2 %) and directions as mode 1, only with more spill and less evaporation,
    about 1 fs earlier. It also produced clamped cloud energy in 3 of 24 runs
    (up to 4.6 keV) and a larger seed spread, because the jumping level lets a
    cold cloud's tail evaporate in bulk between solves. Use 1; 2 is a check.
  - `sfmd-laser-heating`: inverse bremsstrahlung, the absorption of a small
    Drude sphere (Ditmire et al. 1996), with the electron-ion collision rate
    (NRL formula) capped at `sfmd-heating-nu-cap` x omega_p. It includes the
    Mie resonance as the cloud expands through 3 n_c.
    `sfmd-heating-surface    = A` adds surface damping, nu_s = A vbar/R (vbar the
    mean thermal speed), which keeps that resonance finite for a hot, thin
    cloud.
    `sfmd-heating-shape = ellipsoid` uses the depolarization factors L_i of
    the sample's ellipsoid instead, |E_in|^2 = (E0^2/3) sum_i |1 + L_i (eps - 1)|^-2.
    That is averaged over the orientation of the polarization, because the
    model has no field direction. The resonance then falls at n_c / L_i for
    each axis rather than at 3 n_c. Lysozyme starts at L = 0.21 / 0.37 / 0.41.
- **Collisions with the ions** (Lotz, as gromacs-mc's built-in channel):
  - `sfmd-collisions`: electron-impact ionization. The rate at atom k is
    n_e(k) <sigma v>, with n_e(k) the cloud density at the atom and
    <sigma v> = 6.7e-7 a q T^-3/2 E1(P/T)/(P/T) cm^3/s (a = 4.5, q the valence
    occupation, P the ionization energy of the state). The new electron stays
    in the cloud, which pays P.
  - `sfmd-coll-recombination`: three-body recombination, the inverse by detailed
    balance (Saha, statistical weights C(shell size, occupation)). The captured
    electron leaves the cloud, which gets P.

### Parameters

```
sfmd-electrons              (default = none) none | pbgrid
sfmd-pb-interval            (default = 0.2)  fs between solves
sfmd-pb-smoothing           (default = 0.25) nm, charge smoothing s_e
sfmd-pb-cells-per-smoothing (default = 2.5)  grid spacing = s_e / this
sfmd-pb-box-cut             (default = 0.02) electrons allowed outside the box
sfmd-pb-box-max             (default = 3)    largest box half-width, in R
sfmd-pb-grid-limit          (default = 384)  largest grid per dimension
sfmd-pb-newton-tol          (default = 0.01) Newton tolerance (relative)
sfmd-pb-long-range          (default = auto) 1: GROMACS does the ion pairs
                                             within rcoulomb, the grid the
                                             rest (gromacs-mc's long range);
                                             auto: all pairs up to 10000
                                             atoms, above that 1 with a 2 nm
                                             cutoff (grompp sets the cut-offs)
sfmd-evaporation            (default = 1)    hot electrons leave over the
                                             escape level
sfmd-laser-barrier          (default = 1)    the laser lowers the escape level:
                                             1 by the envelope E0(t);
                                             2 spill and evaporation follow
                                             the instantaneous |E(t)|
sfmd-laser-heating          (default = 1)    inverse-bremsstrahlung heating
sfmd-heating-nu-cap         (default = 1)    nu_ei <= this x omega_p; 0: no cap
sfmd-heating-surface        (default = 1)    surface damping nu_s = A vbar/R
                                             added to nu_ei (A; 0 = off)
sfmd-heating-shape          (default = ellipsoid) sphere | ellipsoid: the field
                                             inside the absorbing cloud for a
                                             sphere, or for the sample's
                                             ellipsoid (orientation-averaged)
sfmd-collisions             (default = 1)    electron-impact ionization
sfmd-coll-recombination     (default = 0)    0 off; 1 always; 2 only after the
                                             pulse (past the peak and below
                                             1 % of the peak intensity)
sfmd-te-min                 (default = 1)    eV, temperature floor of the PB
                                             solve, evaporation and
                                             recombination; impact ionization
                                             at the actual T_e; E_kin is not
                                             clamped. 0: the old model (solve
                                             floor 1 eV, E_kin clamped at 0)
sfmd-laser-polarization     (default = 0)    DEVELOPMENT. x, y or z: the
                                             instantaneous field along that
                                             axis in the PB solve (the cloud
                                             polarizes), escape over the
                                             downhill saddle of net charge,
                                             dipole and field, q E on the ions.
                                             Needs the laser barrier on. See
                                             the manual, "Linear polarization"
```

The cloud options need `sfmd-electrons = pbgrid`, and
`sfmd-coll-recombination` needs `sfmd-collisions = 1`. `grompp` refuses any other
combination, and also `sfmd-initial-charges` with the cloud, because the cloud
has no restart state. With every option at its default, a run is bit for bit
the same as without the cloud code (`test/invariance.sh`).

### Recommended settings

For lysozyme-sized samples at 10^15–10^16 W/cm^2 and optical pulses. Since
2026-11-11 the cloud's defaults are this model, so `sfmd-electrons = pbgrid`
alone gives it (barrier 1, heating with the ellipsoid and surface damping,
impact ionization, te-min 1). Recombination stays off by default; the runs
below used it after the pulse:

```
sfmd-electrons          = pbgrid
sfmd-coll-recombination = 2      ; optional: gives the charge back after the
                                 ; pulse (1.72 -> 1.35 at 10^15), explosion unchanged
```

Each part can be switched off on its own (e.g. `sfmd-collisions = 0`). Without
the cloud these defaults are off; setting barrier, heating or collisions
without it is refused. Before 2026-11-11 every one of them defaulted to 0, so
an mdp that relied on that now gets the full model: set them to 0 (and
`sfmd-heating-shape = sphere`) to reproduce it. A tpr keeps what it was made with.

(`sfmd-te-min = 1` changes nothing at 10^15–10^16 up to 200 fs (the floor acts
for ~0.02 fs when the cloud forms) but removes the E_kin clamp; it matters for
longer runs, where the cloud cools towards 1 eV.)

(`sfmd-heating-shape = ellipsoid` changes little: at 10^15 13 % more heating
and 6 % more ion energy, elsewhere ~1 %, directions within the noise. It is the
better physics for a non-spherical sample.)

Why these, from seed scans of lysozyme (1960 atoms, 850 nm, 15 fs FWHM, 30 µm
spot, 8 seeds; details in the sfmd-electrons project log):

- **The field-free cloud** (`pbgrid` alone) traps about 98 % of the electrons at
  every intensity, which is wrong at high intensity. Use the barrier.
- **Barrier without heating** cools the cloud to the 1 eV floor. At 10^15 the
  result is then bistable from seed to seed (1196 ± 436 trapped). With heating
  it is reproducible (1139 ± 9). Cap 1 and no cap agree; a lower cap heats
  less at 10^15 (see Limitations).
- **Impact ionization** raises the ion charge at 10^15 by a third (1.29 → 1.72
  per atom), but the new electrons stay trapped. Net charge, ion energy (+6 %)
  and directions hardly change.
- **Recombination after the pulse** gives most of that charge back
  (1.72 → 1.35) and leaves the explosion as it was. That is the sensible choice
  when charge states matter. Recombination during the pulse (`= 1`) captures
  into the ground state inside a strong field, and the laser then re-ionizes
  the atoms (a capture-ionization pump: about 9x more ADK events, +22 % ion
  energy). Use it only as an upper bound.

What the cloud does, lysozyme. 10^15 with all the recommended options at
200 fs; 10^16 and 10^17 with barrier, heating and impact ionization at 75 fs
(recombination after the pulse was only run at 10^15, where it left the
explosion unchanged):

| peak [W/cm^2] | trapped at the end | ion energy vs cloud off | direction shift vs cloud off (fast 25 % / all heavy atoms) |
|---|---|---|---|
| 10^15 | ~47 % of the electrons | 0.31 | 6.5° / 8.7° |
| 10^16 | ~1 % | 1.00 | 3.7° / 5.0° |
| 10^17 | 0 (gone by ~28 fs) | 1.05 | 3.1° / 4.4° |

The shot-to-shot spread is 1–3° for the fast ions in every case, so these
shifts are systematic. At 10^17 the cloud leaves before the peak, but it holds
the sample compact while ADK charges it, which gives the ions about 5 % more
energy than without the cloud.

### Cost and parallel runs

The solve runs on the master rank, and the other ranks wait for it. Lysozyme on
3 cores (`-nt 3 -pd`), 200 fs at dt = 2 as:

- cloud off: 10 min;
- with the recommended options: 20–26 min.

Any `-nt` with `-pd` works. Particle decomposition is required anyway (see
Running a simulation).

### Environment variables (diagnostics)

```
GMX_SFMD_PB_SEED="charges.txt N E"  start from the charges in the file and a
                                    seeded cloud of N electrons with E eV
                                    kinetic energy in total (tests; see
                                    test/pb_vs_mc.sh)
GMX_SFMD_PB_TMIN=T                  temperature floor of the solve, default 1 eV
GMX_SFMD_PBG_FFT_THREADS=n          FFT threads of the solve
GMX_SFMD_TIMING_DETAIL=1            section timers in the log
```


## Running a simulation

To run a simulation we follow the exact same steps as for a normal GROMACS simulation up to calling `mdrun`. We need to run it with the `-ionize` flag.

```
path/to/gromacs/bin/pdb2gmx -f structure.pdb -ff "charmm27" -ignh
path/to/gromacs/bin/grompp -f exp.mdp -c conf.gro -p topol.top -o explode.tpr -maxwarn 5
path/to/gromacs/bin/mdrun -deffnm explode -v -nt 1 -ionize
```

## Output

### MD output

All standard GROMACS output like the `.trr`, `.edr`, `.gro`, `.log` and `.cpt` files are still given as output.

### Additional output

When `sfmd-detailed-output` is set to 1, additional output is written to `simulation_output/` (created automatically in the directory `mdrun` is called from):

- **`pulse_profile.txt`** — intensity of the laser pulse at each timestep. Column 1: time [ps]. Column 2: laser intensity.
- **`mean_charge_vs_time.txt`** — mean charge of the system over time, written every step. Column 1: time [ps]. Column 2: mean charge [e].
- **`charges_over_time.bin`** — per-atom charge, logged in binary (float32,
  one frame of all atoms every `sfmd-charge-output-stride` steps, default 50).
  Frame *k* is at tinit + *k* x stride x dt with the tpr's dt and stride; with
  `sfmd-schedule` the stride is rescaled per stage to keep that spacing, so
  the frame times are not step x dt (`tools/sfmd_output.py` handles it).
- **`masses.bin`** — per-atom mass, written once.
- **`transition_log.txt`** — log of ionization/electronic transition events and the time [ps] they occur.
- **`charges.txt`** — final per-atom charge dump at the end of the run (also the file read back in via `sfmd-initial-charges` for restarting/chaining a simulation).

If you do not need this data, it is recommended to leave
`sfmd-detailed-output` off, as it cuts into performance.

### Run summaries and the summary tool

`mdrun -ionize` prints a **run summary** at the first step (to stderr and the
`.log`), as gromacs-mc does: the system (atoms per element), the pulse with its
peak intensity and field, the run (steps, end time, schedule, seed, ranks),
every process with its settings (ADK; the cloud and each of its parts;
ion-ion forces), the output and autostop, and **warnings for settings outside
what was tested** (cloud without the barrier or heating, no surface damping,
recombination 1, te-min 0, an intensity outside 10^15–10^17 W/cm^2, harmonic
bonds, dt above 2 as, a cutoff without long range, ...). At the end it prints
the final charge per element and the events (ADK, electron impact,
recombination), followed by the cloud's own lines.

`tools/sfmd_output.py` (Python 3 with numpy, matplotlib for the plots; no
GROMACS needed) reads a run's `simulation_output/`:

```
tools/sfmd_output.py summary   run/            # run, final charges, events per element and per 5 fs, cloud, autostop
tools/sfmd_output.py charges   run/ --every 10 # mean charge per element over time
tools/sfmd_output.py transitions run/ --type adk,impact,recomb --atom 12 --tmin 20 --tmax 40
tools/sfmd_output.py plot      run/ [--gif [--no-h]] [--extract]   # run/overview.png, explosion.gif, extracted/
```

Every file goes on a time axis in fs from what the run wrote, never steps x
dt (wrong under `sfmd-schedule`): `screening_log.txt`, which counts steps, is
timed by interpolation in `energy_ledger.txt`, and the charge frames by the
tpr's dt and stride from the `.log`.

### Electron-cloud output

With `sfmd-electrons = pbgrid` these are written to `simulation_output/`
whatever `sfmd-detailed-output` says (except `collisional_log.txt`). Each has a header line naming its
columns.

- **`pb_log.txt`**: one line per solve. N, Q, T_e, grid spacing, ions outside
  the box, Newton/CG iterations, residual, wall time, mean/max depth, work and
  energy changes, the cloud's kinetic energy, `E_clamped` (energy the ledger
  had to clamp: should stay at or near 0), the core charge, and the
  convolutions, regrids and T iterations.
- **`screening_log.txt`**: per step, escaped and trapped counts, the trapped
  fraction, mean energy, R, Debye length, evaporated count, T_e and pressure
  work.
- **`energy_ledger.txt`**: the cumulative energy ledger (ion-ion and
  ion-electron jumps of charge changes, field work).
- **`pbg_box.txt`**: the adaptive box and grid at each rebuild.
- **`laser_log.txt`** (with `sfmd-laser-barrier` or `sfmd-laser-heating`):
  E0, Q, N, the lowering dW, the saddle radius, R, spilled, escaped at birth
  and evaporated totals, T_e, the heating power and total, nu/omega,
  |E_in|/E0 and Z_eff.
- **`collisional_log.txt`** (with `sfmd-collisions`, and like
  `transition_log.txt` only with `sfmd-detailed-output = 1`): one line per impact
  ionization or recombination. t, atom, element, charge before and after
  (after < before is a recombination), T_e, n_e at the atom [cm^-3], rate,
  probability and the random number.

The end of `mdrun`'s output (and the `.log`) has a summary line: escaped at
birth, trapped, evaporated, spilled, laser heating, the final N, Q and T_e,
solves, and the ledger with `E_clamped`. With collisions on there are further
lines with the counts of impact ionizations and recombinations, and the energy
they moved.

`transition_log.txt` keeps listing only the ADK events.
`mean_charge_vs_time.txt` and `charges.txt` hold the ion charges; the net
charge is the ion charge minus the cloud's N.

## Example

`example/` contains a complete worked run: hen egg-white lysozyme (PDB 1AKI,
1960 atoms) in vacuum, exploded by a 5 mJ, 5 fs pulse.

```bash
cd example
# edit run_example.sh to point at your gromacs bin directory, then
bash run_example.sh
```

which is just:

```bash
pdb2gmx -f 1aki.pdb -ff "charmm27" -water none
grompp  -f exp.mdp -c conf.gro -p topol.top -o explode.tpr
mdrun   -deffnm explode -v -nt 1 -ionize
```

Add `-pd` and raise `-nt` to use more cores
(`-nt 8 -pd`) - `-pd` is required whenever `-nt` > 1, because domain
decomposition cannot follow the explosion.

20000 steps at dt = 1 as covers 20 fs, with the pulse peaking at 10 fs. On a
completed run every atom ends up at least singly charged:

```
min 1   mean 2.83   max 6   neutral atoms: 0/1960
  1+ : 959     every hydrogen, which has only one electron to lose
  4+ : 613
  5+ : 193
  6+ : 195
```

Nothing is left at 2+ or 3+: at this intensity the heavy atoms pass through
the lower charge states almost immediately. The exact numbers vary run to run,
because the ADK draws are unseeded by default - set `GMX_SFMD_SEED` to pin
them. The pulse energy is not delicately tuned: full ionization sets in around
1e-4 J for this system, so the 5 mJ used here is roughly fifty times over
threshold and deep into saturation, where raising it further changes nothing.

Output lands in `simulation_output/` (see above); `charges.txt` holds the final
per-atom charges.

## Testing

- **`test/invariance.sh`**: bitwise invariance, 23 short cases (water and
  lysozyme, -nt 1 and -nt 4 -pd, autostop, fixed charges, and each cloud option
  in turn, the polarization), with checks that hold on any build (event counts against charges,
  no clamped cloud energy, escape + trap = events, ...). By default it compares
  with stored checksums from a frozen build (`test/invariance/reference.md5`,
  machine-specific). `--ref-bindir DIR` instead runs a reference build next to
  it and compares the two directly. Run it after every change to the code.
- **`test/regression.sh`**: the original water-cluster check, comparing event
  counts and charges with `test/reference_2000.txt`.
- **`test/pb_vs_mc.sh`**: the field-free cloud against gromacs-mc, bit for bit,
  on a seeded cloud. Needs a gromacs-mc build with `test/mc_pb_seed.patch`.

## Limitations

### Electron cloud

- **One temperature, Maxwellian.** The cloud is in equilibrium at T_e, and the
  solve does not go below 1 eV. At 10^15 the cloud cools below that after
  about 100 fs (adiabatic expansion), where the model is least reliable.
- **Cycle-averaged laser.** Only the envelope E0(t) acts on the cloud, except
  for the spill and evaporation with `sfmd-laser-barrier = 2`. There is no
  quiver motion, no field direction and no polarization dependence; the barrier
  is lowered in all directions, not only along the polarization.
- **Undamped resonance without surface damping.** With
  `sfmd-heating-surface = 0` only electron-ion collisions damp the Drude
  resonance. A hot, thin cloud (nu/omega ~ 1e-5) can then absorb MeV within a
  step when a solve lands on the resonance: in 1 of 8 runs at 10^16–10^17, +3 %
  ion energy at 10^16. `sfmd-heating-surface = 1` (nu_s = vbar/R, ~0.5–0.9
  omega here) removes it.
  - The seed spread drops (10^16: ±90 → ±28 keV).
  - Heating rises by 6 % at 10^15 and by about 50 % at 10^16 and 10^17.
  - At 10^16 the cloud leaves earlier and completely, so the ion energy rises
    by 7 %, 9 % above cloud-off.
  - The directions stay within the noise.
  - Weakly sensitive to A (10^16, 8 seeds): A = 0.5 / 1 / 2 give 7806 / 7912 /
    7986 keV, about +1.4 % per doubling of A; directions identical within noise.
- **The collision-rate cap is not fitted.** `sfmd-heating-nu-cap = 1` limits
  nu_ei to omega_p (strong coupling). It matters only while the cloud is cold
  and dense: at 10^15, cap 0.5 instead of 1 gives 16 % less heating and 8 %
  less ion energy; at 10^16 no change. No cap gave the same as cap 1 (earlier
  scan, without collisions and surface damping). Directions unchanged in all.
  A = 1 is the usual order of magnitude, not a fitted value.
- **Valence shell only.** Collisions use the same charge states as ADK (H 1, C
  4, N 5, O 6, S 6 electrons), so no inner-shell ionization.
- **Recombination into the ground state.** In reality three-body capture goes
  into highly excited states; during the pulse this overstates recombination
  and its heating (see `sfmd-coll-recombination = 1` above).
- **Rates limited by the time step.** At the high local densities at the atoms
  (up to 10^24 cm^-3), collision rates can exceed 1/dt. At most one event per
  atom and step is drawn, so the step limits them; this matters only for the
  fast cycling.
- **Autostop 2 and the schedule, tested on lysozyme** (10^15 and 10^16, 8 seeds,
  against 300 fs runs). `sfmd-autostop = 2` (defaults) stopped at 125–130 fs
  (10^15) and 95 fs (10^16); the directions still turn 0.4–2.2 deg after the
  stop (seed spread 1–5 deg) and the ion energy is ~6 % short of its 300 fs
  value; 2.5–3.3x less wall time. The schedule `65:0.01:0.2 95:0.1:2 135:0.1:5`
  moved the directions by up to 1.2 deg and the ion energy by -0.2 % at
  300 fs, 2.8–3.1x faster. Compare runs stopped/scheduled the same way.
- **Long range only tested on lysozyme.** `sfmd-pb-long-range = 1` is
  gromacs-mc's grid long range (refreshed every `sfmd-pb-interval`). On
  lysozyme at 10^16 with a 2 nm cutoff: ion energy +0.6 % against all pairs,
  directions within the seed noise, force error 1 % rms (3.5 % at most); with
  a 1 nm cutoff +2.5 %. It only pays above ~10000 atoms (gromacs-mc: 1.9x at
  GroEL), which `auto` uses; the speed-up is not measured in this build.
  Without it, a cutoff loses the net charge's push and the sample barely
  explodes (grompp warns).
- **No restart.** The cloud's state is not checkpointed, and
  `sfmd-initial-charges` is refused with it.

### Numerical stability

For high ionization we get huge forces, which can make the numerical integration unstable. If you suspect this, check the kinetic and potential energy of the system — as long as they look reasonably smooth it should be okay. The workaround is usually to lower the step size; 1 as is a reasonable starting point.

### Parallelization

Multithreaded runs require `-pd` rather than the default domain decomposition (see "Running a simulation" above).

### Compatibility

Has only been tested on Linux systems.

### Renamed parameters

Old legacy parameters, does not work anymore, kept here for reference. `grompp`
rejects an `.mdp` that still uses the old names and prints the replacement
for each one, because several changed meaning as well as name:

| old | new |
|---|---|
| `userint1` | removed - `-ionize` alone enables the altered force field |
| `userint2`, `userint4`, `userint6` | removed  |
| `userint3` | `sfmd-autostop` |
| `userint5` | `sfmd-detailed-output` |
| `userint7` | `sfmd-charge-output-stride` - **not the same knob**, see below |
| `userint9` | `sfmd-initial-charges` |
| `userreal1` | `sfmd-pulse-peak-time` - **now fs**, multiply by 1000 |
| `userreal2` | `sfmd-pulse-energy` |
| `userreal3` | `sfmd-pulse-fwhm` - **now the FWHM in fs**, multiply by 2354.82 |
| `userreal4` | `sfmd-pulse-focal-diameter` |
| `userreal5` | `sfmd-pulse-wavelength` |
| `userreal6` | `sfmd-autostop-energy-ratio` |
