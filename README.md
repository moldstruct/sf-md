# Hybrid Strong-Field / Molecular Dynamics (SF/MD)

This is a strong-field ionization Molecular Dynamics (SF/MD) model that is part of the **MOLDSTRUCT** toolbox. It can be used to simulate laser-induced ionization in the optical regieme based on ADK theory and the resulting Coulomb explosion of molecular systems, from small clusters up to whole proteins.
The code is based on a modified version of GROMACS ([webpage](https://www.gromacs.org/)), with functionality close to normal GROMACS plus additional parameters (an `-ionize` module) that drive ionization from a laser pulse.
The model is developed by the Biophysics group at Uppsala University.

# Manual

This is a brief manual that will cover how to install, use the model and evaluate output.
Basic knowledge of GROMACS is assumed, check out [GROMACS webpage](https://www.gromacs.org/) for more information.

## Contents

- Installation
- List of input parameters
- Running a simulation
- Output
- Example
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
sfmd-autostop               (default = 0)    Stop once E_kin/E_tot exceeds
                                             the threshold below
sfmd-autostop-threshold     (default = 0.99) That threshold
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
- **`charges_over_time.bin`** — per-atom charge, logged in binary, one
  frame every `sfmd-charge-output-stride` steps (default 50). Frame *k* is
  step *k* x stride, so take every stride'th row of
  `mean_charge_vs_time.txt` for the matching times.
- **`masses.bin`** — per-atom mass, written once.
- **`transition_log.txt`** — log of ionization/electronic transition events and the time [ps] they occur.
- **`charges.txt`** — final per-atom charge dump at the end of the run (also the file read back in via `sfmd-initial-charges` for restarting/chaining a simulation).

If you do not need this data, it is recommended to leave
`sfmd-detailed-output` off, as it cuts into performance.

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

## Limitations

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
| `userreal6` | `sfmd-autostop-threshold` |
