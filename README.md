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

The parameters of the model can be set like any other MD parameters in the `.mdp` file and are accessed through the userints and userreals.
Userints are used to enable/disable certain features while the userreals are used for the physical pulse parameters.

```
userint1 - (default = 1) Alter forcefield. If set to zero everything should run as unmodified gromacs 4.5.4 (hopefully)
userint2 - (default = 1) Do charge transfer. Enables the charge transfer module
userint3 - (default = 0) Autostop simulation once E_kin/E_tot exceeds a threshold (threshold set by userreal6)
userint4 - (default = 0) Read electronic states from file. Useful for continued sims (Experimental)
userint5 - (default = 0) Enable logging of electronic dynamics. Big performance drop due to I/O.
userint6 - (default = 0) Enable collisional ionization. (Not implemented yet). Requires collisional data.
userint7 - (default = 0, treated as 1) Charge-logging interval in steps: charges_over_time.bin / mean_charge_vs_time.txt are gathered across ranks and flushed every userint7 steps instead of every step.
userint9 - (default = 0) Read charges from a file. File must be named "charges.txt" and placed in the same directory from where mdrun is called. File consists of two columns, atom index and charge.

userreal1 - Pulse center / peak time [ps]
userreal2 - (default = 0.0) Pulse energy [J]
userreal3 - (default = 0.0) Pulse sigma, width of the peak (sigma value of the gaussian) [ps]
userreal4 - (default = 0.0) Diameter of the focal spot [nm]
userreal5 - (default = 0.0) Wavelength [nm]
userreal6 - (default = 0.99) Threshold for energy autostop (userint3)
```

**Important:** these parameters are only read when `mdrun` is called with the `-ionize` flag.
The assignment happens inside the ionization block, so without `-ionize` the run behaves as unmodified GROMACS 4.5.4 no matter what the `.mdp` says.

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

When `userint5` is set to 1, additional output is written to `simulation_output/` (created automatically in the directory `mdrun` is called from):

- **`pulse_profile.txt`** — intensity of the laser pulse at each timestep. Column 1: time [ps]. Column 2: laser intensity.
- **`mean_charge_vs_time.txt`** — mean charge of the system over time, flushed every `userint7` steps. Column 1: time [ps]. Column 2: mean charge [e].
- **`charges_over_time.bin`** — per-atom charge, logged in binary, same flush interval as above.
- **`masses.bin`** — per-atom mass, written once.
- **`transition_log.txt`** — log of ionization/electronic transition events and the time [ps] they occur.
- **`charges.txt`** — final per-atom charge dump at the end of the run (also the file read back in via `userint9` for restarting/chaining a simulation).

If you do not need this data, it is recommended to leave `userint5` off, as it cuts into performance.

## Example

_(TODO: add a worked example,`.mdp`, `pdb2gmx`/`grompp`/`mdrun` commands, and expected output.)_

## Limitations

### Numerical stability

For high ionization we get huge forces, which can make the numerical integration unstable. If you suspect this, check the kinetic and potential energy of the system — as long as they look reasonably smooth it should be okay. The workaround is usually to lower the step size; 1 as is a reasonable starting point.

### Parallelization

Multithreaded runs require `-pd` rather than the default domain decomposition (see "Running a simulation" above).

### Compatibility

Has only been tested on Linux systems.
