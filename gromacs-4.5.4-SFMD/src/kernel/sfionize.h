/*
 * sfionize.h
 *
 * The MolDStruct strong-field ionization module: ADK tunnelling driven by a
 * Gaussian laser pulse, followed by the Coulomb explosion of the ionized
 * system.
 *
 * Enabled by running mdrun with -ionize.  The mdp options it reads are
 * documented in sfionize.c above sfionize_init().
 *
 * This file is part of the MolDStruct modifications to GROMACS 4.5.4 and is
 * distributed under the same licence as GROMACS.
 */

#ifndef _sfionize_h
#define _sfionize_h

#include <stdio.h>

#include "typedefs.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Opaque module state.  One instance per mdrun. */
typedef struct t_sfionize t_sfionize;

/* Set up the module: build the ADK rate tables, initialise the laser pulse
 * and the random stream, create ./simulation_output/ and open the per-step
 * output.  Call once before the MD loop.  Returns NULL if -ionize was not
 * given, in which case every other entry point is a no-op.
 *
 * Also raises the moldstruct_altered_ff flag that selects the modified
 * bonded/non-bonded terms in libgmx: the altered force field and the
 * ionization are enabled together, by -ionize alone. */
t_sfionize *sfionize_init(FILE *fplog, const t_inputrec *ir,
                          t_mdatoms *mdatoms, const t_commrec *cr, int nHarmonicBonds);

/* Advance the ionization state by one MD step: ADK trials for every home
 * atom, then the per-step output.  Updates mdatoms->chargeA, which must
 * happen before do_force() so that this step's forces see this step's
 * charges.
 *
 * Under particle decomposition each rank only updates its own home range, so
 * this re-syncs the full charge array across ranks before returning.  Without
 * that, every cross-rank pair - about (nranks-1)/nranks of all pairs - would
 * be evaluated against a stale charge for the far atom, in both the forces
 * and the reported energies. */
void sfionize_step(t_sfionize *sf, const t_inputrec *ir, t_mdatoms *mdatoms,
                   rvec *x, double t, gmx_large_int_t step,
                   gmx_bool bFirstStep, const t_commrec *cr, t_nrnb *nrnb);

/* After do_force(): the force of the trapped-electron cloud on the home
 * atoms (sfmd-electrons = pbgrid; a no-op without the cloud or -ionize).
 * x must hold every atom, which sfionize_step() ensures with the cloud on. */
void sfionize_screen_forces(t_sfionize *sf, const t_inputrec *ir,
                            t_mdatoms *mdatoms, rvec *x, rvec *v, rvec *f,
                            const t_commrec *cr);

/* Autostop test (sfmd-autostop): TRUE once E_kin/E_tot exceeds
 * sfmd-autostop-energy-ratio.  Kept out of the step function because do_md only
 * knows the energies later in the step.
 *
 * bEnergiesGlobal must be do_md's bGStat: enerd->term[] is only reduced
 * across ranks on global-communication steps, and on every other step it
 * holds this rank's partial sums, so the ratio is not the system's.  The
 * decision is taken on one rank and broadcast, because every rank has to
 * reach the same answer - if one leaves the MD loop and the others do not,
 * they hang in the next collective. */
gmx_bool sfionize_autostop(t_sfionize *sf, const t_inputrec *ir,
                           double E_kin, double E_tot, gmx_large_int_t step,
                           gmx_bool bEnergiesGlobal, const t_commrec *cr);

/* sfmd-autostop = 2: TRUE once the median heavy-ion direction turn still to
 * come, extrapolated from the last sfmd-autostop-interval, is below
 * sfmd-autostop-angle (and t is sfmd-autostop-min-time past the pulse peak).
 * Called on every rank every step; decided on the master and broadcast. */
gmx_bool sfionize_dirstop(t_sfionize *sf, const t_inputrec *ir, const t_mdatoms *md, rvec *v,
                          gmx_large_int_t step, double t, const t_commrec *cr);

/* sfmd-schedule (as gromacs-mc's mcmd-schedule).  sfmd_sched_stage: the
 * stage of MD step `step`, its first step and start time (ps); dt0 the tpr's
 * dt (ps).  sfmd_sched_dt / _pb: stage k's dt (ps) and PB interval (fs).
 * sfmd_sched_steps_to: MD steps from step 0 to t_end (ps).
 * sfionize_schedule: set the module's step-counted intervals for the stage
 * (call with ir->delta_t already the stage's); bChange: it starts now, after
 * a stage with step dt_prev (ps). */
int             sfmd_sched_stage(const t_inputrec *ir, double dt0, gmx_large_int_t step, gmx_large_int_t *step0,
                                 double *t0);
double          sfmd_sched_dt(const t_inputrec *ir, int k, double dt0);
double          sfmd_sched_pb(const t_inputrec *ir, int k);
gmx_large_int_t sfmd_sched_steps_to(const t_inputrec *ir, double dt0, double t_end);
void            sfionize_schedule(t_sfionize *sf, FILE *fplog, const t_inputrec *ir, int stage, double pb_fs,
                                  gmx_large_int_t step0, double t0, gmx_bool bChange, double dt_prev);

/* Write the final per-atom charge dump and release everything.  Safe to call
 * with sf == NULL. */
void sfionize_done(t_sfionize *sf, t_mdatoms *mdatoms, int natoms,
                   const t_commrec *cr);

#ifdef __cplusplus
}
#endif

#endif /* _sfionize_h */
