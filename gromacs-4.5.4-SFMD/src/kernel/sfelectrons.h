/*
 * sfelectrons.h
 *
 * The trapped electrons of the MolDStruct strong-field module
 * (sfmd-electrons = pbgrid): the Poisson-Boltzmann electron cloud of
 * gromacs-mc (mcmd-electrons = pbgrid), ported from mc-md 1eb798f.  The
 * electrons released by ADK tunnelling either escape or join a cloud that is
 * the equilibrium density of N electrons at one temperature in the field of
 * the ions and of themselves, recomputed every sfmd-pb-interval on a grid
 * that follows the sample.  The cloud screens the ions (forces), decides
 * whether a new electron escapes (the depth at its atom), and evaporates as
 * the well becomes shallow.  The model, the numerics and the output are those
 * of the gromacs-mc manual, chapters "Trapped electrons" and "Numerics of the
 * electron model".
 *
 * Tunnelled electrons are born at rest at their atom.  Without
 * sfmd-laser-barrier the laser does not act on the cloud (field-free cloud):
 * the escape and evaporation depths are the cloud's own.  With it, the
 * laser's envelope field E0(t) lowers the escape level of the net charge Q by
 * 2 sqrt(k Q e E0) (sfel_barrier): newborn electrons above it escape, cells
 * above it spill out after each solve, and evaporation runs above it.
 *
 * Call order in one MD step (all ranks unless noted):
 *   sfelec_step_begin   gather x, geometry, evaporation (master)
 *   sfelec_birth        per ADK event, master only: escape or trap
 *   sfelec_step_end     energy ledger (master), broadcast of the cloud scalars
 *   do_force()
 *   sfelec_forces       PB solve when due (master), forces on the home atoms
 *
 * This file is part of the MolDStruct modifications to GROMACS 4.5.4 and is
 * distributed under the same licence as GROMACS.
 */

#ifndef _sfelectrons_h
#define _sfelectrons_h

#include <stdio.h>

#include "typedefs.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct t_sfelec t_sfelec;

/* NULL unless sfmd-electrons = pbgrid. */
t_sfelec *sfelec_init(FILE *fplog, const t_inputrec *ir, const t_commrec *cr);

/* The laser's envelope field this step (V/m), for sfmd-laser-barrier.
 * Every rank, before sfelec_step_begin. */
void sfelec_set_laser(t_sfelec *el, double E0_V_m);
/* The instantaneous field E(t) (V/m, carrier included), for sfmd-laser-barrier = 2. */
void sfelec_set_laser_now(t_sfelec *el, double E_V_m);

/* Start of the step, before the ionization: gathers all coordinates onto
 * every rank (full PD ring), then on the master the sample geometry (the
 * reference on the first step) and evaporation.  On the first step, after the
 * charges have been initialised. */
void sfelec_step_begin(t_sfelec *el, const t_inputrec *ir, t_mdatoms *md,
                       rvec *x, double t, gmx_bool bFirstStep,
                       const t_commrec *cr, t_nrnb *nrnb);

/* Master only: an electron of kinetic energy eps (eV) released at atom k
 * escapes (TRUE) or is trapped (FALSE, booked into the cloud).  The charge of
 * k may already be raised; k itself is left out of the depth. */
gmx_bool sfelec_birth(t_sfelec *el, const t_mdatoms *md, rvec *x, int k,
                      double eps);

/* sfmd-collisions, master only.  sfelec_coll_Te: the cloud's T_e (eV) when
 * impact ionization can act this step, else 0.  sfelec_coll_ne: the electron
 * density at atom k, cm^-3 (0 outside the cloud).  sfelec_coll_event: atom k
 * (its charge already raised) was ionized by impact, the cloud pays P (eV)
 * and keeps the new electron; FALSE (nothing booked) if the cloud has less
 * than P, and the caller undoes the charge. */
double   sfelec_coll_Te(const t_sfelec *el);
double   sfelec_coll_ne(const t_sfelec *el, int k);
/* sfmd-schedule: set the cloud's step-counted intervals for a stage (see
 * sfionize_schedule); sfelec_schedule_str: a summary for the log ("" without
 * the cloud). */
void        sfelec_schedule(t_sfelec *el, const t_inputrec *ir, double pb_fs, gmx_large_int_t step0, double t0,
                            gmx_bool bChange, double dt_prev);
const char *sfelec_schedule_str(const t_sfelec *el);
/* impact ionizations and recombinations so far (master; 0 without the cloud) */
void        sfelec_counts(const t_sfelec *el, double *n_coll, double *n_rec);
/* sfmd-coll-recombination's T_e: as sfelec_coll_Te, but at least sfmd-te-min */
double   sfelec_rec_Te(const t_sfelec *el);
gmx_bool sfelec_coll_event(t_sfelec *el, const t_mdatoms *md, rvec *x, int k, double P);
/* sfmd-coll-recombination, master only: is it on; atom k (its charge already
 * lowered) captured a cloud electron, the cloud gets P (eV); FALSE (nothing
 * booked) if the cloud has no electron left, and the caller undoes the charge. */
int      sfelec_rec_on(const t_sfelec *el);   /* 0 off, 1 always, 2 after the pulse */
gmx_bool sfelec_rec_event(t_sfelec *el, const t_mdatoms *md, rvec *x, int k, double P);

/* End of the ionization: books the cloud-ion energy of the charge changes
 * (master) and broadcasts the cloud scalars.  The charges must already be
 * the same on every rank. */
void sfelec_step_end(t_sfelec *el, const t_mdatoms *md, rvec *x,
                     const t_commrec *cr);

/* After do_force(): the PB solve when due and the cloud's force on the home
 * atoms, its work booked; screening_log.txt and energy_ledger.txt. */
void sfelec_forces(t_sfelec *el, const t_inputrec *ir, t_mdatoms *md,
                   rvec *x, rvec *v, rvec *f, const t_commrec *cr);

/* Summary to the log, close files, free.  Safe with el == NULL. */
void sfelec_done(t_sfelec *el, FILE *fplog, const t_commrec *cr);

#ifdef __cplusplus
}
#endif

#endif /* _sfelectrons_h */
