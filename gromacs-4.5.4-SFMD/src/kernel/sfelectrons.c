/*
 * sfelectrons.c
 *
 * The trapped-electron cloud of the MolDStruct strong-field module
 * (sfmd-electrons = pbgrid), ported from gromacs-mc (mc-md 1eb798f,
 * src/kernel/mcionize.c).  See sfelectrons.h for what it does and the call
 * order, and the gromacs-mc manual (chapters "Trapped electrons: the
 * Poisson-Boltzmann model" and "Numerics of the electron model") for the
 * model, which this file follows exactly.
 *
 * Porting notes.  The grid, solver, forces, ledger and evaporation are
 * mcionize.c's code with mechanical renames (mcmd_ -> sfel_, the module
 * struct t_mcionize -> t_sfelec, mc -> el); the comments are mc-md's.  What
 * differs:
 *   - the electrons come from ADK tunnelling and are born at rest at their
 *     atom (eps = 0): no photo/Auger energies, no collisions, no charge
 *     transfer, no field ionization in the PB field;
 *   - the module is called from sfionize.c, which runs ADK on the master for
 *     every atom while the cloud is on (one stream, as mc-md's Monte Carlo);
 *   - no restart state: grompp refuses sfmd-initial-charges with the cloud;
 *   - the long-range ion force from the grid (sfmd-pb-long-range) is
 *     mc-md's (as of 6e81e9f: refreshed every sfmd-pb-interval);
 *   - escape and evaporation depths go through sfel_escape_depth() and
 *     sfel_cell_depth(), where the laser will lower the barrier later.
 * The diagnostics of mc-md stay behind the same switches, renamed
 * GMX_SFMD_*: TIMING_DETAIL, PB_EDIAG, PBG_DENSITY, PBG_OUTLAT_CHECK,
 * PBG_FFT_THREADS.
 *
 * This file is part of the MolDStruct modifications to GROMACS 4.5.4 and is
 * distributed under the same licence as GROMACS.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <fftw3.h>
#if defined(GMX_THREADS)
#include <pthread.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "typedefs.h"
#include "smalloc.h"
#include "vec.h"
#include "network.h"
#include "mvdata.h"
#include "partdec.h"
#include "gmx_fatal.h"
#include "sfelectrons.h"

#define SFEL_COULOMB_EV_NM 1.43996454   /* e^2 / (4 pi eps0), eV nm   */
#define SFEL_EV_TO_KJMOL   96.4853321

/* PB grid constants, as gromacs-mc (settled by its box and speed tests of
 * 2026-10-23..26): the adaptive box is at least SFEL_PBG_BOXMIN R and shrinks
 * only to where at most SFEL_PBG_SCUT of the electrons are outside (safe
 * shrink), resizing to SFEL_PBG_BOXMARG x the wanted or safe size; the kernel
 * smoothing is s = max(SFEL_PBG_SFAC dx, s_e); outside ions use sub-lattices
 * of stride dx <= distance / SFEL_PBG_OUTLAT_D, except those within
 * SFEL_PBG_MARG cells of the box, which go into one larger FFT. */
#define SFEL_PBG_BOXMIN   1.3
#define SFEL_PBG_SCUT     0.001
#define SFEL_PBG_BOXMARG  1.05
#define SFEL_PBG_SFAC     2.0
#define SFEL_PBG_OUTLAT_D 3.0
#define SFEL_PBG_MARG     16

#define SFEL_OUTDIR "./simulation_output"

/* GMX_SFMD_TIMING_DETAIL timers (td[], ms) */
#define SFEL_TD_CONV    0  /* grid convolutions, total                             */
#define SFEL_TD_FFT     1  /* of which the FFTs                                    */
#define SFEL_TD_CG      2  /* grid CG, total (convolutions included)               */
#define SFEL_TD_CGCONV  3  /* convolutions inside the CG                           */
#define SFEL_TD_NEWTON  4  /* grid Newton loop, total                              */
#define SFEL_TD_EVAP    8  /* evaporation                                          */
#define SFEL_TD_EXTRAP  9  /* the extrapolation test itself (one convolution)      */
#define SFEL_TD_OUTPOT 10  /* potential of the ions outside the grid box           */

struct t_sfelec
{
    /* The cloud scalars (the master's; broadcast every step) */
    double num_free_electrons;      /* N, trapped                              */
    double escaped_photoelectrons;  /* Q, every electron that left (name kept
                                       from mc-md, where it is used the same way) */
    double screen_E_trapped;        /* kinetic energy of the trapped e-, eV    */

    /* Sample geometry: heavy-atom centre and gyration ellipsoid */
    gmx_bool geom_heavy;
    double   com[3];
    double   radius_now;            /* (a1 a2 a3)^(1/3), nm                    */
    double   radius_of_sample;      /* at the first step                       */
    double   ell_axis[3][3];
    double   ell_a[3];

    /* Counters and logs (sf) */
    double  count_trapped, count_escaped_birth;
    double  eb_dep, eb_evap;

    int     screen_mode;            /* 0 off, 5 pbgrid (mc-md's numbering)     */
    gmx_bool screen_evap;
    double  screen_lambda_nm, screen_ND, screen_evaporated;
    FILE   *fscreen;
    double  pressure_h_now, pressure_work;
    double  tm[10];

    /* From here on as in mcionize.c's struct t_mcionize */
    int      pb_nstep;
    double  *pb_ne;
    double   pb_N, pb_h, pb_Dmean;
    double   pb_W;
    double   pb_Eclamp;
    gmx_bool pb_etot_init;
    double   pb_Etot;
    gmx_bool pb_ediag;
    double   pbd_W, pbd_J, pbd_N, pbd_Ulast;
    real    *pbd_q;
    FILE    *fpbd;
    gmx_bool pb_ready;
    gmx_large_int_t pb_step_solve;
    int      pb_nsolve;
    FILE    *fpb, *fpbw;
    gmx_bool pbg_dens_out;
    gmx_bool tdet;
    double   td[11];
    double   td_rhs2;
    double   td_nsolve, td_nskip, td_newton0, td_newtonex;
    double  *pbg_nprev;
    gmx_bool pbg_nprev_ok;
    FILE    *ftd;
    float   *led_q;
    double   led_Jii, led_Jie;
    double   led_nchg;
    FILE    *fled;
    int      pb_evn;
    int     *pb_evlist, pb_evnl;
    int      pbg_nthreads;
    long     park_seq;
    double  *pb_evw, *pb_evV;
    int      pb_evh_ver;
    double  *pb_evhw, *pb_evhV;

    int      pbg_n;
    int      pbg_nmin, pbg_nmax, pbg_ncap;
    int      pbg_capset;
    int      pbg_nglimit;
    int      pbg_nplan;
    double   pbg_dxt;
    double   pbg_R0;
    double   pbg_lo[3], pbg_dx;
    double   pbg_R, pbg_c[3];
    double   pbg_box;
    double   pbg_cut;
    double   pbg_hbox;
    FILE    *fbox;
    double   pbg_se;
    double   pbg_ntol;
    int      pbg_outlat;
    FILE    *fol;
    FILE    *fop;
    gmx_bool pbg_have;
    int      pbg_nregrid, pbg_nconv;
    double  *pbg_ne;
    double  *pbg_Phi, *pbg_V, *pbg_g, *pbg_d, *pbg_t, *pbg_Kn, *pbg_nt, *pbg_Vt, *pbg_cg;
    float   *pbg_Ve;
    float   *pbg_buf, *pbg_Kh;
    fftwf_complex *pbg_hat;
    fftwf_plan pbg_fwd, pbg_bwd;
    fftwf_plan pbg_fwdE, pbg_bwdE;
    float   *pbg_KhE;
    double   pbg_K0, pbg_ecen[3];
    gmx_large_int_t screen_step;

    /* sfmd-laser-barrier: the laser envelope lowers the escape level */
    gmx_bool laser_on;
    int      barrier_mode;          /* sfmd-laser-barrier: 1 envelope, 2 instantaneous for spill/evaporation */
    double   E_now;                 /* |E(t)| this step, V/nm (barrier mode 2) */
    double   E0;                    /* envelope field this step, V/nm         */
    double   pb_evh_dW;             /* the lowering the depth histogram was made with, eV */
    double   pb_Z;                  /* ion charge Q + N at the last solve     */
    double   pb_Tmin;               /* temperature floor of the solve, eV     */
    double   te_min;                /* sfmd-te-min, eV; 0: the old model      */
    double   te_floor_steps;        /* steps with the actual T_e below te_min */
    double   te_floor_dt;           /* MD step, fs, for that report           */
    /* sfmd-pb-long-range: long-range ion-ion Coulomb from the grid (GROMACS does r < rcoulomb) */
    int      pbg_lr, pbg_lr_nstep, lr_nout, lr_headn, lr_nc[3];
    gmx_bool lr_ready;
    gmx_large_int_t lr_step;
    double   lr_rc, lr_W, lr_U, lr_Ush, lr_cs, lr_clo[3];
    double  *pbg_PhiL;            /* master: ion potential on the grid, eV per e */
    float   *pbg_PhiLf;           /* every rank: the same, single precision */
    double  *lr_E;                /* long-range field at the home atoms, eV/nm per e, 3 per atom */
    int     *lr_head, *lr_next;   /* cell list of the charged atoms (cells >= rc) */
    FILE    *flr;
    /* sfmd-schedule: the module time of step s is sched_t0 + (s - sched_step0) dt
     * (sched_t0 = tinit, sched_step0 = 0 without a schedule) */
    gmx_large_int_t sched_step0;
    double   sched_t0;
    gmx_bool sched_on;
    char     sched_str[96];
    /* sfmd-laser-heating */
    gmx_bool heat_on;
    double   heat_cap;              /* nu <= heat_cap omega_p (0: no cap)     */
    int      heat_shape;            /* 0 sphere, 1 ellipsoid (sfmd-heating-shape) */
    double   heat_surf;             /* surface damping A (sfmd-heating-surface), 0 off */
    double   heat_L[3];             /* last step: depolarization factors (ellipsoid) */
    double   omega_L;               /* laser angular frequency, rad/s         */
    double   heat_P, heat_nu_w, heat_Ein, heat_Zeff; /* last step: eV/fs, nu/omega, |E_in|/E0, Z_eff */
    double   eb_laser;              /* energy put in by the laser, eV         */
    /* sfmd-collisions */
    gmx_bool coll_on;
    double   count_coll, eb_coll;   /* impact ionizations; ionization energy the cloud paid, eV */
    int      rec_on;                /* sfmd-coll-recombination: 0 off, 1 always, 2 after the pulse */
    double   count_rec, eb_rec;     /* recombinations; binding energy the cloud got, eV */
    double  *pb_evhVt, *pb_evhK;    /* per depth bin: mean true potential V and cloud
                                       potential K n, eV (laser barrier only)  */
    double   count_spilled, eb_spill, spill_last;
    FILE    *flaser;                /* laser_log.txt                          */
    /* sfmd-laser-polarization: the instantaneous field along one axis in the solve */
    int      pol_axis;              /* -1 off; 0, 1, 2: x, y, z               */
    double   E_sig;                 /* E(t) this step along the axis, V/nm, signed */
    double   pol_E;                 /* the field the cloud was solved in, V/nm, signed
                                       (without a cloud: E(t) of this step)    */
    double   pol_p[3];              /* dipole of ions and cloud about com at the last solve, e nm */
    double   pol_X;                 /* sum over the trapped electrons of x_axis, nm (tracked
                                       through births, captures, evaporation, spills) */
    double   eb_pol;                /* polarization work of the field on the cloud, eV (cumulative) */
    double   pol_nosaddle;          /* solves without a saddle (the dipole term wins) */
    double  *pbg_UL;                /* master: the laser's potential energy per cell, eV */
    double  *pb_evhX;               /* per depth bin: mean x_axis of the cells, nm */
    int      pol_logged;            /* laser_log.txt has the polarization header */

    gmx_bool pbg_cold;              /* TRUE after the cloud spilled out completely:
                                       the next solve starts from the ions    */
    gmx_bool pbg_rst_grid;          /* always FALSE                           */
};

/* ---------------------------------------------------------------- *
 * Small helpers
 * ---------------------------------------------------------------- */

/* An mdp real as the user wrote it (grompp stores floats): 0.2 is 0.2. */
static double sfel_mdp_real(real r)
{
    char buf[32];

    sprintf(buf, "%.7g", (double)r);
    return atof(buf);
}

static FILE *sfel_open_output(const char *name, const char *mode)
{
    char  path[512];
    FILE *fp;

    sprintf(path, "%s/%s", SFEL_OUTDIR, name);
    fp = fopen(path, mode);
    if (fp == NULL)
    {
        gmx_fatal(FARGS, "Could not open '%s' for mode '%s'.", path, mode);
    }

    return fp;
}

/* Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations.
 * A is destroyed; lam gets the eigenvalues and the rows of V the matching
 * unit eigenvectors. */
static void sfel_jacobi3(double A[3][3], double lam[3], double V[3][3])
{
    int i, j, k, sweep;

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++) { V[i][j] = (i == j); }
    }
    for (sweep = 0; sweep < 50; sweep++)
    {
        double off = sqr(A[0][1]) + sqr(A[0][2]) + sqr(A[1][2]);

        if (off < 1e-30 * (sqr(A[0][0]) + sqr(A[1][1]) + sqr(A[2][2]) + 1e-300))
        {
            break;
        }
        for (i = 0; i < 2; i++)
        {
            for (j = i + 1; j < 3; j++)
            {
                double th, t, c, sn;

                if (A[i][j] == 0) { continue; }
                th = (A[j][j] - A[i][i]) / (2.0 * A[i][j]);
                t  = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0));
                c  = 1.0 / sqrt(t * t + 1.0);
                sn = t * c;
                for (k = 0; k < 3; k++)            /* A <- A G (columns i, j) */
                {
                    double aki = A[k][i], akj = A[k][j];

                    A[k][i] = c * aki - sn * akj;
                    A[k][j] = sn * aki + c * akj;
                }
                for (k = 0; k < 3; k++)            /* A <- G^T A (rows i, j) */
                {
                    double aik = A[i][k], ajk = A[j][k];

                    A[i][k] = c * aik - sn * ajk;
                    A[j][k] = sn * aik + c * ajk;
                }
                for (k = 0; k < 3; k++)            /* eigenvectors as rows */
                {
                    double vik = V[i][k], vjk = V[j][k];

                    V[i][k] = c * vik - sn * vjk;
                    V[j][k] = sn * vik + c * vjk;
                }
            }
        }
    }
    for (i = 0; i < 3; i++) { lam[i] = A[i][i]; }
}


/* Weight of atom i in the sample geometry: its mass, or 0 for hydrogen
 * (geom_heavy, always on: fast protons leave the electron cloud behind and
 * would otherwise stretch R and the ellipsoid). */
#define SFEL_GEOM_W(i) ((el->geom_heavy && mdatoms->massT[i] < 1.5) ? 0.0 : (double)mdatoms->massT[i])

static void sfel_sample_geometry(t_sfelec *el, const t_mdatoms *mdatoms,
                                 rvec *x, double *rg)
{
    double mtot = 0, sum = 0;
    int    i, d;

    el->com[0] = el->com[1] = el->com[2] = 0;
    for (i = 0; i < mdatoms->nr; i++)
    {
        for (d = 0; d < 3; d++)
        {
            el->com[d] += SFEL_GEOM_W(i) * x[i][d];
        }
        mtot += SFEL_GEOM_W(i);
    }
    for (d = 0; d < 3; d++)
    {
        el->com[d] /= mtot;
    }
    for (i = 0; i < mdatoms->nr; i++)
    {
        sum += SFEL_GEOM_W(i) * (sqr(x[i][0] - el->com[0]) +
                                    sqr(x[i][1] - el->com[1]) +
                                    sqr(x[i][2] - el->com[2]));
    }
    *rg = sqrt(sum / mtot);

    {
        double S[3][3] = {{0}}, lam[3];
        int    e;

        for (i = 0; i < mdatoms->nr; i++)
        {
            double dx[3];

            for (d = 0; d < 3; d++) { dx[d] = x[i][d] - el->com[d]; }
            for (d = 0; d < 3; d++)
            {
                for (e = 0; e < 3; e++) { S[d][e] += SFEL_GEOM_W(i) * dx[d] * dx[e]; }
            }
        }
        for (d = 0; d < 3; d++)
        {
            for (e = 0; e < 3; e++) { S[d][e] /= mtot; }
        }
        sfel_jacobi3(S, lam, el->ell_axis);
        for (d = 0; d < 3; d++)
        {
            /* a uniform ellipsoid has <x_i^2> = a_i^2 / 5; keep a floor so a
             * flat sample still has a finite thickness */
            el->ell_a[d] = sqrt(5.0 * (lam[d] > 1e-4 * sqr(*rg) ? lam[d] : 1e-4 * sqr(*rg)));
        }
    }
}


static double sfel_plasma_Te(const t_sfelec *el)
{
    if (el->num_free_electrons <= 0)
    {
        return 0.0;
    }
    return (2.0 / 3.0) * el->screen_E_trapped / el->num_free_electrons;
}

/* The temperature of the PB solve, the evaporation and three-body
 * recombination (gromacs-mc's mcmd_model_Te): the actual T_e, but at least
 * sfmd-te-min.  The actual T_e (above) keeps its own bookkeeping and may go
 * below the floor, or below 0 when the cloud's total energy is under the
 * potential energy of the floored density: the floor never adds energy.
 * sfmd-te-min = 0 is the old model: the solve floored at 1 eV, E_kin clamped
 * at 0 (E_clamped), evaporation at the actual T_e. */
/* Time (ps) of module step s: with sfmd-schedule from the current stage's start, else as GROMACS has it */
static double sfel_time(const t_sfelec *el, const t_inputrec *ir, gmx_large_int_t s)
{
    if (el->sched_on)
    {
        return el->sched_t0 + (s - el->sched_step0) * ir->delta_t;
    }
    return s * ir->delta_t + ir->init_t;
}

static double sfel_model_Te(const t_sfelec *el)
{
    const double T = sfel_plasma_Te(el);

    if (el->te_min > 0 && el->num_free_electrons > 0 && T < el->te_min)
    {
        return el->te_min;
    }
    return T;
}


#define SFEL_PB_SQRTPI 1.7724538509055159

/* Wall-clock timers of the module's sections, on the master (el->tm[], ms):
 * where the "Rest" of GROMACS's cycle accounting goes.  The force sections are
 * timed on the master rank only; under -pd the others do their share at the
 * same time. */
#define SFEL_TM_CT     0   /* charge-transfer pass                                */
#define SFEL_TM_PLASMA 1   /* geometry, plasma update, evaporation, rates         */
#define SFEL_TM_MC     2   /* Monte Carlo (photo, Auger, collisional, escape)     */
#define SFEL_TM_OUT    3   /* per-step output                                     */
#define SFEL_TM_STEP   4   /* the whole master part of sfelectrons_step              */
#define SFEL_TM_PBF    5   /* PB forces, solves included                          */
#define SFEL_TM_SCREEN 8   /* the whole of sfelectrons_screen_forces                 */
#define SFEL_TM_SOLVE  9   /* PB solves (master)                                  */

static double sfel_pb_G(double r, double s)
{
    return (r < 1e-6 * s) ? 2.0 / (s * SFEL_PB_SQRTPI) : erf(r / s) / r;
}

/* dG/dr */
static double sfel_pb_dG(double r, double s)
{
    double u = r / s;

    if (u > 6.0)
    {
        return -1.0 / (r * r);
    }
    return (2.0 / SFEL_PB_SQRTPI * u * exp(-u * u) - erf(u)) / (r * r);
}

static double sfel_pb_d2(const rvec a, const rvec b)
{
    return sqr(a[XX] - b[XX]) + sqr(a[YY] - b[YY]) + sqr(a[ZZ] - b[ZZ]);
}

static double sfel_pbg_depth(const t_sfelec *el, const t_mdatoms *md, rvec *x, int k);

/* Depth, eV (> 0 bound), of an electron at atom k: ions other than k and the
 * electron cloud, see sfel_pbg_depth(). */
static double sfel_pb_depth(const t_sfelec *el, const t_mdatoms *md, rvec *x, int k)
{
    return sfel_pbg_depth(el, md, x, k);
}

/* Master-only sections (the PB solve, the long-range ion potential): the
 * other thread-MPI ranks would spin in the broadcast that follows (sched_yield,
 * all their cores busy) and take the cores from the FFTW threads of the
 * master.  They sleep on a condition variable instead until the master is
 * done.  Each rank counts the sections it has passed (park_seq); the master
 * publishes its count when done, so a rank that arrives late does not wait.
 * Thread-MPI ranks are threads of one process; with other MPI (several
 * processes) this is a no-op. */
#if defined(GMX_THREADS)
static pthread_mutex_t sfel_park_mx   = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  sfel_park_cv   = PTHREAD_COND_INITIALIZER;
static long            sfel_park_done = 0;
#endif

static void sfel_park(t_sfelec *el, const t_commrec *cr)
{
#if defined(GMX_THREADS)
    if (!PAR(cr))
    {
        return;
    }
    el->park_seq++;
    pthread_mutex_lock(&sfel_park_mx);
    if (MASTER(cr))
    {
        sfel_park_done = el->park_seq;
        pthread_cond_broadcast(&sfel_park_cv);
    }
    else
    {
        while (sfel_park_done < el->park_seq)
        {
            pthread_cond_wait(&sfel_park_cv, &sfel_park_mx);
        }
    }
    pthread_mutex_unlock(&sfel_park_mx);
#endif
}

static double sfel_pb_wallms(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return 1e3 * tv.tv_sec + 1e-3 * tv.tv_usec;
}

/* ---------------------------------------------------------------- *
 * Self-consistent trapped electrons on a 3D grid (sfmd-electrons = pbgrid):
 * the full Poisson-Boltzmann solve.
 *
 * The electrons are a density on a cubic grid (cells per side chosen at each
 * rebuild from sfmd-pb-cells-per-smoothing, at least 24, at most
 * sfmd-pb-grid-limit) over at most +-sfmd-pb-box-max R about the heavy-atom
 * centre, cell counts n_i
 * (sum N).  The ions are deposited on the same grid (cloud-in-cell; ions
 * outside the box are added point by point).  Every cell is a Gaussian of
 * width sigma = the grid spacing, so two cells interact with
 *     K(r) = k erf(r / 2 sigma) / r,
 * an open-boundary convolution (Hockney: zero-padded (2n)^3 grid, FFTW).
 * The electrons minimize the same free energy as on the atoms,
 *     F(n) = -sum_i n_i Phi_i + 1/2 n.K n + T sum_i n_i [ln(n_i / dV) - 1],
 * Phi = K * q the ions' attraction: n_i ~ exp(-V_i / T), V = -Phi + K n.
 * Newton with Jacobi-preconditioned conjugate gradients (one convolution per
 * CG step), warm-started from the last solution; the grid is rebuilt (and
 * the last solution interpolated onto it) when R has changed by 10 % or the
 * centre has moved by a cell.
 *
 * Unlike the atom version the electrons do not move with the atoms between
 * solves: the electron potential K n stays on the grid (scaled by
 * N / N_solve) and the ions move through it.  The force on an ion is
 * q grad(K n) at its position, from the whole cloud: there is no cutoff, so
 * run with a Coulomb cutoff that covers the sample (the ion-ion forces must
 * be long-range too).  Outside the grid the cloud acts as a point charge at
 * its centroid.  Energy bookkeeping, escape test, deposits, collisions at
 * the local density and evaporation (per cell) are as in the ion-weight version (removed 2026-10-22).
 * ---------------------------------------------------------------- */
#define SFEL_PBG_IDX(i, j, k) ((((size_t)(i)) * ng + (j)) * ng + (k))

/* out = K * a on the grid (open boundaries) */
static void sfel_pbg_conv(t_sfelec *el, const double *a, double *out)
{
    const int    ng = el->pbg_n, m = 2 * ng;
    const size_t nh = (size_t)m * m * (m / 2 + 1);
    size_t       c;
    int          i, j, k;
    const double tc0 = el->tdet ? sfel_pb_wallms() : 0;
    double       tf;

#pragma omp parallel for
    for (i = 0; i < m; i++)
    {
        memset(el->pbg_buf + (size_t)i * m * m, 0, sizeof(float) * (size_t)m * m);
    }
#pragma omp parallel for private(j, k)
    for (i = 0; i < ng; i++)
    {
        for (j = 0; j < ng; j++)
        {
            for (k = 0; k < ng; k++)
            {
                el->pbg_buf[((size_t)i * m + j) * m + k] = (float)a[SFEL_PBG_IDX(i, j, k)];
            }
        }
    }
    tf = el->tdet ? sfel_pb_wallms() : 0;
    fftwf_execute(el->pbg_fwd);
    if (el->tdet) { el->td[SFEL_TD_FFT] += sfel_pb_wallms() - tf; }
#pragma omp parallel for
    for (c = 0; c < nh; c++)
    {
        el->pbg_hat[c][0] *= el->pbg_Kh[c];
        el->pbg_hat[c][1] *= el->pbg_Kh[c];
    }
    tf = el->tdet ? sfel_pb_wallms() : 0;
    fftwf_execute(el->pbg_bwd);
    if (el->tdet) { el->td[SFEL_TD_FFT] += sfel_pb_wallms() - tf; }
#pragma omp parallel for private(j, k)
    for (i = 0; i < ng; i++)
    {
        for (j = 0; j < ng; j++)
        {
            for (k = 0; k < ng; k++)
            {
                out[SFEL_PBG_IDX(i, j, k)] = el->pbg_buf[((size_t)i * m + j) * m + k];
            }
        }
    }
    el->pbg_nconv++;
    if (el->tdet) { el->td[SFEL_TD_CONV] += sfel_pb_wallms() - tc0; }
}

/* Smoothing length of the grid kernel k erf(r/s)/r: 2 dx, or the physical
 * floor s_e (sfmd-pb-smoothing) if larger. */
static double sfel_pbg_s(const t_sfelec *el)
{
    const double s = SFEL_PBG_SFAC * el->pbg_dx;

    return (el->pbg_se > s) ? el->pbg_se : s;
}

/* Kernel of the current spacing, Fourier transformed, normalization included. */
static void sfel_pbg_kernel(t_sfelec *el)
{
    const int    ng = el->pbg_n, m = 2 * ng;
    const double s  = sfel_pbg_s(el);
    const size_t nh = (size_t)m * m * (m / 2 + 1);
    size_t       c;
    int          i, j, k;

#pragma omp parallel for private(j, k)
    for (i = 0; i < m; i++)
    {
        double xi = ((i < m - i) ? i : m - i) * el->pbg_dx;

        for (j = 0; j < m; j++)
        {
            double yj = ((j < m - j) ? j : m - j) * el->pbg_dx;

            for (k = 0; k < m; k++)
            {
                double zk = ((k < m - k) ? k : m - k) * el->pbg_dx;

                el->pbg_buf[((size_t)i * m + j) * m + k] =
                    (float)(SFEL_COULOMB_EV_NM * sfel_pb_G(sqrt(xi * xi + yj * yj + zk * zk), s));
            }
        }
    }
    el->pbg_K0 = SFEL_COULOMB_EV_NM * sfel_pb_G(0, s);
    fftwf_execute(el->pbg_fwd);
#pragma omp parallel for
    for (c = 0; c < nh; c++)
    {
        el->pbg_Kh[c] = el->pbg_hat[c][0] / ((double)m * m * m);
    }
}

/* The same kernel on the FFT of 2 (ng + M) per side (ion potential with the
 * margin, sfel_pbg_conv_margin). */
static void sfel_pbg_kernel_margin(t_sfelec *el)
{
    const int    ng = el->pbg_n, m = 2 * (ng + SFEL_PBG_MARG);
    const double s  = sfel_pbg_s(el);
    const size_t nh = (size_t)m * m * (m / 2 + 1);
    size_t       c;
    int          i, j, k;

    for (i = 0; i < m; i++)
    {
        double xi = ((i < m - i) ? i : m - i) * el->pbg_dx;

        for (j = 0; j < m; j++)
        {
            double yj = ((j < m - j) ? j : m - j) * el->pbg_dx;

            for (k = 0; k < m; k++)
            {
                double zk = ((k < m - k) ? k : m - k) * el->pbg_dx;

                el->pbg_buf[((size_t)i * m + j) * m + k] =
                    (float)(SFEL_COULOMB_EV_NM * sfel_pb_G(sqrt(xi * xi + yj * yj + zk * zk), s));
            }
        }
    }
    fftwf_execute(el->pbg_fwdE);
    for (c = 0; c < nh; c++)
    {
        el->pbg_KhE[c] = el->pbg_hat[c][0] / ((double)m * m * m);
    }
}

/* Cloud-in-cell weights of point x: corner cell i0 and fractions f; FALSE if
 * outside the grid. */
static gmx_bool sfel_pbg_cic(const t_sfelec *el, const real *x, int i0[3], double f[3])
{
    int d;

    for (d = 0; d < 3; d++)
    {
        double u = (x[d] - el->pbg_lo[d]) / el->pbg_dx;

        i0[d] = (int)floor(u);
        f[d]  = u - i0[d];
        if (i0[d] < 0 || i0[d] >= el->pbg_n - 1)
        {
            return FALSE;
        }
    }
    return TRUE;
}

/* Ion attraction Phi (eV) on the grid for charges q at positions x; returns
 * the number of charged ions outside the box. */
static int sfel_pbg_ionpot_q(t_sfelec *el, const t_mdatoms *md, rvec *x, const real *qq, double *Phi);

static int sfel_pbg_ionpot(t_sfelec *el, const t_mdatoms *md, rvec *x, double *Phi)
{
    return sfel_pbg_ionpot_q(el, md, x, md->chargeA, Phi);
}

/* Potential (eV per e) of the charged ions that CIC cannot deposit (outside
 * the box or in its last cell), added to Phi, with the grid's smoothed kernel
 * G(r, s), s = 2 dx (bare 1/r beyond 4 s, where they agree).  stride 0: exact on every grid
 * point; otherwise each ion on the sub-lattice of stride S (S dx <= d / D, d
 * its distance from the box, S a power of two <= 16), interpolated. */
static void sfel_pbg_outpot(t_sfelec *el, const t_mdatoms *md, rvec *x, const real *qq, double *Phi, int lat,
                            const gmx_bool *skip)
{
    const int    ng = el->pbg_n;
    const double dx = el->pbg_dx, s = sfel_pbg_s(el);
    double      *c = NULL, *fr;
    int         *li, *node, *lev, L, S, nc, a, i, j, k, d, i0[3];
    double       f[3];
    int          nlev[5] = {0, 0, 0, 0, 0};
    double       tsum[5] = {0, 0, 0, 0, 0}, tint[5] = {0, 0, 0, 0, 0}, tl;

    snew(lev, md->nr);
    for (a = 0; a < md->nr; a++)
    {
        double dd = 0;

        lev[a] = -1;
        if (qq[a] == 0 || (skip != NULL && skip[a]) || sfel_pbg_cic(el, x[a], i0, f))
        {
            continue;
        }
        for (d = 0; d < 3; d++)
        {
            double u = el->pbg_lo[d] - x[a][d], v = x[a][d] - (el->pbg_lo[d] + (ng - 1) * dx);

            u   = (u > v) ? u : v;
            dd += (u > 0) ? u * u : 0;
        }
        dd     = sqrt(dd);
        lev[a] = 0;
        while (lat && lev[a] < 4 && (2 << lev[a]) * dx <= dd / SFEL_PBG_OUTLAT_D) { lev[a]++; }
        nlev[lev[a]]++;
    }
    snew(li, ng); snew(fr, ng); snew(node, ng + 1);
    for (L = 0; L <= 4; L++)
    {
        S  = 1 << L;
        nc = (ng - 1 + S - 1) / S + 1;
        for (a = 0; a < md->nr && lev[a] != L; a++) { }
        if (a == md->nr)
        {
            continue;
        }
        tl = sfel_pb_wallms();
        for (i = 0; i < nc; i++) { node[i] = (i * S < ng - 1) ? i * S : ng - 1; }
        snew(c, (size_t)nc * nc * nc);
        for (; a < md->nr; a++)
        {
            if (lev[a] != L)
            {
                continue;
            }
            for (i = 0; i < nc; i++)
            {
                double ex = el->pbg_lo[0] + node[i] * dx - x[a][0];

                for (j = 0; j < nc; j++)
                {
                    double ey = el->pbg_lo[1] + node[j] * dx - x[a][1];

                    for (k = 0; k < nc; k++)
                    {
                        double ez = el->pbg_lo[2] + node[k] * dx - x[a][2];

                        double r = sqrt(ex * ex + ey * ey + ez * ez);

                        c[((size_t)i * nc + j) * nc + k] += SFEL_COULOMB_EV_NM * qq[a] * (r < 4 * s ? sfel_pb_G(r, s) : 1.0 / r);
                    }
                }
            }
        }
        tsum[L] = sfel_pb_wallms() - tl;
        tl      = sfel_pb_wallms();
        if (S == 1)
        {
            for (i = 0; i < ng * ng * ng; i++) { Phi[i] += c[i]; }
        }
        else
        {
            for (i = 0; i < ng; i++)
            {
                li[i] = (i / S < nc - 1) ? i / S : nc - 2;
                fr[i] = (double)(i - node[li[i]]) / (node[li[i] + 1] - node[li[i]]);
            }
            for (i = 0; i < ng; i++)
            {
                for (j = 0; j < ng; j++)
                {
                    for (k = 0; k < ng; k++)
                    {
                        double v = 0;
                        int    u, w, z;

                        for (u = 0; u < 2; u++)
                        {
                            for (w = 0; w < 2; w++)
                            {
                                for (z = 0; z < 2; z++)
                                {
                                    v += (u ? fr[i] : 1 - fr[i]) * (w ? fr[j] : 1 - fr[j]) * (z ? fr[k] : 1 - fr[k]) *
                                         c[((size_t)(li[i] + u) * nc + li[j] + w) * nc + li[k] + z];
                                }
                            }
                        }
                        Phi[SFEL_PBG_IDX(i, j, k)] += v;
                    }
                }
            }
        }
        tint[L] = sfel_pb_wallms() - tl;
        sfree(c);
    }
    sfree(li); sfree(fr); sfree(node); sfree(lev);
    if (el->pbg_outlat == 2 && lat)
    {
        if (el->fop == NULL)
        {
            el->fop = sfel_open_output("pbg_outpot.txt", "w");
            fprintf(el->fop, "# per call: step ng dx[nm] ions at stride 1 2 4 8 16 | sum time [ms] at 1 2 4 8 16 | "
                    "interpolation/add time [ms] at 1 2 4 8 16  (GMX_SFMD_PBG_OUTLAT_CHECK)\n");
        }
        fprintf(el->fop, "%d %d %.4f  %d %d %d %d %d  %.1f %.1f %.1f %.1f %.1f  %.1f %.1f %.1f %.1f %.1f\n",
                (int)el->screen_step, ng, dx, nlev[0], nlev[1], nlev[2], nlev[3], nlev[4], tsum[0], tsum[1], tsum[2],
                tsum[3], tsum[4], tint[0], tint[1], tint[2], tint[3], tint[4]);
    }
}

/* Phi = potential on the box of the ions inside it and within SFEL_PBG_MARG = M
 * cells outside: deposited (CIC) on the grid extended by M cells per side and
 * convolved with the kernel on an FFT of 2 (ng + M) per side, which keeps the
 * open boundary for every pair (offsets up to ng - 1 + M).  dep[a] marks the
 * deposited ions; *nout counts the charged ions outside the box, as without
 * the margin.  Returns the number of charged ions beyond the margin. */
static int sfel_pbg_conv_margin(t_sfelec *el, const t_mdatoms *md, rvec *x, const real *qq, double *Phi,
                                gmx_bool *dep, int *nout)
{
    const int    ng = el->pbg_n, M = SFEL_PBG_MARG, m = 2 * (ng + M);
    const size_t nh = (size_t)m * m * (m / 2 + 1);
    const double tc0 = el->tdet ? sfel_pb_wallms() : 0;
    int          a, d, i, j, k, i0[3], nfar = 0;
    double       f[3], tf;
    size_t       c;

    for (i = 0; i < m; i++)
    {
        memset(el->pbg_buf + (size_t)i * m * m, 0, sizeof(float) * (size_t)m * m);
    }
    for (a = 0; a < md->nr; a++)
    {
        gmx_bool ok = TRUE;
        int      u, v, w;

        dep[a] = FALSE;
        if (qq[a] == 0)
        {
            continue;
        }
        if (!sfel_pbg_cic(el, x[a], i0, f))
        {
            (*nout)++;
        }
        for (d = 0; d < 3; d++)
        {
            double r = (x[a][d] - el->pbg_lo[d]) / el->pbg_dx;

            i0[d] = (int)floor(r);
            f[d]  = r - i0[d];
            ok    = ok && i0[d] >= -M && i0[d] <= ng - 2 + M;
        }
        if (!ok)
        {
            nfar++;
            continue;
        }
        dep[a] = TRUE;
        for (u = 0; u < 2; u++)
        {
            for (v = 0; v < 2; v++)
            {
                for (w = 0; w < 2; w++)
                {
                    el->pbg_buf[((size_t)(i0[0] + M + u) * m + i0[1] + M + v) * m + i0[2] + M + w] +=
                        (float)(qq[a] * (u ? f[0] : 1 - f[0]) * (v ? f[1] : 1 - f[1]) * (w ? f[2] : 1 - f[2]));
                }
            }
        }
    }
    tf = el->tdet ? sfel_pb_wallms() : 0;
    fftwf_execute(el->pbg_fwdE);
    if (el->tdet) { el->td[SFEL_TD_FFT] += sfel_pb_wallms() - tf; }
    for (c = 0; c < nh; c++)
    {
        el->pbg_hat[c][0] *= el->pbg_KhE[c];
        el->pbg_hat[c][1] *= el->pbg_KhE[c];
    }
    tf = el->tdet ? sfel_pb_wallms() : 0;
    fftwf_execute(el->pbg_bwdE);
    if (el->tdet) { el->td[SFEL_TD_FFT] += sfel_pb_wallms() - tf; }
    for (i = 0; i < ng; i++)
    {
        for (j = 0; j < ng; j++)
        {
            for (k = 0; k < ng; k++)
            {
                Phi[SFEL_PBG_IDX(i, j, k)] = el->pbg_buf[((size_t)(i + M) * m + j + M) * m + k + M];
            }
        }
    }
    el->pbg_nconv++;
    if (el->tdet) { el->td[SFEL_TD_CONV] += sfel_pb_wallms() - tc0; }
    return nfar;
}

static int sfel_pbg_ionpot_q(t_sfelec *el, const t_mdatoms *md, rvec *x, const real *qq, double *Phi)
{
    const int    ng = el->pbg_n;
    const size_t G  = (size_t)ng * ng * ng;
    int          nout = 0, nfar;
    gmx_bool    *dep = NULL;

    snew(dep, md->nr);
    nfar = sfel_pbg_conv_margin(el, md, x, qq, Phi, dep, &nout);
    if (nfar > 0)
    {
        const double to0 = el->tdet ? sfel_pb_wallms() : 0;

        if (el->pbg_outlat == 2)
        {
            /* check: sub-lattices against the exact sum, both on zero */
            double *ex, *ap, emax = 0, e2 = 0, pmax = 0;
            size_t  c;

            snew(ex, G); snew(ap, G);
            sfel_pbg_outpot(el, md, x, qq, ex, 0, dep);
            sfel_pbg_outpot(el, md, x, qq, ap, 1, dep);
            for (c = 0; c < G; c++)
            {
                double e = fabs(ap[c] - ex[c]);

                emax  = (e > emax) ? e : emax;
                e2   += e * e;
                pmax  = (fabs(ex[c]) > pmax) ? fabs(ex[c]) : pmax;
                Phi[c] += ap[c];
            }
            if (el->fol == NULL)
            {
                el->fol = sfel_open_output("pbg_outlat.txt", "w");
                fprintf(el->fol, "# per ion-potential call: ions_outside box[nm] dx[nm] max|Phi_out|[eV] "
                        "max_err[eV] rms_err[eV]  (stride dx <= distance / %g)\n", SFEL_PBG_OUTLAT_D);
            }
            fprintf(el->fol, "%d %.4f %.5f %.4e %.3e %.3e\n", nout, 0.5 * (ng - 1) * el->pbg_dx, el->pbg_dx, pmax, emax,
                    sqrt(e2 / G));
            fflush(el->fol);
            sfree(ex); sfree(ap);
        }
        else
        {
            sfel_pbg_outpot(el, md, x, qq, Phi, el->pbg_outlat, dep);
        }
        if (el->tdet) { el->td[SFEL_TD_OUTPOT] += sfel_pb_wallms() - to0; }
    }
    sfree(dep);
    return nout;
}

/* F(n) and V = -Phi + K n */
static double sfel_pbg_free(t_sfelec *el, const double *Phi, double T, const double *n, double *V)
{
    const int    ng = el->pbg_n;
    const size_t G  = (size_t)ng * ng * ng;
    const double dV = el->pbg_dx * el->pbg_dx * el->pbg_dx;
    double       F  = 0;
    size_t       c;

    sfel_pbg_conv(el, n, el->pbg_Kn);
    if (el->pbg_UL != NULL)
    {
        /* sfmd-laser-polarization: the laser's potential energy in the cloud */
        const double *UL = el->pbg_UL;

#pragma omp parallel for reduction(+:F)
        for (c = 0; c < G; c++)
        {
            V[c] = -Phi[c] + UL[c] + el->pbg_Kn[c];
            F   += n[c] * (-Phi[c] + UL[c] + 0.5 * el->pbg_Kn[c] + T * (log(n[c] / dV) - 1.0));
        }
        return F;
    }
#pragma omp parallel for reduction(+:F)
    for (c = 0; c < G; c++)
    {
        V[c] = -Phi[c] + el->pbg_Kn[c];
        F   += n[c] * (-Phi[c] + 0.5 * el->pbg_Kn[c] + T * (log(n[c] / dV) - 1.0));
    }
    return F;
}

/* ---------------------------------------------------------------- *
 * sfmd-laser-polarization: a linearly polarized field E(t) (V/nm, signed,
 * along the axis e) in the cloud.  The field pushes an electron along
 * s = -sign(E) e, its potential energy is U_L = E (x - c).e = -|E| zeta,
 * zeta = (x - c).s, c the heavy-atom centre.  Along s, outside the sample,
 *     U(r) = -kQ/r - kP/r^2 - |E| r,        P = p.s,
 * Q the net charge and p the dipole of ions and cloud about c (the cloud
 * pulled along s gives P < 0, which raises the saddle).  The saddle is the
 * largest root of |E| r^3 - kQ r - 2kP = 0, the escape level U_s = U(r_s)
 * if that lies outside the sample (r_s >= R_e, the gyration ellipsoid's
 * radius along the axis).  Otherwise -- the root inside the sample, or none
 * because the dipole term wins (a strongly polarized cloud of small Q) --
 * U falls monotonically beyond R_e and the level is U(R_e), the highest
 * point on the way out (until 2026-11-13 the root itself, or for no root
 * r = sqrt(kQ/3|E|), where the dipole term made the level huge: a neutral
 * cloud was then held at 1e17 W/cm^2 and absorbed 130 MeV).
 * Beyond the saddle plane (zeta > r_s) the field term is held at its value
 * there: those cells are outside the trap and spill (V >= U_s), and an
 * unbounded linear term would pull the whole solve to the box face.
 * ---------------------------------------------------------------- */
static void sfel_pol_saddle(const t_sfelec *el, double Q, double E, double *rho, double *Us, gmx_bool *nosad)
{
    const double K  = SFEL_COULOMB_EV_NM, aE = fabs(E);
    const int    ax = el->pol_axis;
    double       P, kq, rm, r, Re;
    int          it;

    *nosad = FALSE;
    if (ax < 0 || aE <= 0 || Q <= 0)
    {
        *rho = 0;       /* no field term */
        *Us  = 0;
        return;
    }
    P  = el->pol_p[ax] * ((E > 0) ? -1.0 : 1.0);
    kq = K * Q;
    rm = sqrt(kq / (3.0 * aE));
    {
        /* R_e: the sample's gyration ellipsoid radius along the axis */
        double w = 0;
        int    i;

        for (i = 0; i < 3; i++)
        {
            if (el->ell_a[i] > 0) { w += el->ell_axis[i][ax] * el->ell_axis[i][ax] / (el->ell_a[i] * el->ell_a[i]); }
        }
        Re = (w > 0) ? 1.0 / sqrt(w) : el->radius_now;
    }
    if (P < 0 && aE * rm * rm * rm - kq * rm - 2.0 * K * P > 0)
    {
        /* no maximum along s: the dipole term wins; U falls all the way out */
        r      = Re;
        *nosad = TRUE;
    }
    else
    {
        /* Newton from the right: f is convex for r > 0, so it falls monotonically to the largest root */
        r = 2.0 * sqrt(kq / aE) + 2.0 * cbrt(fabs(2.0 * K * P) / aE) + 1e-3;
        for (it = 0; it < 100; it++)
        {
            const double f = aE * r * r * r - kq * r - 2.0 * K * P, fp = 3.0 * aE * r * r - kq, dr = f / fp;

            r -= dr;
            if (fabs(dr) < 1e-12 * r)
            {
                break;
            }
        }
        if (r < Re)
        {
            /* the saddle inside the sample: the highest point outside it is its surface */
            r      = Re;
            *nosad = TRUE;
        }
    }
    *rho = r;
    *Us  = -kq / r - K * P / (r * r) - aE * r;
}

/* The laser's potential energy (eV) at xr = x - c for field E and saddle distance rho */
static double sfel_pol_UL(const t_sfelec *el, const double xr[3], double E, double rho)
{
    double zeta;

    if (rho <= 0)
    {
        return 0;
    }
    zeta = ((E > 0) ? -1.0 : 1.0) * xr[el->pol_axis];
    return -fabs(E) * ((zeta < rho) ? zeta : rho);
}

/* sum over cells of n x_axis (nm), absolute coordinates */
static double sfel_pol_X(const t_sfelec *el, const double *n)
{
    const int    ng = el->pbg_n, ax = el->pol_axis;
    double       X  = 0;
    int          i, j, k;

    for (i = 0; i < ng; i++)
    {
        for (j = 0; j < ng; j++)
        {
            for (k = 0; k < ng; k++)
            {
                const int ix[3] = {i, j, k};

                X += n[SFEL_PBG_IDX(i, j, k)] * (el->pbg_lo[ax] + ix[ax] * el->pbg_dx);
            }
        }
    }
    return X;
}

/* Electrostatic energy of the electrons n with the ions (Phi) and themselves. */
static double sfel_pbg_energy(t_sfelec *el, const double *Phi, const double *n, double sc)
{
    const int    ng = el->pbg_n;
    const size_t G  = (size_t)ng * ng * ng;
    double       U  = 0;
    size_t       c;

    sfel_pbg_conv(el, n, el->pbg_Kn);
#pragma omp parallel for reduction(+:U)
    for (c = 0; c < G; c++)
    {
        U += sc * n[c] * (-Phi[c] + 0.5 * sc * el->pbg_Kn[c]);
    }
    return U;
}

/* The constrained Newton step in one solve: H d + mu 1 = -g, sum(d) = 0,
 * H = K + T diag(1/n), by projected Jacobi-preconditioned CG.  The
 * preconditioned residual z = M^-1 (r - lambda 1) with lambda such that
 * sum(z) = 0, so all directions (and d) keep the electron count.  Stops when
 * the projected residual r - lambda 1 is down by 1e-4 (squared 1e-8, as the
 * two-solve CG). */
static int sfel_pbg_pcg(t_sfelec *el, double T, const double *n, const double *g, double *dsol)
{
    const int    ng = el->pbg_n;
    const size_t G  = (size_t)ng * ng * ng;
    double      *p = el->pbg_cg, *q = p + G, *r = q + G, *z = r + G;
    double       sm = 0, rz = 0, rr0 = 0, lam, sr;
    size_t       c;
    int          it;

#pragma omp parallel for reduction(+:sm)
    for (c = 0; c < G; c++) { sm += 1.0 / (el->pbg_K0 + T / n[c]); }
#pragma omp parallel for
    for (c = 0; c < G; c++) { dsol[c] = 0; r[c] = -g[c]; }

    /* z = P M^-1 r; returns r.z, and the squared projected residual in *rrp */
#define SFEL_PCG_PRECOND(rrp)                                                      \
    {                                                                              \
        double sz = 0, rrl = 0, srl = 0;                                           \
        _Pragma("omp parallel for reduction(+:sz)")                                \
        for (c = 0; c < G; c++) { sz += r[c] / (el->pbg_K0 + T / n[c]); }          \
        lam = sz / sm;                                                             \
        _Pragma("omp parallel for reduction(+:srl, rrl)")                          \
        for (c = 0; c < G; c++)                                                    \
        {                                                                          \
            const double rp = r[c] - lam;                                          \
                                                                                   \
            z[c] = rp / (el->pbg_K0 + T / n[c]);                                   \
            srl += r[c] * z[c];                                                    \
            rrl += rp * rp;                                                        \
        }                                                                          \
        sr     = srl;                                                              \
        *(rrp) = rrl;                                                              \
    }

    SFEL_PCG_PRECOND(&rr0);
    rz = sr;
#pragma omp parallel for
    for (c = 0; c < G; c++) { p[c] = z[c]; }
    for (it = 0; it < 500 && rr0 > 0; it++)
    {
        double pq = 0, a, rzn, rr;

        sfel_pbg_conv(el, p, q);
#pragma omp parallel for reduction(+:pq)
        for (c = 0; c < G; c++)
        {
            q[c] += T * p[c] / n[c];
            pq   += p[c] * q[c];
        }
        a = rz / pq;
#pragma omp parallel for
        for (c = 0; c < G; c++)
        {
            dsol[c] += a * p[c];
            r[c]    -= a * q[c];
        }
        SFEL_PCG_PRECOND(&rr);
        rzn = sr;
        if (rr < 1e-8 * rr0)
        {
            it++;
            break;
        }
#pragma omp parallel for
        for (c = 0; c < G; c++) { p[c] = z[c] + rzn / rz * p[c]; }
        rz = rzn;
    }
#undef SFEL_PCG_PRECOND
    return it;
}

/* The grid limit of this sample (master; once, before the arrays are made):
 * the cells the target spacing needs at the largest box, at R0.  The arrays
 * are made just before the first rebuild, which takes R0 = the radius now, so
 * R0 is fixed here; the spacing grows with R / R0 and the box is at most
 * pbg_box R, so the rebuilds never need more cells than this. */
static void sfel_pbg_setcap(t_sfelec *el)
{
    int ng;

    if (el->pbg_capset)
    {
        return;
    }
    if (el->pbg_R0 <= 0) { el->pbg_R0 = el->radius_now; }
    ng = (int)ceil(2.0 * el->pbg_box * el->pbg_R0 / el->pbg_dxt) + 1;
    ng = 8 * ((ng + 7) / 8);
    ng = (ng < el->pbg_nmin) ? el->pbg_nmin : ng;
    if (ng > el->pbg_nglimit)
    {
        fprintf(stderr, "\nsfelectrons: WARNING the PB grid would need %d cells per side at %g R; limited to %d "
                "(sfmd-pb-grid-limit), so the spacing is coarser than s_e / %g late in the run\n",
                ng, el->pbg_box, el->pbg_nglimit, el->pbg_se / el->pbg_dxt);
        ng = el->pbg_nglimit;
    }
    el->pbg_nmax   = ng;
    el->pbg_ncap   = ng;
    el->pbg_n      = ng;
    el->pbg_capset = 1;
    fprintf(stderr, "\nsfelectrons: PB grid up to %d cells per side (R0 %.4g nm, spacing %.3g nm at R0, box up to %g R)\n",
            ng, el->pbg_R0, el->pbg_dxt, el->pbg_box);
}

/* FFT plans for the grid size and the kernels for its spacing (master) */
static void sfel_pbg_plans(t_sfelec *el)
{
    if (el->pbg_nplan != el->pbg_n)
    {
        const int m = 2 * el->pbg_n;

        if (el->pbg_nplan > 0)
        {
            fftwf_destroy_plan(el->pbg_fwd);
            fftwf_destroy_plan(el->pbg_bwd);
            fftwf_destroy_plan(el->pbg_fwdE);
            fftwf_destroy_plan(el->pbg_bwdE);
        }
        el->pbg_fwd   = fftwf_plan_dft_r2c_3d(m, m, m, el->pbg_buf, el->pbg_hat, FFTW_ESTIMATE);
        el->pbg_bwd   = fftwf_plan_dft_c2r_3d(m, m, m, el->pbg_hat, el->pbg_buf, FFTW_ESTIMATE);
        {
            const int mE = 2 * (el->pbg_n + SFEL_PBG_MARG);

            el->pbg_fwdE = fftwf_plan_dft_r2c_3d(mE, mE, mE, el->pbg_buf, el->pbg_hat, FFTW_ESTIMATE);
            el->pbg_bwdE = fftwf_plan_dft_c2r_3d(mE, mE, mE, el->pbg_hat, el->pbg_buf, FFTW_ESTIMATE);
        }
        el->pbg_nplan = el->pbg_n;
    }
    sfel_pbg_kernel(el);
    sfel_pbg_kernel_margin(el);
}

/* The grid arrays and FFT plans (master; once). */
static void sfel_pbg_alloc(t_sfelec *el, int n)
{
    int    ng, m;
    size_t G;

    if (el->pbg_ne != NULL)
    {
        return;
    }
    sfel_pbg_setcap(el);
    ng = el->pbg_ncap;
    m  = 2 * ng;
    G  = (size_t)ng * ng * ng;
    snew(el->pbg_ne, G); snew(el->pbg_Phi, G); snew(el->pbg_V, G); snew(el->pbg_g, G); snew(el->pbg_d, G);
    snew(el->pbg_t, G); snew(el->pbg_Kn, G);
    snew(el->pbg_nt, G); snew(el->pbg_Vt, G); snew(el->pbg_cg, 6 * G);
    {
        /* the FFT buffers also hold the ion potential's FFT with the margin */
        const int mE = 2 * (ng + SFEL_PBG_MARG);

        el->pbg_buf = fftwf_malloc(sizeof(float) * (size_t)mE * mE * mE);
        el->pbg_hat = fftwf_malloc(sizeof(fftwf_complex) * (size_t)mE * mE * (mE / 2 + 1));
        snew(el->pbg_KhE, (size_t)mE * mE * (mE / 2 + 1));
    }
    snew(el->pbg_Kh, (size_t)m * m * (m / 2 + 1));
#ifdef SFEL_FFTW_THREADS
    /* the other ranks wait at the broadcast while the master solves, so
     * the FFTs can use their cores (GMX_SFMD_PBG_FFT_THREADS, default the
     * number of ranks) */
    fftwf_init_threads();
    fftwf_plan_with_nthreads(el->pbg_nthreads > 0 ? el->pbg_nthreads : 1);
    fprintf(stderr, "\nsfelectrons: PB grid FFTs on %d threads\n", el->pbg_nthreads > 0 ? el->pbg_nthreads : 1);
#endif
#ifdef _OPENMP
    /* the grid loops (padding, CG, Newton) on as many OpenMP threads, in the
     * master's parallel regions */
    omp_set_num_threads(el->pbg_nthreads > 0 ? el->pbg_nthreads : 1);
#endif
    el->pbg_nplan = 0;            /* plans made by the first rebuild, for its size */
    if (el->pb_ne == NULL)    { snew(el->pb_ne, n); }
    if (el->pbg_rst_grid)
    {
        /* a continuation: the grid of the checkpoint (size, spacing, origin,
         * restored by sfel_state_apply), so that the first solve's energy is
         * on the same smoothing as before */
        sfel_pbg_plans(el);
        el->pbg_have     = TRUE;
        el->pbg_rst_grid = FALSE;
    }
}

/* (Re)build the grid about the current centre and radius; the last solution
 * is carried over by trilinear interpolation. */
static void sfel_pbg_grid(t_sfelec *el)
{
    const int    ng_old = el->pbg_n;
    const double half   = el->pbg_hbox * el->radius_now;
    double       lo_old[3], dx_old = el->pbg_dx, dV, dV_old = dx_old * dx_old * dx_old;
    int          ng, i, j, k, d;
    size_t       G;

    {
        /* adaptive size: spacing near the target, which grows with the
         * sample (R / R0), a multiple of 8 */
        double dxt;

        if (el->pbg_R0 <= 0) { el->pbg_R0 = el->radius_now; }
        dxt = el->pbg_dxt * ((el->radius_now > el->pbg_R0) ? el->radius_now / el->pbg_R0 : 1.0);
        ng  = (int)ceil(2.0 * half / dxt) + 1;
        ng = 8 * ((ng + 7) / 8);
        ng = (ng < el->pbg_nmin) ? el->pbg_nmin : ((ng > el->pbg_nmax) ? el->pbg_nmax : ng);
    }
    G = (size_t)ng * ng * ng;
    for (d = 0; d < 3; d++) { lo_old[d] = el->pbg_lo[d]; }
    el->pbg_dx = 2.0 * half / (ng - 1);
    dV         = el->pbg_dx * el->pbg_dx * el->pbg_dx;
    for (d = 0; d < 3; d++) { el->pbg_lo[d] = el->com[d] - half; }
    el->pbg_R = el->radius_now;
    for (d = 0; d < 3; d++) { el->pbg_c[d] = el->com[d]; }
    if (el->pbg_have)
    {
        /* density of the old grid at the new cell centres */
        for (i = 0; i < ng; i++)
        {
            for (j = 0; j < ng; j++)
            {
                for (k = 0; k < ng; k++)
                {
                    double u[3] = {(el->pbg_lo[0] + i * el->pbg_dx - lo_old[0]) / dx_old,
                                   (el->pbg_lo[1] + j * el->pbg_dx - lo_old[1]) / dx_old,
                                   (el->pbg_lo[2] + k * el->pbg_dx - lo_old[2]) / dx_old};
                    int    i0[3], ok = 1, a, b, c;
                    double f[3], s = 0;

                    for (d = 0; d < 3; d++)
                    {
                        i0[d] = (int)floor(u[d]);
                        f[d]  = u[d] - i0[d];
                        if (i0[d] < 0 || i0[d] >= ng_old - 1) { ok = 0; }
                    }
                    if (ok)
                    {
                        for (a = 0; a < 2; a++)
                        {
                            for (b = 0; b < 2; b++)
                            {
                                for (c = 0; c < 2; c++)
                                {
                                    /* the old density in the old layout */
                                    s += (a ? f[0] : 1 - f[0]) * (b ? f[1] : 1 - f[1]) * (c ? f[2] : 1 - f[2]) *
                                         el->pbg_ne[(((size_t)(i0[0] + a)) * ng_old + (i0[1] + b)) * ng_old + (i0[2] + c)];
                                }
                            }
                        }
                    }
                    el->pbg_t[SFEL_PBG_IDX(i, j, k)] = s / dV_old * dV;
                }
            }
        }
        memcpy(el->pbg_ne, el->pbg_t, sizeof(double) * G);
    }
    el->pbg_n = ng;
    sfel_pbg_plans(el);
    el->pbg_have = TRUE;
    el->pbg_nregrid++;
}

/* Adaptive box (sfmd-pb-box-cut): the half-width / R that leaves a
 * fraction pbg_cut of the electrons outside.  Inside the current box the
 * density is the grid's; beyond it, up to the upper limit pbg_box, the
 * Boltzmann density in the field of the net charge Q at the centre,
 * A exp(k Q / (T r)), A fitted to the outermost layer of cells.  *tail: the
 * electrons so estimated outside the current box; *hsafe: the half-width / R
 * that leaves only SFEL_PBG_SCUT outside (the size a shrink may go to). */
static double sfel_pbg_boxwant(t_sfelec *el, double T, double *tail, double *hsafe)
{
    const int    ng = el->pbg_n;
    const double R = el->radius_now, w = 0.5 * el->pbg_dx, dV = el->pbg_dx * el->pbg_dx * el->pbg_dx;
    const double hmax = el->pbg_box * R, half = 0.5 * (ng - 1) * el->pbg_dx;
    const double kQT = SFEL_COULOMB_EV_NM * el->escaped_photoelectrons / T;
    const int    nb = (int)(hmax / w) + ng + 4;
    double      *h, tot = 0, sn = 0, sw = 0, A, cum, want;
    int          i, j, k, b, bs;

    snew(h, nb);
    for (i = 0; i < ng; i++)
    {
        for (j = 0; j < ng; j++)
        {
            for (k = 0; k < ng; k++)
            {
                const double v = el->pbg_ne[SFEL_PBG_IDX(i, j, k)];
                const int    ix[3] = {i, j, k};
                double       dm = 0, r2 = 0;
                int          e = ng, d;

                for (d = 0; d < 3; d++)
                {
                    double u = el->pbg_lo[d] + ix[d] * el->pbg_dx - el->com[d];

                    r2 += u * u;
                    u   = fabs(u);
                    dm  = (u > dm) ? u : dm;
                    e   = (ix[d] < e) ? ix[d] : e;
                    e   = (ng - 1 - ix[d] < e) ? ng - 1 - ix[d] : e;
                }
                b = (int)(dm / w);
                h[b < nb ? b : nb - 1] += v;
                tot += v;
                if (e == 0)
                {
                    /* exponent relative to the box half-width: <= 0 here and outside */
                    sn += v;
                    sw += dV * exp(kQT * (1.0 / sqrt(r2) - 1.0 / half));
                }
            }
        }
    }
    *tail  = 0;
    *hsafe = el->pbg_hbox;
    if (tot <= 0)
    {
        sfree(h);
        return el->pbg_hbox;
    }
    A = (sw > 0) ? sn / sw : 0;
    if (A > 0 && hmax > 1.001 * half)
    {
        const double hs = 2.0 * el->pbg_dx, vs = hs * hs * hs;
        const int    M  = (int)ceil(hmax / hs);

        for (i = -M; i <= M; i++)
        {
            for (j = -M; j <= M; j++)
            {
                for (k = -M; k <= M; k++)
                {
                    const double u[3] = {i * hs, j * hs, k * hs};
                    double       dm = 0, v;
                    int          d;

                    for (d = 0; d < 3; d++) { dm = (fabs(u[d]) > dm) ? fabs(u[d]) : dm; }
                    if (dm <= half || dm > hmax)
                    {
                        continue;
                    }
                    v = A * vs * exp(kQT * (1.0 / sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) - 1.0 / half));
                    b = (int)(dm / w);
                    h[b < nb ? b : nb - 1] += v;
                    *tail += v;
                }
            }
        }
    }
    /* the cuts count the electrons outside against all of them (counting
     * them against the net charge Q was tested 2026-10-25 and not kept) */
    {
        const double ref = tot + *tail;

        for (i = 0, b = -1, bs = -1, cum = 0; i < nb && bs < 0; i++)
        {
            cum += h[i];
            if (b < 0 && (tot + *tail - cum) / ref <= el->pbg_cut) { b = i; }
            if ((tot + *tail - cum) / ref <= SFEL_PBG_SCUT) { bs = i; }
        }
    }
    b      = (b < 0) ? nb - 1 : b;
    want   = ((b + 1) * w + el->pbg_dx) / R;
    *hsafe = (bs < 0) ? el->pbg_box : ((bs + 1) * w + el->pbg_dx) / R;
    *hsafe = (*hsafe > el->pbg_box) ? el->pbg_box : ((*hsafe < SFEL_PBG_BOXMIN) ? SFEL_PBG_BOXMIN : *hsafe);
    *tail /= tot;
    sfree(h);
    want = (want > el->pbg_box) ? el->pbg_box : want;
    return (want < SFEL_PBG_BOXMIN) ? SFEL_PBG_BOXMIN : want;
}

/* Master: new density for N trapped electrons at T_e, and the energy
 * bookkeeping of the interval since the last solve. */
static void sfel_pbg_solve(t_sfelec *el, const t_inputrec *ir, const t_mdatoms *md, rvec *x, double t)
{
    const int    n  = md->nr;
    int          ng = el->pbg_n;
    size_t       G  = (size_t)ng * ng * ng;
    const double N  = el->num_free_electrons;
    /* 1 eV, as gromacs-mc (GMX_SFMD_PB_TMIN, diagnostic), or sfmd-te-min */
    const double Tmin = (el->te_min > 0) ? el->te_min : el->pb_Tmin;
    double       T  = (sfel_plasma_Te(el) > Tmin) ? sfel_plasma_Te(el) : Tmin;
    int          nT = 1;
    double       t0 = sfel_pb_wallms(), Unew, dU = 0, F, resid = 0, sum, dV, Dmean = 0, Dmax = -1e30;
    double       ecore = 0, qcore = 0;
    int          it, cgit = 0, nout, i, j, k, a, d;
    size_t       c;

    sfel_pbg_alloc(el, n);
    el->pbg_nconv = 0;

    /* total energy: the electrostatic steps from the last solve, on the grid
     * the last density lives on (energy split, GMX_SFMD_PB_EDIAG), and U3 =
     * the energy now (atoms moved, charges changed, N scaled) before a rebuild */
    const gmx_bool etot_on = el->pb_etot_init && el->pbg_have && el->pb_nsolve > 0 && !el->pbg_cold;
    double         U1 = 0, U2 = 0, U3 = 0, U4 = 0;
    gmx_bool       rebuilt = FALSE;

    if (etot_on)
    {
        if (el->pb_ediag && el->pbd_q != NULL)
        {
            sfel_pbg_ionpot_q(el, md, x, el->pbd_q, el->pbg_Phi);
            U1 = sfel_pbg_energy(el, el->pbg_Phi, el->pbg_ne, 1.0);
            sfel_pbg_ionpot(el, md, x, el->pbg_Phi);
            U2 = sfel_pbg_energy(el, el->pbg_Phi, el->pbg_ne, 1.0);
        }
        else
        {
            sfel_pbg_ionpot(el, md, x, el->pbg_Phi);
        }
        U3 = sfel_pbg_energy(el, el->pbg_Phi, el->pbg_ne, N / el->pb_N);
        U4 = U3;
    }

    /* adaptive box: the half-width / R the last density asks for */
    double   hwant = el->pbg_hbox, hsafe = el->pbg_hbox, btail = 0;
    gmx_bool bgrow, bshrink;

    if (el->pbg_cut > 0 && el->pbg_have && el->pb_nsolve > 0 && N > 0 && !el->pbg_cold)
    {
        hwant = sfel_pbg_boxwant(el, T, &btail, &hsafe);
    }
    /* grow at once, 5 % beyond what is wanted (no lag behind a spreading
     * cloud).  A shrink folds the outer electrons into the box and the
     * rebuild's energy shift is booked as numerical: a box that grew and
     * shrank on every solve (600 eV 1e7, up to 36 times in a run) drained
     * ~0.6 MeV from the cloud.  So shrink only to where at most SFEL_PBG_SCUT
     * (<< pbg_cut) is outside, and by at least 10 %: the box then never
     * flips back at the next solve */
    bgrow   = (hwant > el->pbg_hbox);
    bshrink = (!bgrow && SFEL_PBG_BOXMARG * hsafe < 0.9 * el->pbg_hbox);

    /* rebuild the grid first (the last density interpolated onto it, same
     * count), so that the energy before and after are both on this grid: a
     * rebuild changes the smoothing, which is not the electrons' energy */
    if (!el->pbg_have || (!el->pbg_cold && (el->radius_now > 1.1 * el->pbg_R || el->radius_now < 0.9 * el->pbg_R ||
        sqrt(sqr(el->com[0] - el->pbg_c[0]) + sqr(el->com[1] - el->pbg_c[1]) + sqr(el->com[2] - el->pbg_c[2])) >
        el->pbg_dx || bgrow || bshrink)))
    {
        gmx_bool had = el->pbg_have;

        hwant = bshrink ? SFEL_PBG_BOXMARG * hsafe : (!bgrow ? el->pbg_hbox : SFEL_PBG_BOXMARG * hwant);
        hwant = (hwant < el->pbg_box) ? hwant : el->pbg_box;
        hwant = (hwant > SFEL_PBG_BOXMIN) ? hwant : SFEL_PBG_BOXMIN;
        el->pbg_hbox = hwant;
        sfel_pbg_grid(el);
        ng = el->pbg_n;
        G  = (size_t)ng * ng * ng;
        if (had && el->pb_nsolve > 0)
        {
            sum = 0;
            for (c = 0; c < G; c++) { sum += el->pbg_ne[c]; }
            for (c = 0; c < G; c++) { el->pbg_ne[c] = (sum > 0) ? el->pbg_ne[c] * el->pb_N / sum : el->pb_N / G; }
        }
        rebuilt = had;
    }
    if (el->pbg_cut > 0)
    {
        if (el->fbox == NULL)
        {
            el->fbox = sfel_open_output("pbg_box.txt", "w");
            fprintf(el->fbox, "# t[ps] R[nm] box[R] dx[nm] outside_est(before rebuild) rebuilt N  (sfmd-pb-box-cut %g, "
                    "box %g..%g R, shrink cut %g, margin %g)\n",
                    el->pbg_cut, SFEL_PBG_BOXMIN, el->pbg_box, SFEL_PBG_SCUT, SFEL_PBG_BOXMARG);
        }
        fprintf(el->fbox, "%.5f %.4f %.3f %.5f %.3e %d %d\n", t, el->radius_now, el->pbg_hbox, el->pbg_dx,
                btail / (1.0 + btail), (int)rebuilt, ng);
        fflush(el->fbox);
    }
    if (etot_on && rebuilt)
    {
        /* the rebuild's change of the energy (smoothing, interpolation) is
         * numerical: E_tot moves with it, E_kin does not */
        sfel_pbg_ionpot(el, md, x, el->pbg_Phi);
        U4           = sfel_pbg_energy(el, el->pbg_Phi, el->pbg_ne, N / el->pb_N);
        el->pb_Etot += U4 - U3;
    }
    dV = el->pbg_dx * el->pbg_dx * el->pbg_dx;
    nout = sfel_pbg_ionpot(el, md, x, el->pbg_Phi);

    /* sfmd-laser-polarization: this solve's field in the cloud.  Its work on
     * the cloud since the last solve, int E dX (X = sum n x_axis), is booked
     * with the mean of the two fields: for a cloud that follows the field
     * linearly that sum telescopes to zero over a cycle, while the field of
     * the new solve alone would dissipate 1/2 alpha dE^2 into E_kin at every
     * solve (several keV per solve for lysozyme at 1e16 W/cm^2). */
    double   pol_Emid = 0, pol_Xold = 0;
    gmx_bool pol_on   = (el->pol_axis >= 0);

    if (pol_on)
    {
        const double E = el->E_sig;
        double       rho, Us, xr[3];
        gmx_bool     nosad;

        if (el->pbg_UL == NULL) { snew(el->pbg_UL, (size_t)el->pbg_ncap * el->pbg_ncap * el->pbg_ncap); }
        sfel_pol_saddle(el, el->escaped_photoelectrons, E, &rho, &Us, &nosad);
        el->pol_nosaddle += nosad;
        for (i = 0; i < ng; i++)
        {
            for (j = 0; j < ng; j++)
            {
                for (k = 0; k < ng; k++)
                {
                    xr[0] = el->pbg_lo[0] + i * el->pbg_dx - el->com[0];
                    xr[1] = el->pbg_lo[1] + j * el->pbg_dx - el->com[1];
                    xr[2] = el->pbg_lo[2] + k * el->pbg_dx - el->com[2];
                    el->pbg_UL[SFEL_PBG_IDX(i, j, k)] = sfel_pol_UL(el, xr, E, rho);
                }
            }
        }
        pol_Emid  = 0.5 * (el->pol_E + E);
        pol_Xold  = el->pol_X;
        el->pol_E = E;
    }

    /* starting density: the last one (scaled), or on the ion charge (also
     * the first solve after a restart, which has no density) */
    if (el->pb_nsolve == 0 || el->pbg_cold)
    {
        el->pbg_cold = FALSE;
        memset(el->pbg_ne, 0, sizeof(double) * G);
        for (a = 0; a < n; a++)
        {
            int    i0[3], u, v, w;
            double f[3];

            if (md->chargeA[a] <= 0 || !sfel_pbg_cic(el, x[a], i0, f)) { continue; }
            for (u = 0; u < 2; u++)
            {
                for (v = 0; v < 2; v++)
                {
                    for (w = 0; w < 2; w++)
                    {
                        el->pbg_ne[SFEL_PBG_IDX(i0[0] + u, i0[1] + v, i0[2] + w)] +=
                            md->chargeA[a] * (u ? f[0] : 1 - f[0]) * (v ? f[1] : 1 - f[1]) * (w ? f[2] : 1 - f[2]);
                    }
                }
            }
        }
    }
    sum = 0;
    for (c = 0; c < G; c++) { sum += el->pbg_ne[c]; }
    for (c = 0; c < G; c++)
    {
        el->pbg_ne[c] = el->pbg_ne[c] / sum * N * (1.0 - 1e-3) + 1e-3 * N / G;
    }

    /* timing detail: would a guess extrapolated from the last two solves start
     * closer?  (its residual, one extra convolution; behaviour unchanged) */
    double resid_ex = -1, tn0 = el->tdet ? sfel_pb_wallms() : 0;

    if (el->tdet && el->pbg_nprev_ok && !rebuilt && el->pb_nsolve > 1)
    {
        const double tx0 = sfel_pb_wallms();
        double       st = 0, gb = 0, r = 0;

        for (c = 0; c < G; c++)
        {
            el->pbg_nt[c] = 2.0 * el->pbg_ne[c] - el->pbg_nprev[c] * (N / el->pb_N);
            if (el->pbg_nt[c] < 1e-300) { el->pbg_nt[c] = 1e-300; }
            st += el->pbg_nt[c];
        }
        for (c = 0; c < G; c++) { el->pbg_nt[c] *= N / st; }
        sfel_pbg_free(el, el->pbg_Phi, T, el->pbg_nt, el->pbg_Vt);
        for (c = 0; c < G; c++)
        {
            el->pbg_g[c] = el->pbg_Vt[c] + T * log(el->pbg_nt[c] / dV);
            gb          += el->pbg_nt[c] * el->pbg_g[c];
        }
        gb /= N;
        for (c = 0; c < G; c++) { r += el->pbg_nt[c] * sqr(el->pbg_g[c] - gb); }
        resid_ex = sqrt(r / N) / T;
        el->td[SFEL_TD_EXTRAP] += sfel_pb_wallms() - tx0;
    }

    /* the density at T (Newton, inner); T is then moved until it is the
     * electrons' own temperature: the root of g(T) = E_tot - U(n_T) - 3/2 N T,
     * which decreases with T (outer; bracketed secant, geometric bisection) */
    const gmx_bool tsc = el->pb_etot_init && el->pb_nsolve > 0 && N > 0;
    double         resid0 = -1, Tlo = 0, glo = 0, Thi = 0, ghi = 0;
    gmx_bool       have_lo = FALSE, have_hi = FALSE;

    for (nT = 1;; nT++)
    {
        /* Newton */
        F = sfel_pbg_free(el, el->pbg_Phi, T, el->pbg_ne, el->pbg_V);
        for (it = 0; it < 60; it++)
        {
            double gbar = 0, alpha, Ft = 0;
            int    ls;
            gmx_bool accept;

#pragma omp parallel for reduction(+:gbar)
            for (c = 0; c < G; c++)
            {
                el->pbg_g[c] = el->pbg_V[c] + T * log(el->pbg_ne[c] / dV);
                gbar        += el->pbg_ne[c] * el->pbg_g[c];
            }
            gbar /= N;
            resid = 0;
#pragma omp parallel for reduction(+:resid)
            for (c = 0; c < G; c++) { resid += el->pbg_ne[c] * sqr(el->pbg_g[c] - gbar); }
            resid = sqrt(resid / N) / T;
            if (it == 0) { resid0 = resid; }
            if (resid < el->pbg_ntol)
            {
                break;
            }
#pragma omp parallel for
            for (c = 0; c < G; c++) { el->pbg_g[c] -= gbar; }
            {
                const double tg0 = el->tdet ? sfel_pb_wallms() : 0, tcv0 = el->td[SFEL_TD_CONV];

                cgit += sfel_pbg_pcg(el, T, el->pbg_ne, el->pbg_g, el->pbg_d);
                if (el->tdet)
                {
                    el->td[SFEL_TD_CG]     += sfel_pb_wallms() - tg0;
                    el->td[SFEL_TD_CGCONV] += el->td[SFEL_TD_CONV] - tcv0;
                }
            }
            /* line search: take the step when F or the residual falls (no
             * convolution: V_t is known), at most 8 halvings.  Until 2026-10-25
             * only a fall of F counted, halving up to 30 times; near the target
             * the fall of F (~ N T resid^2) is below the single-precision noise
             * of F, so a tenth of the solves stalled after ~400 convolutions. */
            accept = FALSE;
            for (alpha = 1.0, ls = 0; ls < 8; ls++, alpha *= 0.5)
            {
                double st = 0;

#pragma omp parallel for reduction(+:st)
                for (c = 0; c < G; c++)
                {
                    double u = alpha * el->pbg_d[c] / el->pbg_ne[c];

                    if (u > 3.0) { u = 3.0; }
                    if (u < -3.0) { u = -3.0; }
                    el->pbg_nt[c] = el->pbg_ne[c] * exp(u);
                    if (el->pbg_nt[c] < 1e-300) { el->pbg_nt[c] = 1e-300; }
                    st += el->pbg_nt[c];
                }
#pragma omp parallel for
                for (c = 0; c < G; c++) { el->pbg_nt[c] *= N / st; }
                Ft = sfel_pbg_free(el, el->pbg_Phi, T, el->pbg_nt, el->pbg_Vt);
                if (Ft <= F)
                {
                    accept = TRUE;
                    break;
                }
                {
                    double gt = 0, rt = 0;

#pragma omp parallel for reduction(+:gt)
                    for (c = 0; c < G; c++)
                    {
                        el->pbg_g[c] = el->pbg_Vt[c] + T * log(el->pbg_nt[c] / dV);
                        gt          += el->pbg_nt[c] * el->pbg_g[c];
                    }
                    gt /= N;
#pragma omp parallel for reduction(+:rt)
                    for (c = 0; c < G; c++) { rt += el->pbg_nt[c] * sqr(el->pbg_g[c] - gt); }
                    if (sqrt(rt / N) / T < resid)
                    {
                        accept = TRUE;
                        break;
                    }
                }
            }
            if (!accept)
            {
                break;      /* no descent left at single-precision convolutions */
            }
            F = Ft;
            memcpy(el->pbg_ne, el->pbg_nt, sizeof(double) * G);
            memcpy(el->pbg_V, el->pbg_Vt, sizeof(double) * G);
        }

        if (!tsc || nT >= 12)
        {
            break;
        }
        {
            const double U  = sfel_pbg_energy(el, el->pbg_Phi, el->pbg_ne, 1.0);
            /* sfmd-laser-polarization: E_tot with the field's work on this density */
            const double Et = pol_on ? el->pb_Etot + pol_Emid * (pol_Xold - sfel_pol_X(el, el->pbg_ne)) : el->pb_Etot;
            const double g  = Et - U - 1.5 * N * T;
            const double Tt = (Et - U) / (1.5 * N);
            double       Tn;

            if (fabs(g) <= 1e-2 * 1.5 * N * T || (T <= Tmin && g < 0))
            {
                break;
            }
            if (g > 0) { Tlo = T; glo = g; have_lo = TRUE; }
            else       { Thi = T; ghi = g; have_hi = TRUE; }
            if (have_lo && have_hi)
            {
                Tn = Tlo + glo * (Thi - Tlo) / (glo - ghi);
                if (!(Tn > Tlo + 0.05 * (Thi - Tlo) && Tn < Thi - 0.05 * (Thi - Tlo)))
                {
                    Tn = sqrt(Tlo * Thi);
                }
            }
            else if (g > 0)
            {
                Tn = (Tt < 4.0 * T) ? Tt : 4.0 * T;
            }
            else
            {
                Tn = (Tt > 0.25 * T) ? Tt : 0.25 * T;
                if (Tn < Tmin) { Tn = Tmin; }
            }
            T = Tn;
        }
    }

    if (el->tdet)
    {
        el->td[SFEL_TD_NEWTON] += sfel_pb_wallms() - tn0;
        el->td_nsolve          += 1;
        el->td_nskip           += (resid0 >= 0 && resid0 < el->pbg_ntol);
        if (resid0 >= 0 && resid_ex >= 0) { el->td_newton0 += resid0; el->td_newtonex += resid_ex; }
        if (el->ftd == NULL)
        {
            el->ftd = sfel_open_output("pbg_timing.txt", "w");
            fprintf(el->ftd, "# t[ps] newton cg convolutions conv_rhs2_total t_fft_total[ms] t_conv_total[ms] "
                    "t_cg_total[ms] t_cgconv_total[ms] t_newton_total[ms] resid0 resid_extrapolated rebuilt\n");
        }
        fprintf(el->ftd, "%.5f %d %d %d %.0f %.0f %.0f %.0f %.0f %.0f %.3e %.3e %d\n", t, it, cgit, el->pbg_nconv,
                el->td_rhs2, el->td[SFEL_TD_FFT], el->td[SFEL_TD_CONV], el->td[SFEL_TD_CG], el->td[SFEL_TD_CGCONV],
                el->td[SFEL_TD_NEWTON], resid0, resid_ex, rebuilt);
        fflush(el->ftd);
        if (el->pbg_nprev == NULL) { snew(el->pbg_nprev, (size_t)el->pbg_ncap * el->pbg_ncap * el->pbg_ncap); }
        memcpy(el->pbg_nprev, el->pbg_ne, sizeof(double) * G);
        el->pbg_nprev_ok = TRUE;
    }

    /* the electron potential on the grid (for the forces and the escape test),
     * per-cell weights and depths (evaporation), centroid, local n_e at the
     * atoms (collisions), diagnostics */
    sfel_pbg_conv(el, el->pbg_ne, el->pbg_Kn);
    if (el->pbg_Ve == NULL) { snew(el->pbg_Ve, (size_t)el->pbg_ncap * el->pbg_ncap * el->pbg_ncap); }
    for (d = 0; d < 3; d++) { el->pbg_ecen[d] = 0; }
    for (i = 0; i < ng; i++)
    {
        for (j = 0; j < ng; j++)
        {
            for (k = 0; k < ng; k++)
            {
                double xc[3] = {el->pbg_lo[0] + i * el->pbg_dx, el->pbg_lo[1] + j * el->pbg_dx,
                                el->pbg_lo[2] + k * el->pbg_dx};

                c               = SFEL_PBG_IDX(i, j, k);
                el->pbg_Ve[c]   = (float)el->pbg_Kn[c];
                Dmean          -= el->pbg_ne[c] * el->pbg_V[c];
                if (-el->pbg_V[c] > Dmax && el->pbg_ne[c] > 1e-6 * N / G) { Dmax = -el->pbg_V[c]; }
                for (d = 0; d < 3; d++) { el->pbg_ecen[d] += el->pbg_ne[c] * xc[d]; }
                if (sqr(xc[0] - el->com[0]) + sqr(xc[1] - el->com[1]) + sqr(xc[2] - el->com[2]) <
                    0.25 * sqr(el->radius_now))
                {
                    ecore += el->pbg_ne[c];
                }
            }
        }
    }
    Dmean /= N;
    for (d = 0; d < 3; d++) { el->pbg_ecen[d] /= N; }
    el->pb_evn = (int)G;
    el->pb_evw = el->pbg_ne;
    el->pb_evV = el->pbg_V;
    /* evaporation runs every step: only over the cells that hold electrons
     * (the rest together hold < 3e-5 N) */
    if (el->pb_evlist == NULL) { snew(el->pb_evlist, (size_t)el->pbg_ncap * el->pbg_ncap * el->pbg_ncap); }
    el->pb_evnl = 0;
    for (c = 0; c < G; c++)
    {
        if (el->pbg_ne[c] > 1e-10 * N) { el->pb_evlist[el->pb_evnl++] = (int)c; }
    }
    for (a = 0; a < n; a++)
    {
        int    i0[3];
        double f[3];

        el->pb_ne[a] = 0;
        if (sfel_pbg_cic(el, x[a], i0, f))
        {
            int u, v, w;

            for (u = 0; u < 2; u++)
            {
                for (v = 0; v < 2; v++)
                {
                    for (w = 0; w < 2; w++)
                    {
                        el->pb_ne[a] += (u ? f[0] : 1 - f[0]) * (v ? f[1] : 1 - f[1]) * (w ? f[2] : 1 - f[2]) *
                                        el->pbg_ne[SFEL_PBG_IDX(i0[0] + u, i0[1] + v, i0[2] + w)] / dV;
                    }
                }
            }
        }
        if (sqr(x[a][0] - el->com[0]) + sqr(x[a][1] - el->com[1]) + sqr(x[a][2] - el->com[2]) <
            0.25 * sqr(el->radius_now))
        {
            qcore += md->chargeA[a];
        }
    }

    /* energy bookkeeping */
    {
        const double K0 = el->screen_E_trapped;

        Unew = sfel_pbg_energy(el, el->pbg_Phi, el->pbg_ne, 1.0);
        if (etot_on && el->pb_ediag && el->pbd_q != NULL)
        {
            if (el->fpbd == NULL)
            {
                el->fpbd = sfel_open_output("pb_ediag.txt", "w");
                fprintf(el->fpbd, "# t[ps] N dK_solve  m1_motion m2_charges m3_N m4_regrid(not charged) m5_rearrange  "
                        "booked_W booked_J booked_N  [eV; dK_solve = -(m1+m2+m3+m5)]\n");
            }
            fprintf(el->fpbd, "%.5f %.0f %.1f  %.1f %.1f %.1f %.1f %.1f  %.1f %.1f %.1f\n", t, N,
                    (el->pb_Etot - Unew) - K0,
                    (U1 - el->pbd_Ulast) - el->pbd_W, (U2 - U1) - el->pbd_J, (U3 - U2) - el->pbd_N,
                    U4 - U3, Unew - U4, el->pbd_W, el->pbd_J, el->pbd_N);
            fflush(el->fpbd);
        }
        if (el->pb_ediag)
        {
            if (el->pbd_q == NULL) { snew(el->pbd_q, n); }
            for (a = 0; a < n; a++) { el->pbd_q[a] = md->chargeA[a]; }
            el->pbd_Ulast = Unew;
            el->pbd_W = el->pbd_J = el->pbd_N = 0;
        }
        if (pol_on)
        {
            /* the field's work on the cloud (see above), and the dipole of
             * ions and cloud about com for the saddle */
            const double Xnew = sfel_pol_X(el, el->pbg_ne);

            if (tsc)
            {
                const double W = pol_Emid * (pol_Xold - Xnew);

                el->pb_Etot += W;
                el->eb_pol  += W;
            }
            el->pol_X = Xnew;
            for (d = 0; d < 3; d++) { el->pol_p[d] = 0; }
            for (a = 0; a < n; a++)
            {
                for (d = 0; d < 3; d++) { el->pol_p[d] += md->chargeA[a] * (x[a][d] - el->com[d]); }
            }
            for (i = 0; i < ng; i++)
            {
                for (j = 0; j < ng; j++)
                {
                    for (k = 0; k < ng; k++)
                    {
                        const double w = el->pbg_ne[SFEL_PBG_IDX(i, j, k)];

                        el->pol_p[0] -= w * (el->pbg_lo[0] + i * el->pbg_dx - el->com[0]);
                        el->pol_p[1] -= w * (el->pbg_lo[1] + j * el->pbg_dx - el->com[1]);
                        el->pol_p[2] -= w * (el->pbg_lo[2] + k * el->pbg_dx - el->com[2]);
                    }
                }
            }
        }
        if (!el->pb_etot_init)
        {
            el->pb_Etot      = K0 + Unew;
            el->pb_etot_init = TRUE;
        }
        el->screen_E_trapped = el->pb_Etot - Unew;
        /* old model: E_kin >= 0, E_tot raised to match (energy created,
         * pb_Eclamp).  With sfmd-te-min > 0, E_kin is kept as it is, below 0
         * if need be: the density is the floored one, the energy the actual. */
        if (el->te_min <= 0 && el->screen_E_trapped < 0)
        {
            el->pb_Eclamp        -= el->screen_E_trapped;
            el->screen_E_trapped  = 0;
            el->pb_Etot           = Unew;
        }
        dU = el->screen_E_trapped - K0;
    }
    el->pb_h     = el->pbg_dx;
    el->pb_N     = N;
    el->pb_Dmean = Dmean;

    if (el->fpb == NULL)
    {
        el->fpb = sfel_open_output("pb_log.txt", "w");
        fprintf(el->fpb, "# t[ps] N Q T_e[eV] dx[nm] ions_outside newton cg resid wall[ms] D_mean[eV] D_max[eV] "
                "W_interval[eV] dU_interval[eV] E_kin[eV] E_clamped[eV] core_e/q(r<R/2) convolutions regrids T_iterations\n");
        if (el->pbg_dens_out)
        {
            el->fpbw = sfel_open_output("pbg_density.bin", "wb");
        }
    }
    fprintf(el->fpb, "%.5f %.0f %.0f %.2f %.4f %d %d %d %.2e %.0f %.1f %.1f %.1f %.1f %.1f %.1f %.4f %d %d %d\n", t, N,
            el->escaped_photoelectrons, T, el->pbg_dx, nout, it, cgit, resid, sfel_pb_wallms() - t0, Dmean, Dmax,
            el->pb_W, dU, el->screen_E_trapped, el->pb_Eclamp, qcore > 0 ? ecore / qcore : 0.0, el->pbg_nconv,
            el->pbg_nregrid, nT);
    fflush(el->fpb);
    /* GMX_SFMD_PBG_DENSITY=1 (diagnostic, off since 2026-10-27: ~80 % of a run's output): the density every
     * 10th solve: header t[ps] N T dx lo[3] ng, then ng^3 floats n_e [nm^-3], x slowest */
    if (el->fpbw != NULL && el->pb_nsolve % 10 == 0)
    {
        float hdr[8] = {(float)t, (float)N, (float)T, (float)el->pbg_dx, (float)el->pbg_lo[0], (float)el->pbg_lo[1],
                        (float)el->pbg_lo[2], (float)ng};

        fwrite(hdr, sizeof(float), 8, el->fpbw);
        for (c = 0; c < G; c++)
        {
            float v = (float)(el->pbg_ne[c] / dV);

            fwrite(&v, sizeof(float), 1, el->fpbw);
        }
        fflush(el->fpbw);
    }
    el->pb_nsolve++;
    el->tm[SFEL_TM_SOLVE] += sfel_pb_wallms() - t0;
    el->pb_W = 0;
}

/* electron potential (eV, repulsive for an electron) at x and its gradient;
 * outside the grid the cloud is a point charge at its centroid */
static double sfel_pbg_Ve(const t_sfelec *el, const real *x, double grad[3])
{
    const int ng = el->pbg_n;
    int       i0[3], d;
    double    f[3];

    if (sfel_pbg_cic(el, x, i0, f) && i0[0] >= 1 && i0[1] >= 1 && i0[2] >= 1 &&
        i0[0] < ng - 2 && i0[1] < ng - 2 && i0[2] < ng - 2)
    {
        const float *V = el->pbg_Ve;
        double       v = 0;
        int          a, b, c;

        for (d = 0; d < 3; d++) { grad[d] = 0; }
        for (a = 0; a < 2; a++)
        {
            for (b = 0; b < 2; b++)
            {
                for (c = 0; c < 2; c++)
                {
                    const int    i = i0[0] + a, j = i0[1] + b, k = i0[2] + c;
                    const double wt = (a ? f[0] : 1 - f[0]) * (b ? f[1] : 1 - f[1]) * (c ? f[2] : 1 - f[2]);

                    v       += wt * V[SFEL_PBG_IDX(i, j, k)];
                    grad[0] += wt * (V[SFEL_PBG_IDX(i + 1, j, k)] - V[SFEL_PBG_IDX(i - 1, j, k)]);
                    grad[1] += wt * (V[SFEL_PBG_IDX(i, j + 1, k)] - V[SFEL_PBG_IDX(i, j - 1, k)]);
                    grad[2] += wt * (V[SFEL_PBG_IDX(i, j, k + 1)] - V[SFEL_PBG_IDX(i, j, k - 1)]);
                }
            }
        }
        for (d = 0; d < 3; d++) { grad[d] /= 2.0 * el->pbg_dx; }
        return v;
    }
    {
        double r[3], r2 = 0, rr;

        for (d = 0; d < 3; d++) { r[d] = x[d] - el->pbg_ecen[d]; r2 += r[d] * r[d]; }
        rr = sqrt(r2);
        for (d = 0; d < 3; d++) { grad[d] = -SFEL_COULOMB_EV_NM * el->pb_N * r[d] / (r2 * rr); }
        return SFEL_COULOMB_EV_NM * el->pb_N / rr;
    }
}

/* Depth, eV, of an electron at atom k: ions other than k (direct, smoothed
 * over the grid kernel's length, sfel_pbg_s), the electron cloud from the grid. */
static double sfel_pbg_depth(const t_sfelec *el, const t_mdatoms *md, rvec *x, int k)
{
    const double s  = el->pbg_have ? sfel_pbg_s(el) : 0.1;
    const double sc = (el->pb_ready && el->pb_N > 0) ? el->num_free_electrons / el->pb_N : 0.0;
    double       U  = 0, grad[3];
    int          j;

    for (j = 0; j < md->nr; j++)
    {
        if (j != k && md->chargeA[j] != 0)
        {
            U += SFEL_COULOMB_EV_NM * md->chargeA[j] * sfel_pb_G(sqrt(sfel_pb_d2(x[k], x[j])), s);
        }
    }
    if (sc > 0 && el->pbg_Ve != NULL)
    {
        U -= sc * sfel_pbg_Ve(el, x[k], grad);
    }
    return U;
}


static double sfel_barrier(const t_sfelec *el);
static double sfel_barrier_now(const t_sfelec *el);

/* sfmd-laser-barrier, master, right after a solve: the cells whose depth is
 * below the laser-lowered escape level (-pbg_V <= dW) spill out at once.
 * Their electrons leave with their kinetic energy, 3/2 T each, and take their
 * electrostatic energy out of the cloud, U(before) - U(after), computed
 * exactly; E_kin of the rest is unchanged.  The net charge grows, so dW grows
 * (as sqrt Q) and the well deepens (as Q): repeated with a fresh solve it
 * settles in a few rounds.  Returns 0 if (almost) nothing spilled, 1 if the
 * cloud must be solved again, 2 if it is gone (the next solve starts from the
 * ions). */
static int sfel_spill(t_sfelec *el)
{
    const int    ng = el->pbg_n;
    const size_t G  = (size_t)ng * ng * ng;
    const double N  = el->num_free_electrons, dW = sfel_barrier_now(el);
    double       dN = 0, U0, U1, T, dK;
    size_t       c;

    /* with the polarization the level is the saddle's, which a dipole can
     * put above zero (dW < 0); the field term can lift cells above it */
    if ((el->pol_axis < 0 && dW <= 0) || N <= 0)
    {
        return 0;
    }
    for (c = 0; c < G; c++)
    {
        if (-el->pbg_V[c] <= dW) { dN += el->pbg_ne[c]; }
    }
    /* below 1 % of the cloud (or 2 electrons) a re-solve is not worth it: those cells sit
     * above the lowered level in the evaporation histogram too, where
     * Q(3/2, 0) = 1, so they leave within about one e-e collision time */
    if (dN < 2.0 || dN < 1e-2 * N)
    {
        return 0;
    }
    T  = sfel_plasma_Te(el);
    if (el->te_min > 0 && T < 0) { T = 0; }   /* E_kin < 0 under the floor: the spilled take none */
    U0 = sfel_pbg_energy(el, el->pbg_Phi, el->pbg_ne, 1.0);
    if (N - dN < 1.0)
    {
        dN = N;
        memset(el->pbg_ne, 0, sizeof(double) * G);
        U1 = 0;
        el->pol_X = 0;
    }
    else
    {
        if (el->pol_axis >= 0)
        {
            /* the spilled electrons leave the tracked X */
            int i, j, k;

            for (i = 0; i < ng; i++)
            {
                for (j = 0; j < ng; j++)
                {
                    for (k = 0; k < ng; k++)
                    {
                        const int    ix[3] = {i, j, k};
                        const size_t cc    = SFEL_PBG_IDX(i, j, k);

                        if (-el->pbg_V[cc] <= dW)
                        {
                            el->pol_X -= el->pbg_ne[cc] * (el->pbg_lo[el->pol_axis] + ix[el->pol_axis] * el->pbg_dx);
                        }
                    }
                }
            }
        }
        for (c = 0; c < G; c++)
        {
            if (-el->pbg_V[c] <= dW) { el->pbg_ne[c] = 0; }
        }
        U1 = sfel_pbg_energy(el, el->pbg_Phi, el->pbg_ne, 1.0);
    }
    dK = 1.5 * T * dN;
    if (dK > el->screen_E_trapped)
    {
        dK = (el->te_min > 0 && el->screen_E_trapped < 0) ? 0 : el->screen_E_trapped;
    }
    el->pb_Etot                -= dK + (U0 - U1);
    el->screen_E_trapped       -= dK;
    el->num_free_electrons     -= dN;
    el->escaped_photoelectrons += dN;
    el->count_spilled          += dN;
    el->eb_spill               += dK;
    el->spill_last             += dN;
    el->pb_N                    = el->num_free_electrons;
    if (el->num_free_electrons < 1.0)
    {
        /* gone: no cloud until new electrons are trapped, and that one
         * starts from the ions */
        el->escaped_photoelectrons += el->num_free_electrons;
        el->count_spilled          += el->num_free_electrons;
        el->num_free_electrons      = 0;
        el->pol_X                   = 0;
        el->screen_E_trapped        = 0;
        el->pb_Etot                 = 0;
        el->pb_N                    = 0;
        el->pbg_cold                = TRUE;
        return 2;
    }
    return 1;
}

/* laser_log.txt, one line per solve (master) */
static void sfel_laser_log(t_sfelec *el, double t, int rounds)
{
    const double dW = sfel_barrier(el), Q = el->escaped_photoelectrons;

    if (el->flaser == NULL)
    {
        el->flaser = sfel_open_output("laser_log.txt", "w");
        fprintf(el->flaser, "# t[ps] E0[V/nm] Q N dW[eV] r_saddle[nm] R[nm] spilled_this_solve spill_rounds "
                "spilled_total escaped_at_birth_total evaporated_total T_e[eV] heating[eV/fs] laser_energy_total[eV] "
                "nu/omega |E_in|/E0 Z_eff  (sfmd-laser-barrier: dW = 2 sqrt(kQeE0), r_saddle = sqrt(kQ/(eE0)); "
                "heating: last step, sfmd-laser-heating)\n");
    }
    if (el->pol_axis >= 0)
    {
        /* sfmd-laser-polarization: dW = -U_s and r_saddle the saddle's; then
         * the field of the solve, the dipole along the axis and the
         * polarization work */
        double   rho, Us;
        gmx_bool nosad;

        if (el->pol_logged == 0)
        {
            fprintf(el->flaser, "# sfmd-laser-polarization %c: dW = -U_saddle (net charge, dipole, field E of the solve), "
                    "r_saddle from com along -sign(E) axis; extra columns: E[V/nm] p_axis[e nm] polarization_work[eV] "
                    "saddle_at_surface_solves (no saddle outside the sample: level at R_e along the axis)\n", "xyz"[el->pol_axis]);
            el->pol_logged = 1;
        }
        sfel_pol_saddle(el, Q, el->pol_E, &rho, &Us, &nosad);
        fprintf(el->flaser, "%.5f %.4g %.1f %.1f %.2f %.3f %.4f %.1f %d %.1f %.0f %.1f %.3f %.4g %.1f %.3g %.3f %.2f "
                "%.4g %.2f %.1f %.0f\n", t,
                el->E0, Q, el->num_free_electrons, dW, rho,
                el->radius_now, el->spill_last, rounds, el->count_spilled, el->count_escaped_birth,
                el->screen_evaporated, sfel_plasma_Te(el), el->heat_P, el->eb_laser, el->heat_nu_w, el->heat_Ein,
                el->heat_Zeff, el->pol_E, el->pol_p[el->pol_axis], el->eb_pol, el->pol_nosaddle);
        fflush(el->flaser);
        el->spill_last = 0;
        return;
    }
    fprintf(el->flaser, "%.5f %.4g %.1f %.1f %.2f %.3f %.4f %.1f %d %.1f %.0f %.1f %.3f %.4g %.1f %.3g %.3f %.2f\n", t,
            el->E0, Q, el->num_free_electrons, dW, (el->E0 > 0 && Q > 0) ? sqrt(SFEL_COULOMB_EV_NM * Q / el->E0) : 0.0,
            el->radius_now, el->spill_last, rounds, el->count_spilled, el->count_escaped_birth,
            el->screen_evaporated, sfel_plasma_Te(el), el->heat_P, el->eb_laser, el->heat_nu_w, el->heat_Ein,
            el->heat_Zeff);
    fflush(el->flaser);
    el->spill_last = 0;
}

static void sfel_pbg_forces(t_sfelec *el, const t_inputrec *ir, const t_mdatoms *md, rvec *x, rvec *v,
                            rvec *f, const t_commrec *cr)
{
    size_t       G;
    const double N  = el->num_free_electrons;
    double       work = 0, sc;
    int          a, d;

    if (N <= 0)
    {
        return;
    }
    if (el->pbg_nthreads == 0)
    {
        const char *e = getenv("GMX_SFMD_PBG_FFT_THREADS");

        el->pbg_nthreads = (e != NULL && atoi(e) > 0) ? atoi(e) : (PAR(cr) ? cr->nnodes : 1);
    }
    /* With the laser barrier the cloud can change a lot between solves: the
     * ions charge up under it at onset (the depth histogram goes stale) and
     * a marginal cloud can evaporate in a few steps.  Solve again then too:
     * when the ion charge Q + N has grown by 10 %, or N has dropped by 20 %. */
    if (!el->pb_ready || el->screen_step - el->pb_step_solve >= el->pb_nstep || N > 1.2 * el->pb_N + 10 ||
        el->pb_N <= 0 ||
        (el->laser_on && (el->escaped_photoelectrons + N > 1.1 * el->pb_Z + 5 || N < 0.8 * el->pb_N - 5)))
    {
        double hb[17];

        if (MASTER(cr))
        {
            const double t = sfel_time(el, ir, el->screen_step);
            int          r = 0, rounds = 0;

            sfel_pbg_solve(el, ir, md, x, t);
            if (el->laser_on)
            {
                /* spill what the lowered level no longer holds, solve the
                 * rest again; at most 6 rounds */
                while (rounds < 6 && (r = sfel_spill(el)) == 1)
                {
                    rounds++;
                    sfel_pbg_solve(el, ir, md, x, t);
                }
                rounds += (r == 2);
            }
            if (el->laser_on || el->heat_on)
            {
                sfel_laser_log(el, t, rounds);
            }
        }
        sfel_park(el, cr);
        if (PAR(cr))
        {
            /* the grid limit is set on the master at its first solve */
            gmx_bcast(sizeof(int), &el->pbg_n, cr);
            gmx_bcast(sizeof(int), &el->pbg_ncap, cr);
        }
        if (el->pbg_Ve == NULL) { snew(el->pbg_Ve, (size_t)el->pbg_ncap * el->pbg_ncap * el->pbg_ncap); }
        G = (size_t)el->pbg_n * el->pbg_n * el->pbg_n;
        hb[0] = el->pb_N; hb[1] = el->pb_Dmean; hb[2] = el->screen_E_trapped; hb[3] = el->pbg_dx;
        for (d = 0; d < 3; d++) { hb[4 + d] = el->pbg_lo[d]; hb[7 + d] = el->pbg_ecen[d]; }
        hb[10] = el->pbg_have; hb[11] = el->pb_h; hb[12] = el->pb_Etot; hb[13] = el->pb_etot_init;
        /* a spill changes N and Q on the master after the step's broadcast */
        hb[14] = el->num_free_electrons; hb[15] = el->escaped_photoelectrons;
        hb[16] = el->pol_E;
        if (PAR(cr))
        {
            gmx_bcast(G * sizeof(float), el->pbg_Ve, cr);
            gmx_bcast(sizeof(hb), hb, cr);
        }
        el->pb_N = hb[0]; el->pb_Dmean = hb[1]; el->screen_E_trapped = hb[2]; el->pbg_dx = hb[3];
        for (d = 0; d < 3; d++) { el->pbg_lo[d] = hb[4 + d]; el->pbg_ecen[d] = hb[7 + d]; }
        el->pbg_have = (gmx_bool)hb[10]; el->pb_h = hb[11]; el->pb_Etot = hb[12]; el->pb_etot_init = (gmx_bool)hb[13];
        el->num_free_electrons = hb[14]; el->escaped_photoelectrons = hb[15];
        el->pol_E = hb[16];
        el->pb_Z = hb[14] + hb[15];
        el->pb_ready      = TRUE;
        el->pb_step_solve = el->screen_step;
    }
    /* N again: a spill in the solve above may have changed it */
    if (el->pb_N <= 0 || el->num_free_electrons <= 0)
    {
        return;
    }
    sc = el->num_free_electrons / el->pb_N;
    for (a = md->start; a < md->start + md->homenr; a++)
    {
        double grad[3];

        if (md->chargeA[a] == 0)
        {
            continue;
        }
        sfel_pbg_Ve(el, x[a], grad);
        for (d = 0; d < 3; d++)
        {
            double F = md->chargeA[a] * sc * grad[d] * SFEL_EV_TO_KJMOL;    /* kJ/mol/nm */

            f[a][d] += F;
            work    += F * v[a][d] * ir->delta_t;
        }
    }
    if (PAR(cr))
    {
        gmx_sumd(1, &work, cr);
    }
    work /= SFEL_EV_TO_KJMOL;
    el->pressure_work    += work;
    el->pb_W             += work;
    el->pb_Etot -= work;
    el->pbd_W   -= work;
}

/* ---------------------------------------------------------------- *
 * The laser on the cloud (sfmd-laser-barrier), cycle-averaged.
 *
 * Far from the sample the net charge Q (every electron that has left: at
 * birth, by spilling, by evaporation) and the laser's envelope field E0 make
 * the potential energy -kQ/r - e E0 z for an electron; its saddle, on the
 * field axis at r_s = sqrt(kQ/(e E0)), lies at -2 sqrt(k Q e E0).  An
 * electron is held only below that level, so every depth D (measured from
 * zero at infinity) is lowered by
 *     dW = 2 sqrt(k Q e E0(t)).
 * The cloud itself is overdense (omega_p several times omega_laser), so the
 * laser does not reach inside it and the equilibrium solve is unchanged; only
 * the level the electrons have to stay below moves.  Valid while the saddle
 * lies outside the sample (r_s > R); when it does not, dW exceeds the well and
 * everything spills, which is also what happens.
 * Not modelled (by construction): heating by the laser (inverse
 * bremsstrahlung, the Mie resonance), rescattering, and the drift energy of
 * an escaping electron.
 * ---------------------------------------------------------------- */
/* sfmd-laser-polarization: minus the saddle's level U_s (relative to the
 * field term's zero at com) for the current net charge and the field of the
 * last solve; the field term itself is in the cell potentials (spill,
 * evaporation) and added at the atom for a birth (sfel_escape_depth) */
static double sfel_pol_level(const t_sfelec *el)
{
    double   rho, Us;
    gmx_bool nosad;

    sfel_pol_saddle(el, el->escaped_photoelectrons, el->pol_E, &rho, &Us, &nosad);
    return -Us;
}

static double sfel_barrier(const t_sfelec *el)
{
    if (el->pol_axis >= 0)
    {
        return sfel_pol_level(el);
    }
    if (!el->laser_on || el->E0 <= 0 || el->escaped_photoelectrons <= 0)
    {
        return 0;
    }
    return 2.0 * sqrt(SFEL_COULOMB_EV_NM * el->escaped_photoelectrons * el->E0);
}

/* The lowering for the spill and the evaporation: the envelope's (mode 1), or
 * the instantaneous field's 2 sqrt(k Q e |E(t)|) (mode 2), which breathes
 * twice per optical cycle; the solves (every sfmd-pb-interval, ~14 per cycle
 * at 850 nm and 0.2 fs) sample it */
static double sfel_barrier_now(const t_sfelec *el)
{
    if (el->barrier_mode != 2 || el->pol_axis >= 0)
    {
        return sfel_barrier(el);
    }
    if (!el->laser_on || el->E_now <= 0 || el->escaped_photoelectrons <= 0)
    {
        return 0;
    }
    return 2.0 * sqrt(SFEL_COULOMB_EV_NM * el->escaped_photoelectrons * el->E_now);
}

/* Depth (eV) of a cell of potential energy V for evaporation, below the
 * escape level of the depth histogram's solve (pb_evh_dW, 0 without the
 * laser barrier): -V - dW. */
static double sfel_cell_depth(const t_sfelec *el, double V)
{
    return -V - el->pb_evh_dW;
}

static void sfel_ledger_start(t_sfelec *el, const t_mdatoms *md)
{
    int a;

    if (el->led_q == NULL)
    {
        snew(el->led_q, md->nr);
    }
    for (a = 0; a < md->nr; a++) { el->led_q[a] = md->chargeA[a]; }
}

/* The energy jumps of atom k's charge change since the ledger last saw it
 * (mcmd_ledger_step's loop body): ion-ion (log only) and ion-cloud, the
 * latter booked into E_tot with the cloud scale sc.
 *
 * sf: called at each ADK event, before its electron is booked, with the cloud
 * as it is at that moment, and at the end of the step for whatever is left
 * (nothing, with ADK the only process).  mc-md books all of a step's charge
 * changes at the end of the step, with the cloud scaled to the step's final
 * N: every pair of a new ion and an electron born after it in the same step
 * is then counted twice, in the electron's depth and in the ion's jump, an
 * error ~ n^2/2 pair energies for n events in one step.  Negligible for a few
 * photo/Auger events per step; ADK fires hundreds per step at onset (634 in
 * the first step of the lysozyme test), which drove E_kin below zero by
 * ~200 keV per solve (E_clamped).  Booking each change as it happens counts
 * every pair once, when its later member appears. */
static void sfel_ledger_atom(t_sfelec *el, const t_mdatoms *md, rvec *x, int k, double sc)
{
    int j;

    {
        const double dq = md->chargeA[k] - el->led_q[k];
        double       u  = 0;

        if (dq == 0)
        {
            return;
        }
        for (j = 0; j < md->nr; j++)
        {
            if (j != k && el->led_q[j] != 0)
            {
                u += el->led_q[j] / sqrt(sfel_pb_d2(x[k], x[j]));
            }
        }
        el->led_Jii += dq * SFEL_COULOMB_EV_NM * u;
        if (sc > 0 && el->pbg_Ve != NULL)
        {
            double grad[3];

            {
                const double dj = -dq * sc * sfel_pbg_Ve(el, x[k], grad);

                el->led_Jie += dj;
                el->pb_Etot += dj;
                el->pbd_J   += dj;
            }
        }
        el->led_q[k]  = md->chargeA[k];
        el->led_nchg += 1;
    }
}

static void sfel_ledger_step(t_sfelec *el, const t_mdatoms *md, rvec *x)
{
    const double sc = (el->pb_ready && el->pb_N > 0) ? el->num_free_electrons / el->pb_N : 0.0;
    int          k;

    if (el->led_q == NULL)
    {
        return;
    }
    for (k = 0; k < md->nr; k++)
    {
        sfel_ledger_atom(el, md, x, k, sc);
    }
}

static void sfel_bcast_plasma(t_sfelec *el, const t_commrec *cr)
{
        double buf[21];
        int    i, j;

        buf[0] = el->num_free_electrons; buf[1] = el->escaped_photoelectrons;
        buf[2] = el->screen_E_trapped;   buf[3] = el->radius_now;
        buf[4] = el->com[0]; buf[5] = el->com[1]; buf[6] = el->com[2];
        /* the ellipsoid too: the PB solve, which runs on every rank, picks its
         * sites by ellipsoidal distance */
        for (i = 0; i < 3; i++)
        {
            buf[7 + i] = el->ell_a[i];
            for (j = 0; j < 3; j++) { buf[10 + 3 * i + j] = el->ell_axis[i][j]; }
        }
        buf[19] = el->pb_Etot;
        buf[20] = el->pb_etot_init;
        gmx_bcast(sizeof(buf), buf, cr);
        el->pb_Etot      = buf[19];
        el->pb_etot_init = (gmx_bool)buf[20];
        el->num_free_electrons = buf[0]; el->escaped_photoelectrons = buf[1];
        el->screen_E_trapped   = buf[2]; el->radius_now = buf[3];
        el->com[0] = buf[4]; el->com[1] = buf[5]; el->com[2] = buf[6];
        for (i = 0; i < 3; i++)
        {
            el->ell_a[i] = buf[7 + i];
            for (j = 0; j < 3; j++) { el->ell_axis[i][j] = buf[10 + 3 * i + j]; }
        }
}

/* Evaporation tail at T over sites w (electrons), V (potential energy, eV),
 * the n entries of list (or 0..n-1 without one): the Maxwell tail above each
 * site's depth, Q(3/2, D/T) of the electrons, Q(5/2, D/T) of their energy,
 * and their depth. */
static void sfel_evap_tail(const double *w, const double *V, const int *list, int n, double T,
                           double *dN, double *dE, double *dND)
{
    int ib;

    *dN = *dE = *dND = 0;
    for (ib = 0; ib < n; ib++)
    {
        const int b = (list != NULL) ? list[ib] : ib;
        double    xb, ex3, q32;

        if (w[b] <= 0) { continue; }
        xb  = -V[b] / T;
        if (xb < 0) { xb = 0; }
        ex3 = exp(-xb);
        q32 = erfc(sqrt(xb)) + 2.0 * sqrt(xb / M_PI) * ex3;
        *dN  += w[b] * q32;
        *dND += w[b] * q32 * (-V[b]);
        *dE  += w[b] * 1.5 * T * (q32 + 4.0 * xb * sqrt(xb) * ex3 / (3.0 * sqrt(M_PI)));
    }
}

#define SFEL_EVH_NB 4000
/* The sites binned by depth: bin 0 holds depth <= 0 (potential set to 0, as
 * the tail clamps it), bins 1.. SFEL_EVH_NB the depths up to the deepest. */
static void sfel_evap_hist(t_sfelec *el)
{
    const int n = (el->pb_evlist != NULL) ? el->pb_evnl : el->pb_evn;
    double    Dmax = 0;
    int       ib, k;

    if (el->pb_evhw == NULL) { snew(el->pb_evhw, SFEL_EVH_NB + 1); snew(el->pb_evhV, SFEL_EVH_NB + 1); }
    if (el->laser_on && el->pb_evhVt == NULL) { snew(el->pb_evhVt, SFEL_EVH_NB + 1); snew(el->pb_evhK, SFEL_EVH_NB + 1); }
    if (el->pol_axis >= 0 && el->pb_evhX == NULL) { snew(el->pb_evhX, SFEL_EVH_NB + 1); }
    if (el->pol_axis >= 0)
    {
        for (k = 0; k <= SFEL_EVH_NB; k++) { el->pb_evhX[k] = 0; }
    }
    el->pb_evh_dW = sfel_barrier_now(el);
    for (k = 0; k <= SFEL_EVH_NB; k++) { el->pb_evhw[k] = el->pb_evhV[k] = 0; }
    if (el->laser_on)
    {
        for (k = 0; k <= SFEL_EVH_NB; k++) { el->pb_evhVt[k] = el->pb_evhK[k] = 0; }
    }
    for (ib = 0; ib < n; ib++)
    {
        const int b = (el->pb_evlist != NULL) ? el->pb_evlist[ib] : ib;

        if (el->pb_evw[b] > 0 && sfel_cell_depth(el, el->pb_evV[b]) > Dmax) { Dmax = sfel_cell_depth(el, el->pb_evV[b]); }
    }
    for (ib = 0; ib < n; ib++)
    {
        const int    b = (el->pb_evlist != NULL) ? el->pb_evlist[ib] : ib;
        const double D = sfel_cell_depth(el, el->pb_evV[b]);

        if (el->pb_evw[b] <= 0) { continue; }
        if (D <= 0)
        {
            k = 0;
            el->pb_evhw[0] += el->pb_evw[b];
        }
        else
        {
            k = 1 + (int)(D / Dmax * SFEL_EVH_NB);
            if (k > SFEL_EVH_NB) { k = SFEL_EVH_NB; }
            el->pb_evhw[k] += el->pb_evw[b];
            el->pb_evhV[k] += el->pb_evw[b] * (-D);
        }
        if (el->pol_axis >= 0)
        {
            /* the leavers take their potential energy without the field term
             * out of the cloud; their position for the tracked X */
            const int ng = el->pbg_n, ix[3] = {b / (ng * ng), (b / ng) % ng, b % ng};

            el->pb_evhVt[k] += el->pb_evw[b] * (el->pb_evV[b] - el->pbg_UL[b]);
            el->pb_evhK[k]  += el->pb_evw[b] * el->pbg_Ve[b];
            el->pb_evhX[k]  += el->pb_evw[b] * (el->pbg_lo[el->pol_axis] + ix[el->pol_axis] * el->pbg_dx);
        }
        else if (el->laser_on)
        {
            el->pb_evhVt[k] += el->pb_evw[b] * el->pb_evV[b];
            el->pb_evhK[k]  += el->pb_evw[b] * el->pbg_Ve[b];
        }
    }
    for (k = 1; k <= SFEL_EVH_NB; k++)
    {
        if (el->pb_evhw[k] > 0) { el->pb_evhV[k] /= el->pb_evhw[k]; }
    }
    if (el->laser_on)
    {
        for (k = 0; k <= SFEL_EVH_NB; k++)
        {
            if (el->pb_evhw[k] > 0) { el->pb_evhVt[k] /= el->pb_evhw[k]; el->pb_evhK[k] /= el->pb_evhw[k]; }
            if (el->pol_axis >= 0 && el->pb_evhw[k] > 0) { el->pb_evhX[k] /= el->pb_evhw[k]; }
        }
    }
    el->pb_evh_ver = el->pb_nsolve;
}


/* ---------------------------------------------------------------- *
 * The escape depth of a newborn electron: the cloud's own depth at the atom,
 * lowered by the laser barrier (sfel_barrier; 0 without sfmd-laser-barrier).
 * ---------------------------------------------------------------- */

/* Depth (eV, > 0 bound) that an electron born at rest at atom k must climb to
 * escape. */
static double sfel_escape_depth(const t_sfelec *el, const t_mdatoms *md, rvec *x, int k, double *D_true)
{
    *D_true = sfel_pb_depth(el, md, x, k);
    if (el->pol_axis >= 0)
    {
        /* the electron at rest at the atom, field term included, against the saddle */
        double   rho, Us, xr[3];
        gmx_bool nosad;
        int      d;

        sfel_pol_saddle(el, el->escaped_photoelectrons, el->pol_E, &rho, &Us, &nosad);
        for (d = 0; d < 3; d++) { xr[d] = x[k][d] - el->com[d]; }
        return *D_true - sfel_pol_UL(el, xr, el->pol_E, rho) + Us;
    }
    return *D_true - sfel_barrier(el);
}

/* ---------------------------------------------------------------- *
 * Sample geometry over the steps
 * ---------------------------------------------------------------- */

/* ---------------------------------------------------------------- *
 * sfmd-laser-heating: inverse bremsstrahlung, cycle-averaged (Ditmire et al.,
 * Phys. Rev. A 53, 3379 (1996)).  The sample (a few nm) is far smaller than
 * the wavelength, so the cloud is a small Drude sphere in a uniform field:
 *   eps  = 1 - omega_p^2 / (omega (omega + i nu)),
 *   E_in = 3 E0 / (eps + 2)                        (field inside),
 *   P    = (omega / 2) eps0 Im(eps) |E_in|^2 V      (absorbed power),
 * with the cloud's mean density n = N / (4/3 pi R^3), its temperature T and
 * the electron-ion collision rate (NRL)
 *   nu = 2.91e-6 Z n[cm^-3] lnL T[eV]^-3/2 s^-1, Z = sum q^2 / sum q,
 * lnL as in the evaporation, capped at sfmd-heating-nu-cap x omega_p: a cold
 * dense cloud is strongly coupled and the textbook rate meaningless there.
 * The formula covers shielding (omega_p >> omega, nu small: E_in small), the
 * Mie resonance (omega_p = sqrt(3) omega) and the resistive regime (nu >>
 * omega: E_in ~ E0).  The power goes into the cloud's kinetic energy and its
 * total energy: work from outside, logged as eb_laser (laser_log.txt).
 * Not included: heating at the surface (vacuum/Brunel heating), the
 * density profile (one mean density), and the energy given to electrons that
 * leave.  T below the solve's floor is taken at the floor.
 * ---------------------------------------------------------------- */
/* Carlson's symmetric elliptic integral R_D(x, y, z) (Numerical Recipes rd,
 * duplication to convergence) */
static double sfel_carlson_rd(double x, double y, double z)
{
    double sum = 0, fac = 1, ave, dx, dy, dz;
    int    it;

    for (it = 0; it < 100; it++)
    {
        double sx = sqrt(x), sy = sqrt(y), sz = sqrt(z), lam = sx * (sy + sz) + sy * sz;

        sum += fac / (sz * (z + lam));
        fac *= 0.25;
        x    = 0.25 * (x + lam);
        y    = 0.25 * (y + lam);
        z    = 0.25 * (z + lam);
        ave  = 0.2 * (x + y + 3.0 * z);
        dx   = (ave - x) / ave;
        dy   = (ave - y) / ave;
        dz   = (ave - z) / ave;
        if (fabs(dx) < 1e-6 && fabs(dy) < 1e-6 && fabs(dz) < 1e-6)
        {
            break;
        }
    }
    {
        const double ea = dx * dy, eb = dz * dz, ec = ea - eb, ed = ea - 6.0 * eb, ee = ed + ec + ec;

        return 3.0 * sum + fac * (1.0 + ed * (-3.0 / 14.0 + 9.0 / 88.0 * ed - 4.5 / 26.0 * dz * ee) +
                                  dz * (ee / 6.0 + dz * (-9.0 / 22.0 * ec + 3.0 / 26.0 * dz * ea))) /
               (ave * sqrt(ave));
    }
}

/* Depolarization factors of an ellipsoid with semi-axes a[]:
 * L_i = (a1 a2 a3 / 3) R_D(a_j^2, a_k^2, a_i^2), sum 1 (1/3 each for a sphere) */
static void sfel_depolarization(const double a[3], double L[3])
{
    const double p = a[0] * a[1] * a[2];
    double       s = 0;
    int          i;

    for (i = 0; i < 3; i++)
    {
        const double aj = a[(i + 1) % 3], ak = a[(i + 2) % 3];

        L[i] = p / 3.0 * sfel_carlson_rd(aj * aj, ak * ak, a[i] * a[i]);
        s   += L[i];
    }
    for (i = 0; i < 3; i++)
    {
        L[i] /= s;   /* exact to rounding already; keeps sum L = 1 */
    }
}

static void sfel_heat(t_sfelec *el, const t_inputrec *ir, const t_mdatoms *md)
{
    const double EPS0 = 8.8541878128e-12, QE = 1.602176634e-19;
    const double N = el->num_free_electrons, R = el->radius_now, w = el->omega_L;
    double       sq = 0, s2 = 0, Z, T, V_nm3, n_cm3, lnL, nu, wp, d, er, ei, E0, Ein2, P;
    int          a;

    el->heat_P = 0;
    if (!el->heat_on || N <= 0 || el->E0 <= 0 || R <= 0 || w <= 0)
    {
        return;
    }
    for (a = 0; a < md->nr; a++)
    {
        const double q = md->chargeA[a];

        if (q > 0) { sq += q; s2 += q * q; }
    }
    Z     = (sq > 0) ? s2 / sq : 1.0;
    T     = sfel_plasma_Te(el);
    {
        const double Tf = (el->te_min > 0) ? el->te_min : el->pb_Tmin;

        T = (T > Tf) ? T : Tf;
    }
    V_nm3 = 4.0 / 3.0 * M_PI * R * R * R;
    n_cm3 = N / (V_nm3 * 1e-21);
    lnL   = 24.0 - log(sqrt(n_cm3) / T);
    if (lnL < 2.0) { lnL = 2.0; }
    nu    = 2.91e-6 * Z * n_cm3 * lnL * pow(T, -1.5);
    wp    = 5.64e4 * sqrt(n_cm3);
    if (el->heat_cap > 0 && nu > el->heat_cap * wp) { nu = el->heat_cap * wp; }
    if (el->heat_surf > 0)
    {
        /* surface (Kreibig) damping of the plasmon of a nm-sized cloud:
         * electrons cross it in R / vbar, vbar = sqrt(8 T / (pi m_e));
         * keeps the Mie resonance finite when nu_ei of a hot, thin cloud is tiny */
        const double vbar = sqrt(8.0 * T * QE / (M_PI * 9.1093837015e-31));   /* m/s */

        nu += el->heat_surf * vbar / (R * 1e-9);
    }
    d     = w * w + nu * nu;
    er    = 1.0 - wp * wp / d;
    ei    = wp * wp * nu / (w * d);
    E0    = el->E0 * 1e9;                                     /* V/m */
    if (el->heat_shape == 1)
    {
        /* the sample's ellipsoid, E0 along a random direction: E0^2/3 per
         * axis, E_in,i = E0_i / (1 + L_i (eps - 1)) */
        int i;

        sfel_depolarization(el->ell_a, el->heat_L);
        Ein2 = 0;
        for (i = 0; i < 3; i++)
        {
            const double L = el->heat_L[i], re = 1.0 + L * (er - 1.0), im = L * ei;

            if (el->pol_axis >= 0)
            {
                /* sfmd-laser-polarization: E0's component along the axis i */
                const double c = el->ell_axis[i][el->pol_axis];

                Ein2 += E0 * E0 * c * c / (re * re + im * im);
            }
            else
            {
                Ein2 += E0 * E0 / 3.0 / (re * re + im * im);
            }
        }
    }
    else
    {
        Ein2  = 9.0 * E0 * E0 / ((er + 2.0) * (er + 2.0) + ei * ei);
    }
    P     = 0.5 * w * EPS0 * ei * Ein2 * V_nm3 * 1e-27;       /* W */
    {
        const double dE = P * ir->delta_t * 1e-12 / QE;       /* eV this step */

        el->screen_E_trapped += dE;
        el->pb_Etot          += dE;
        el->eb_laser         += dE;
        el->heat_P            = dE / (ir->delta_t * 1e3);     /* eV/fs */
    }
    el->heat_nu_w = nu / w;
    el->heat_Ein  = sqrt(Ein2) / E0;
    el->heat_Zeff = Z;
}

/* Reference geometry on the first step (mcmd_first_step) */
static void sfel_first_geometry(t_sfelec *el, const t_mdatoms *md, rvec *x)
{
    double rg = 0;

    sfel_sample_geometry(el, md, x, &rg);
    el->radius_of_sample = sqrt(5.0 / 3.0) * rg;
    el->radius_now       = el->radius_of_sample;
}

/* Every later step (mcmd_update_volume): the volume-equivalent radius of the
 * gyration ellipsoid */
static void sfel_update_geometry(t_sfelec *el, const t_mdatoms *md, rvec *x)
{
    double rg;

    sfel_sample_geometry(el, md, x, &rg);
    el->radius_now = cbrt(el->ell_a[0] * el->ell_a[1] * el->ell_a[2]);
}

/* Evaporation, one MD step (the plasma update of mcionize_step) */
static void sfel_evaporate(t_sfelec *el, const t_inputrec *ir)
{
    const double Q = el->escaped_photoelectrons, R = el->radius_now;

    if (el->te_min > 0 && el->num_free_electrons >= 1.0 && sfel_plasma_Te(el) < el->te_min)
    {
        el->te_floor_steps += 1;
    }
    if (el->num_free_electrons <= 0 || Q <= 0)
    {
        return;
    }
    {
        double T, n_cm3, lnL, nu, frac;

        /* At each cell the Maxwell tail above its depth leaves; the
         * tail refills at the e-e collision rate (NRL) of the mean
         * density, so per step only a fraction nu*dt of it goes. */
        const double tev0 = el->tdet ? sfel_pb_wallms() : 0;

        if (el->screen_evap && el->pb_ready && el->pb_evw != NULL)
        {
            /* the tail above each depth, Q(3/2, D/T) of the electrons,
             * Q(5/2, D/T) of their energy, from the histogram of the
             * cells' depths (rebuilt after every solve) */
            double dN = 0, dE = 0, dND = 0, sc = el->num_free_electrons / el->pb_N;
            double evA = 0, evB = 0;     /* laser barrier: depth and cloud-potential sums of the tail */
            double evX = 0;              /* sfmd-laser-polarization: x_axis sum of the tail */

            T = sfel_model_Te(el);
            if (T > 0)
            {
                if (el->pb_evh_ver != el->pb_nsolve)
                {
                    sfel_evap_hist(el);
                }
                sfel_evap_tail(el->pb_evhw, el->pb_evhV, NULL, SFEL_EVH_NB + 1, T, &dN, &dE, &dND);
                if (el->laser_on)
                {
                    /* The tail is above the lowered depth, but the leavers
                     * take their true potential energy out of the cloud,
                     * and in the cloud as it is now: V - (1 - sc) K n, the
                     * last solve's potential with the electrons gone since
                     * scaled out (as a birth sees the cloud).  With the
                     * barrier most of a marginal cloud can evaporate between
                     * two solves (936 of 1360 electrons in 0.2 fs at 1e16);
                     * booked against the full cloud, the leavers' mutual
                     * repulsion, ~ dn K dn / 2, came out of E_kin (0.5 MeV
                     * clamped).  Without the barrier the leavers are a tiny
                     * fraction and this is mc-md's booking.
                     * The scale is taken halfway through this step's own
                     * departures (below): when a marginal cloud evaporates
                     * half of itself in one step, the leavers' repulsion on
                     * each other is that of the thinning cloud. */
                    int kb;

                    for (kb = 0; kb <= SFEL_EVH_NB; kb++)
                    {
                        const double w = el->pb_evhw[kb];
                        double       xb, q32;

                        if (w <= 0) { continue; }
                        xb  = -el->pb_evhV[kb] / T;
                        if (xb < 0) { xb = 0; }
                        q32  = erfc(sqrt(xb)) + 2.0 * sqrt(xb / M_PI) * exp(-xb);
                        evA += w * q32 * -el->pb_evhVt[kb];
                        evB += w * q32 * el->pb_evhK[kb];
                        if (el->pol_axis >= 0) { evX += w * q32 * el->pb_evhX[kb]; }
                    }
                }
                n_cm3 = el->num_free_electrons / (4.0 / 3.0 * M_PI * R * R * R * 1e-21);
                lnL   = 24.0 - log(sqrt(n_cm3) / T);
                if (lnL < 2.0) { lnL = 2.0; }
                nu    = 2.91e-6 * n_cm3 * lnL * pow(T, -1.5);
                frac  = nu * ir->delta_t * 1e-12;
                if (frac > 1.0) { frac = 1.0; }
                dN *= sc * frac; dE *= sc * frac; dND *= sc * frac;
                {
                    const double dN_tail = dN;

                    if (dN > el->num_free_electrons) { dN = el->num_free_electrons; }
                    if (el->laser_on && dN_tail > 0)
                    {
                        const double sc_mid = (el->num_free_electrons - 0.5 * dN) / el->pb_N;

                        dND = (evA + (1.0 - sc_mid) * evB) * sc * frac * (dN / dN_tail);
                        if (el->pol_axis >= 0) { el->pol_X -= evX * sc * frac * (dN / dN_tail); }
                    }
                }
                /* never more kinetic energy than the cloud has (E_kin may be < 0 under sfmd-te-min) */
                if (dE > el->screen_E_trapped)
                {
                    dE = (el->te_min > 0 && el->screen_E_trapped < 0) ? 0 : el->screen_E_trapped;
                }
                el->pb_Etot                -= dE - dND;    /* kinetic energy minus depth of the leavers */
                el->pbd_N                  += dND;
                el->num_free_electrons     -= dN;
                el->screen_E_trapped       -= dE;
                el->eb_evap                += dE;
                el->escaped_photoelectrons += dN;
                el->screen_evaporated      += dN;
            }
            if (el->tdet) { el->td[SFEL_TD_EVAP] += sfel_pb_wallms() - tev0; }
        }
    }
}

/* ---------------------------------------------------------------- *
 * sfmd-pb-long-range: the long-range ion-ion Coulomb force from the grid.
 * GROMACS does the bare force within rcoulomb (rc).  The grid has every
 * pair, smoothed (kernel k G(r, s), s = 2 dx, after cloud-in-cell deposit),
 * and the smoothed pairs within rc are taken out again:
 *   E_a = -grad Phi(x_a) + sum_{b != a, r < rc} k q_b G'(r) (x_a - x_b) / r,
 * so pairs beyond rc come from the grid alone (error ~ erfc(rc / s)).  Every
 * pair within rc is taken out, excluded ones too: GROMACS gives those nothing
 * (or its 1-4 term), as without a cutoff.  The field is computed every
 * PB solve (sfmd-pb-interval) for every home atom and held, times the atom's
 * current charge, in between.  Atoms outside the grid stencil: the bare
 * pairs beyond rc directly.  GMX_SFMD_PBG_LR_CHECK=1 compares, at each refresh,
 * with the exact all-pairs field (master, O(N^2)).
 * ---------------------------------------------------------------- */

/* Value and gradient of the float grid g at x (cloud-in-cell of central
 * differences); FALSE when the stencil leaves the grid. */
static gmx_bool sfel_pbg_gradf(const t_sfelec *el, const float *g, const real *x, double *val, double grad[3])
{
    const int ng = el->pbg_n;
    int       i0[3], a, b, c, d;
    double    f[3];

    if (!(sfel_pbg_cic(el, x, i0, f) && i0[0] >= 1 && i0[1] >= 1 && i0[2] >= 1 &&
          i0[0] < ng - 2 && i0[1] < ng - 2 && i0[2] < ng - 2))
    {
        return FALSE;
    }
    *val = 0;
    for (d = 0; d < 3; d++) { grad[d] = 0; }
    for (a = 0; a < 2; a++)
    {
        for (b = 0; b < 2; b++)
        {
            for (c = 0; c < 2; c++)
            {
                const int    i = i0[0] + a, j = i0[1] + b, k = i0[2] + c;
                const double wt = (a ? f[0] : 1 - f[0]) * (b ? f[1] : 1 - f[1]) * (c ? f[2] : 1 - f[2]);

                *val    += wt * g[SFEL_PBG_IDX(i, j, k)];
                grad[0] += wt * (g[SFEL_PBG_IDX(i + 1, j, k)] - g[SFEL_PBG_IDX(i - 1, j, k)]);
                grad[1] += wt * (g[SFEL_PBG_IDX(i, j + 1, k)] - g[SFEL_PBG_IDX(i, j - 1, k)]);
                grad[2] += wt * (g[SFEL_PBG_IDX(i, j, k + 1)] - g[SFEL_PBG_IDX(i, j, k - 1)]);
            }
        }
    }
    for (d = 0; d < 3; d++) { grad[d] /= 2.0 * el->pbg_dx; }
    return TRUE;
}

/* The grid potential at x of a unit charge deposited at x itself (its own
 * term in Phi, eV per e). */
static double sfel_pbg_selfpot(const t_sfelec *el, const real *x)
{
    const double s = sfel_pbg_s(el);
    int          i0[3], u, v, d;
    double       f[3], w[8], p[8][3], U = 0;

    if (!sfel_pbg_cic(el, x, i0, f))
    {
        return 0;
    }
    for (u = 0; u < 8; u++)
    {
        const int o[3] = {(u >> 2) & 1, (u >> 1) & 1, u & 1};

        w[u] = 1;
        for (d = 0; d < 3; d++)
        {
            w[u]   *= o[d] ? f[d] : 1 - f[d];
            p[u][d] = o[d] * el->pbg_dx;
        }
    }
    for (u = 0; u < 8; u++)
    {
        for (v = 0; v < 8; v++)
        {
            U += w[u] * w[v] * sfel_pb_G(sqrt(sqr(p[u][0] - p[v][0]) + sqr(p[u][1] - p[v][1]) + sqr(p[u][2] - p[v][2])), s);
        }
    }
    return SFEL_COULOMB_EV_NM * U;
}

/* Cell list of the charged atoms, cells of at least rc (at most 64 a side). */
static void sfel_lr_cells(t_sfelec *el, const t_mdatoms *md, rvec *x)
{
    double hi[3];
    int    a, d, nct;

    if (el->lr_next == NULL) { snew(el->lr_next, md->nr); }
    for (d = 0; d < 3; d++) { el->lr_clo[d] = 1e30; hi[d] = -1e30; }
    for (a = 0; a < md->nr; a++)
    {
        if (md->chargeA[a] == 0) { continue; }
        for (d = 0; d < 3; d++)
        {
            if (x[a][d] < el->lr_clo[d]) { el->lr_clo[d] = x[a][d]; }
            if (x[a][d] > hi[d])         { hi[d] = x[a][d]; }
        }
    }
    el->lr_cs = el->lr_rc;
    for (d = 0; d < 3; d++)
    {
        if (hi[d] < el->lr_clo[d]) { el->lr_clo[d] = hi[d] = 0; }
        if ((hi[d] - el->lr_clo[d]) / 64.0 > el->lr_cs) { el->lr_cs = (hi[d] - el->lr_clo[d]) / 64.0; }
    }
    for (nct = 1, d = 0; d < 3; d++)
    {
        el->lr_nc[d] = (int)((hi[d] - el->lr_clo[d]) / el->lr_cs) + 1;
        nct         *= el->lr_nc[d];
    }
    if (nct > el->lr_headn)
    {
        srenew(el->lr_head, nct);
        el->lr_headn = nct;
    }
    for (a = 0; a < nct; a++) { el->lr_head[a] = -1; }
    for (a = 0; a < md->nr; a++)
    {
        int ci[3], c;

        if (md->chargeA[a] == 0) { continue; }
        for (d = 0; d < 3; d++)
        {
            ci[d] = (int)((x[a][d] - el->lr_clo[d]) / el->lr_cs);
            if (ci[d] >= el->lr_nc[d]) { ci[d] = el->lr_nc[d] - 1; }
        }
        c              = (ci[0] * el->lr_nc[1] + ci[1]) * el->lr_nc[2] + ci[2];
        el->lr_next[a] = el->lr_head[c];
        el->lr_head[c] = a;
    }
}

/* Long-range field E (eV/nm per e) and potential phi (eV per e) at atom a
 * from the other ions; qnear = the charge of the other ions within rc. */
static void sfel_lr_field(const t_sfelec *el, const t_mdatoms *md, rvec *x, int a, double E[3], double *phi,
                          double *qnear)
{
    const double rc2 = el->lr_rc * el->lr_rc, s = sfel_pbg_s(el), k = SFEL_COULOMB_EV_NM;
    double       grad[3], val;
    int          b, d;

    if (sfel_pbg_gradf(el, el->pbg_PhiLf, x[a], &val, grad))
    {
        int ci[3], u, v, w;

        for (d = 0; d < 3; d++) { E[d] = -grad[d]; }
        *phi   = val - md->chargeA[a] * sfel_pbg_selfpot(el, x[a]);
        *qnear = 0;
        for (d = 0; d < 3; d++)
        {
            ci[d] = (int)floor((x[a][d] - el->lr_clo[d]) / el->lr_cs);
            if (ci[d] < 0)                { ci[d] = 0; }
            if (ci[d] >= el->lr_nc[d])    { ci[d] = el->lr_nc[d] - 1; }
        }
        for (u = ci[0] - 1; u <= ci[0] + 1; u++)
        {
            for (v = ci[1] - 1; v <= ci[1] + 1; v++)
            {
                for (w = ci[2] - 1; w <= ci[2] + 1; w++)
                {
                    if (u < 0 || v < 0 || w < 0 || u >= el->lr_nc[0] || v >= el->lr_nc[1] || w >= el->lr_nc[2])
                    {
                        continue;
                    }
                    for (b = el->lr_head[(u * el->lr_nc[1] + v) * el->lr_nc[2] + w]; b >= 0; b = el->lr_next[b])
                    {
                        double dr[3], d2, r, g;

                        if (b == a) { continue; }
                        for (d = 0; d < 3; d++) { dr[d] = x[a][d] - x[b][d]; }
                        d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if (d2 >= rc2) { continue; }
                        r = sqrt(d2);
                        g = sfel_pb_dG(r, s);
                        for (d = 0; d < 3; d++) { E[d] += k * md->chargeA[b] * g * dr[d] / r; }
                        *phi   -= k * md->chargeA[b] * sfel_pb_G(r, s);
                        *qnear += md->chargeA[b];
                    }
                }
            }
        }
    }
    else
    {
        /* outside the grid: the bare pairs beyond rc */
        for (d = 0; d < 3; d++) { E[d] = 0; }
        *phi   = 0;
        *qnear = 0;
        for (b = 0; b < md->nr; b++)
        {
            double dr[3], d2, r;

            if (b == a || md->chargeA[b] == 0) { continue; }
            for (d = 0; d < 3; d++) { dr[d] = x[a][d] - x[b][d]; }
            d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            if (d2 < rc2) { *qnear += md->chargeA[b]; continue; }
            r = sqrt(d2);
            for (d = 0; d < 3; d++) { E[d] += k * md->chargeA[b] * dr[d] / (d2 * r); }
            *phi += k * md->chargeA[b] / r;
        }
    }
}

/* Every rank, every step (grid mode with sfmd-pb-long-range): refresh the field
 * when due, then add q E to the home atoms and book its work. */
static void sfel_pbg_lr(t_sfelec *el, const t_inputrec *ir, const t_mdatoms *md, rvec *x, rvec *v, rvec *f,
                        const t_commrec *cr)
{
    size_t       G;
    double       work = 0;
    int          a, d;

    if (!el->lr_ready || el->screen_step - el->lr_step >= el->pbg_lr_nstep)
    {
        const double t  = sfel_time(el, ir, el->screen_step);
        double       hb[6], U[2] = {0, 0}, cr_rms = -1, cr_max = -1;
        size_t       c;

        if (el->pbg_nthreads == 0)
        {
            const char *e = getenv("GMX_SFMD_PBG_FFT_THREADS");

            el->pbg_nthreads = (e != NULL && atoi(e) > 0) ? atoi(e) : (PAR(cr) ? cr->nnodes : 1);
        }
        if (MASTER(cr))
        {
            sfel_pbg_alloc(el, md->nr);
            if (el->pbg_PhiLf == NULL) { snew(el->pbg_PhiLf, (size_t)el->pbg_ncap * el->pbg_ncap * el->pbg_ncap); }
            if (el->pbg_PhiL == NULL) { snew(el->pbg_PhiL, (size_t)el->pbg_ncap * el->pbg_ncap * el->pbg_ncap); }
            /* the grid follows the sample here only while the PB solve does
             * not (no solve yet, or no electrons): the solve owns the rebuilds
             * and their energy bookkeeping */
            if (!el->pbg_have ||
                ((el->pb_nsolve == 0 || el->num_free_electrons <= 0) &&
                 (el->radius_now > 1.1 * el->pbg_R || el->radius_now < 0.9 * el->pbg_R ||
                  sqrt(sqr(el->com[0] - el->pbg_c[0]) + sqr(el->com[1] - el->pbg_c[1]) +
                       sqr(el->com[2] - el->pbg_c[2])) > el->pbg_dx)))
            {
                sfel_pbg_grid(el);
            }
            G           = (size_t)el->pbg_n * el->pbg_n * el->pbg_n;
            el->lr_nout = sfel_pbg_ionpot(el, md, x, el->pbg_PhiL);
            for (c = 0; c < G; c++) { el->pbg_PhiLf[c] = (float)el->pbg_PhiL[c]; }
        }
        sfel_park(el, cr);
        hb[0] = el->pbg_dx;
        for (d = 0; d < 3; d++) { hb[1 + d] = el->pbg_lo[d]; }
        hb[4] = el->pbg_have; hb[5] = el->lr_nout;
        if (PAR(cr))
        {
            /* the other ranks learn the grid limit (set on the master from the
             * sample) here, before their copy of the potential is made */
            gmx_bcast(sizeof(int), &el->pbg_n, cr);
            gmx_bcast(sizeof(int), &el->pbg_ncap, cr);
            if (el->pbg_PhiLf == NULL) { snew(el->pbg_PhiLf, (size_t)el->pbg_ncap * el->pbg_ncap * el->pbg_ncap); }
        }
        G = (size_t)el->pbg_n * el->pbg_n * el->pbg_n;
        if (PAR(cr))
        {
            gmx_bcast(G * sizeof(float), el->pbg_PhiLf, cr);
            gmx_bcast(sizeof(hb), hb, cr);
        }
        el->pbg_dx = hb[0];
        for (d = 0; d < 3; d++) { el->pbg_lo[d] = hb[1 + d]; }
        el->pbg_have = (gmx_bool)hb[4]; el->lr_nout = (int)hb[5];

        sfel_lr_cells(el, md, x);
        if (el->lr_E == NULL) { snew(el->lr_E, 3 * md->nr); }
        for (a = md->start; a < md->start + md->homenr; a++)
        {
            double phi, qn;

            sfel_lr_field(el, md, x, a, el->lr_E + 3 * a, &phi, &qn);
            U[0] += 0.5 * md->chargeA[a] * phi;
            U[1] += 0.5 * md->chargeA[a] * SFEL_COULOMB_EV_NM * qn / el->lr_rc;
        }
        if (PAR(cr))
        {
            gmx_sumd(2, U, cr);
        }
        el->lr_U   = U[0];
        el->lr_Ush = U[1];

        if (el->pbg_lr == 2 && MASTER(cr))
        {
            /* exact all-pairs bare field against long-range + the bare pairs
             * within rc (what GROMACS adds, exclusions aside), charged atoms */
            double num = 0, den = 0;

            for (a = 0; a < md->nr; a++)
            {
                double El[3], Ex[3] = {0, 0, 0}, En[3] = {0, 0, 0}, phi, qn, e2 = 0, x2 = 0;
                int    b;

                if (md->chargeA[a] == 0) { continue; }
                sfel_lr_field(el, md, x, a, El, &phi, &qn);
                for (b = 0; b < md->nr; b++)
                {
                    double dr[3], d2, r3;

                    if (b == a || md->chargeA[b] == 0) { continue; }
                    for (d = 0; d < 3; d++) { dr[d] = x[a][d] - x[b][d]; }
                    d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    r3 = d2 * sqrt(d2);
                    for (d = 0; d < 3; d++)
                    {
                        Ex[d] += SFEL_COULOMB_EV_NM * md->chargeA[b] * dr[d] / r3;
                        if (d2 < el->lr_rc * el->lr_rc)
                        {
                            En[d] += SFEL_COULOMB_EV_NM * md->chargeA[b] * dr[d] / r3;
                        }
                    }
                }
                for (d = 0; d < 3; d++)
                {
                    e2 += sqr(El[d] + En[d] - Ex[d]);
                    x2 += sqr(Ex[d]);
                }
                num += sqr(md->chargeA[a]) * e2;
                den += sqr(md->chargeA[a]) * x2;
                if (x2 > 0 && sqrt(e2 / x2) > cr_max) { cr_max = sqrt(e2 / x2); }
            }
            cr_rms = (den > 0) ? sqrt(num / den) : 0;
        }
        if (MASTER(cr))
        {
            if (el->flr == NULL)
            {
                el->flr = sfel_open_output("pbg_lr.txt", "w");
                fprintf(el->flr, "# t[ps] rc[nm] s=2dx[nm] erfc(rc/s) ions_outside U_lr[eV] W_lr[eV]  "
                        "check: rms_rel max_rel (|q|-weighted force error against all pairs; -1 off)  "
                        "U_shift[eV] (k q q / rc over pairs within rc: GROMACS's cutoff shift)\n");
            }
            fprintf(el->flr, "%.5f %.3f %.4f %.2e %d %.1f %.1f  %.3e %.3e  %.1f\n", t, el->lr_rc, sfel_pbg_s(el),
                    erfc(el->lr_rc / sfel_pbg_s(el)), el->lr_nout, el->lr_U, el->lr_W, cr_rms, cr_max,
                    el->lr_Ush);
            fflush(el->flr);
        }
        el->lr_ready = TRUE;
        el->lr_step  = el->screen_step;
    }
    for (a = md->start; a < md->start + md->homenr; a++)
    {
        const double q = md->chargeA[a];

        if (q == 0) { continue; }
        for (d = 0; d < 3; d++)
        {
            const double F = q * el->lr_E[3 * a + d] * SFEL_EV_TO_KJMOL;     /* kJ/mol/nm */

            f[a][d] += F;
            work    += F * v[a][d] * ir->delta_t;
        }
    }
    if (PAR(cr))
    {
        gmx_sumd(1, &work, cr);
    }
    el->lr_W += work / SFEL_EV_TO_KJMOL;
}


/* ---------------------------------------------------------------- *
 * Public entry points
 * ---------------------------------------------------------------- */

void sfelec_set_laser(t_sfelec *el, double E0_V_m)
{
    if (el != NULL)
    {
        el->E0 = E0_V_m * 1e-9;     /* V/nm, so that e E0 is in eV/nm */
    }
}

void sfelec_set_laser_now(t_sfelec *el, double E_V_m)
{
    if (el != NULL)
    {
        el->E_now = fabs(E_V_m) * 1e-9;
        el->E_sig = E_V_m * 1e-9;   /* signed, sfmd-laser-polarization */
    }
}

t_sfelec *sfelec_init(FILE *fplog, const t_inputrec *ir, const t_commrec *cr)
{
    t_sfelec   *el;
    const char *e;

    if (ir->sfmd_electrons == 0)
    {
        return NULL;
    }
    snew(el, 1);

    /* As mcionize_init with mcmd-electrons = pbgrid */
    el->screen_mode = 5;
    el->screen_evap = (ir->sfmd_evaporation != 0);
    el->pb_evh_ver  = -1;
    el->pb_nstep    = (int)floor(sfel_mdp_real(ir->sfmd_pb_interval) / (ir->delta_t * 1e3) + 0.5);
    if (el->pb_nstep < 1)
    {
        el->pb_nstep = 1;
    }
    /* sfmd-pb-long-range: the grid's ion field beyond rcoulomb, refreshed
     * every sfmd-pb-interval; GMX_SFMD_PBG_LR_CHECK=1 (diagnostic) checks it
     * against all pairs at each refresh (pbg_lr.txt) */
    e = getenv("GMX_SFMD_PBG_LR_CHECK");
    el->pbg_lr       = (ir->sfmd_pb_long_range == 0) ? 0 : (e != NULL && atoi(e) != 0) ? 2 : 1;
    el->pbg_lr_nstep = el->pb_nstep;
    el->lr_rc        = ir->rcoulomb;
    /* diagnostics (output only; the physics is the same with or without) */
    e = getenv("GMX_SFMD_TIMING_DETAIL");
    el->tdet = (e != NULL && atoi(e) != 0);
    e = getenv("GMX_SFMD_PB_EDIAG");
    el->pb_ediag = (e != NULL && atoi(e) != 0);
    e = getenv("GMX_SFMD_PBG_DENSITY");
    el->pbg_dens_out = (e != NULL && atoi(e) != 0);
    /* Grid: adaptive size and box, see the gromacs-mc manual */
    el->pbg_nmin    = 24;
    el->pbg_nmax    = 96;             /* placeholders until the limit is set */
    el->pbg_ncap    = el->pbg_nmax;
    el->pbg_n       = el->pbg_nmax;   /* until the first rebuild chooses */
    el->pbg_capset  = 0;
    el->pbg_nglimit = 8 * ((ir->sfmd_pb_grid_limit + 7) / 8);
    el->pbg_box     = sfel_mdp_real(ir->sfmd_pb_box_max);
    el->pbg_cut     = sfel_mdp_real(ir->sfmd_pb_box_cut);
    el->pbg_hbox    = (el->pbg_cut > 0) ? SFEL_PBG_BOXMIN : el->pbg_box;
    el->pbg_se      = sfel_mdp_real(ir->sfmd_pb_smoothing);
    el->pbg_dxt     = el->pbg_se / sfel_mdp_real(ir->sfmd_pb_cells_per_smoothing);
    el->pbg_ntol    = sfel_mdp_real(ir->sfmd_pb_newton_tol);
    /* the solve's temperature floor: 1 eV as in gromacs-mc;
     * GMX_SFMD_PB_TMIN (diagnostic) changes it, e.g. to test how much a cold
     * cloud's result depends on it */
    e = getenv("GMX_SFMD_PB_TMIN");
    el->pb_Tmin = (e != NULL && atof(e) > 0) ? atof(e) : 1.0;
    el->te_min      = sfel_mdp_real(ir->sfmd_te_min);
    el->te_floor_dt = ir->delta_t * 1e3;
    e = getenv("GMX_SFMD_PBG_OUTLAT_CHECK");
    el->pbg_outlat  = (e != NULL && atoi(e) != 0) ? 2 : 1;
    el->geom_heavy  = TRUE;
    el->laser_on    = (ir->sfmd_laser_barrier != 0);
    el->barrier_mode = ir->sfmd_laser_barrier;
    el->pol_axis    = (el->laser_on && ir->sfmd_laser_polarization > 0) ? ir->sfmd_laser_polarization - 1 : -1;
    el->heat_on     = (ir->sfmd_laser_heating != 0);
    el->heat_cap    = sfel_mdp_real(ir->sfmd_heating_nu_cap);
    el->heat_shape  = ir->sfmd_heating_shape;
    el->heat_surf   = sfel_mdp_real(ir->sfmd_heating_surface);
    el->coll_on     = (ir->sfmd_collisions != 0);
    el->rec_on      = ir->sfmd_coll_recombination;
    el->omega_L     = (ir->sfmd_pulse_wavelength > 0) ?
        2.0 * M_PI * 2.99792458e8 / (sfel_mdp_real(ir->sfmd_pulse_wavelength) * 1e-9) : 0.0;

    if (MASTER(cr) && fplog != NULL)
    {
        fprintf(fplog, "\nsfelectrons: trapped-electron cloud (sfmd-electrons = pbgrid), field-free: PB every %d "
                "steps (%g fs), s_e %g nm, %g cells per s_e, Newton tol %g, box %g..%g R (cut %g), grid limit %d, "
                "evaporation %s\n",
                el->pb_nstep, sfel_mdp_real(ir->sfmd_pb_interval), el->pbg_se,
                sfel_mdp_real(ir->sfmd_pb_cells_per_smoothing), el->pbg_ntol, SFEL_PBG_BOXMIN, el->pbg_box,
                el->pbg_cut, el->pbg_nglimit, el->screen_evap ? "on" : "off");
        if (el->te_min > 0)
        {
            fprintf(fplog, "sfelectrons: temperature floor %g eV (sfmd-te-min): PB solve, evaporation and "
                    "recombination at >= it, impact ionization at the actual T_e, E_kin not clamped\n", el->te_min);
        }
        else if (el->pb_Tmin != 1.0)
        {
            fprintf(fplog, "sfelectrons: NOTE temperature floor %g eV (GMX_SFMD_PB_TMIN), not 1 eV\n", el->pb_Tmin);
        }
        fprintf(fplog, "sfelectrons: laser heating %s", el->heat_on ? "on" : "off");
        if (el->heat_on)
        {
            fprintf(fplog, ": inverse bremsstrahlung of a Drude %s, nu_ei capped at %g omega_p%s",
                    el->heat_shape != 1 ? "sphere" :
                    el->pol_axis >= 0 ? "ellipsoid (the sample's axes, the field along the polarization)" :
                    "ellipsoid (the sample's axes, orientation-averaged)",
                    el->heat_cap, el->heat_cap > 0 ? "" : " (no cap)");
            if (el->heat_surf > 0)
            {
                fprintf(fplog, "; surface damping nu_s = %g vbar/R added", el->heat_surf);
            }
        }
        fprintf(fplog, "\n");
        if (el->coll_on)
        {
            fprintf(fplog, "sfelectrons: electron-impact ionization on (sfmd-collisions): Lotz, Maxwellian at the "
                    "cloud's T_e and the density at the atom, valence shell; three-body recombination %s\n",
                    el->rec_on == 1 ? "on (detailed balance, sfmd-coll-recombination)" :
                    el->rec_on == 2 ? "on after the pulse only (I < 1 % of peak, sfmd-coll-recombination = 2)" : "off");
        }
        fprintf(fplog, "sfelectrons: laser barrier %s%s\n", el->laser_on ? "on" : "off",
                !el->laser_on ? " (field-free cloud: the laser acts only through ADK)" :
                el->barrier_mode == 2 ?
                ": escape level lowered by 2 sqrt(k Q e E); spill and evaporation follow the instantaneous |E(t)|, "
                "escape at birth the envelope E0(t) (laser_log.txt: dW of E0)" :
                ": escape level lowered by 2 sqrt(k Q e E0(t)), cells above it spill (laser_log.txt)");
        if (el->pol_axis >= 0)
        {
            fprintf(fplog, "sfelectrons: laser polarization %c (sfmd-laser-polarization, DEVELOPMENT): the field E(t) "
                    "along %c is in the PB solve (sampled at each solve, %g per optical cycle), the escape level is the "
                    "saddle of net charge, dipole and field on the downhill side (replaces the direction-free barrier, "
                    "both sfmd-laser-barrier modes), the ions feel q E of the solve%s\n",
                    "xyz"[el->pol_axis], "xyz"[el->pol_axis],
                    (el->omega_L > 0) ? 2.0 * M_PI / el->omega_L * 1e15 / sfel_mdp_real(ir->sfmd_pb_interval) : 0.0,
                    (el->heat_on && el->heat_shape == 1) ? "; heating: the field along the axis on the ellipsoid" : "");
        }
        if (ir->rcoulomb > 0)
        {
            fprintf(fplog, "sfelectrons: NOTE rcoulomb = %g nm: the ion pairs beyond it feel nothing, while the "
                    "cloud acts at every distance\n", ir->rcoulomb);
        }
    }
    return el;
}

/* GMX_SFMD_PB_SEED="file N E_kin" (diagnostic, master): start from the charges
 * in file (charges.txt format) with N trapped electrons of kinetic energy
 * E_kin eV, the rest of the charge escaped.  For comparing the machinery with
 * gromacs-mc on a fixed cloud. */
static void sfel_seed(t_sfelec *el, t_mdatoms *md, const t_commrec *cr)
{
    const char *e = getenv("GMX_SFMD_PB_SEED");
    char        fn[256];
    double      N = 0, E = 0, q = 0;
    int         a;

    if (e == NULL)
    {
        return;
    }
    if (sscanf(e, "%255s %lf %lf", fn, &N, &E) != 3)
    {
        gmx_fatal(FARGS, "GMX_SFMD_PB_SEED must be \"file N E_kin\", not '%s'", e);
    }
    if (MASTER(cr))
    {
        FILE *fp = fopen(fn, "r");
        char  line[128];
        int   i, c;

        if (fp == NULL)
        {
            gmx_fatal(FARGS, "GMX_SFMD_PB_SEED: cannot open %s", fn);
        }
        while (fgets(line, sizeof(line), fp))
        {
            if (sscanf(line, "%d %d", &i, &c) == 2 && i >= 1 && i <= md->nr)
            {
                md->chargeA[i - 1] = c;
            }
        }
        fclose(fp);
        for (a = 0; a < md->nr; a++) { q += md->chargeA[a]; }
        el->num_free_electrons     = N;
        el->screen_E_trapped       = E;
        el->escaped_photoelectrons = q - N;
        fprintf(stderr, "\nsfelectrons: GMX_SFMD_PB_SEED: charges from %s (total %g), N %g, E_kin %g eV\n", fn, q, N, E);
    }
    if (PAR(cr))
    {
        gmx_bcast(md->nr * sizeof(md->chargeA[0]), md->chargeA, cr);
    }
}

void sfelec_step_begin(t_sfelec *el, const t_inputrec *ir, t_mdatoms *md,
                       rvec *x, double t, gmx_bool bFirstStep,
                       const t_commrec *cr, t_nrnb *nrnb)
{
    if (el == NULL)
    {
        return;
    }

    /* Under particle decomposition the integrator has only advanced this
     * rank's home atoms, so the rest of x is a step stale.  The escape depth,
     * the PB solve and the geometry need all of them: gather over the whole
     * ring (move_x() only runs pd_shift pulses, which in gromacs-mc left the
     * last block stale until the trajectory was written). */
    if (PAR(cr))
    {
        move_rvecs(cr, FALSE, FALSE, GMX_LEFT, GMX_RIGHT, x, NULL, cr->nnodes - 1, nrnb);
    }

    if (bFirstStep)
    {
        sfel_seed(el, md, cr);
    }
    /* sfmd-laser-polarization: without a cloud the field is this step's (with
     * one it is the last solve's, set there and broadcast) */
    if (el->pol_axis >= 0 && el->num_free_electrons <= 0)
    {
        el->pol_E = el->E_sig;
    }

    /* the order of mcionize_step: the ledger from the second step on, the
     * reference geometry on the first, the geometry update and evaporation
     * on every step (the first included) */
    if (MASTER(cr))
    {
        if (bFirstStep)
        {
            sfel_first_geometry(el, md, x);
        }
        else
        {
            sfel_ledger_start(el, md);
        }
        sfel_update_geometry(el, md, x);
        sfel_evaporate(el, ir);
        sfel_heat(el, ir, md);
    }
}

gmx_bool sfelec_birth(t_sfelec *el, const t_mdatoms *md, rvec *x, int k, double eps)
{
    double D_k, D_esc;

    /* the ion's charge change first, against the cloud as it is now (see
     * sfel_ledger_atom); none on the first step, before the ledger starts */
    if (el->led_q != NULL)
    {
        sfel_ledger_atom(el, md, x, k,
                         (el->pb_ready && el->pb_N > 0) ? el->num_free_electrons / el->pb_N : 0.0);
    }
    /* the escape test against the laser-lowered level; a trapped electron's
     * energy at its true depth */
    D_esc = sfel_escape_depth(el, md, x, k, &D_k);

    /* An electron at rest (eps = 0) escapes from a well of depth <= 0, which
     * mc-md's strict E > D_k would not let it do: with ADK every electron
     * has eps = 0, and the first one of a neutral sample (D = 0) must go. */
    if (eps >= D_esc)
    {
        el->escaped_photoelectrons += 1;
        el->count_escaped_birth    += 1;
        return TRUE;
    }
    else
    {
        const double dep = (eps > 0 ? eps : 0);

        el->num_free_electrons += 1;
        if (el->pol_axis >= 0) { el->pol_X += x[k][el->pol_axis]; }
        el->pb_Etot            += dep - D_k;
        el->pbd_N              -= D_k;
        el->screen_E_trapped   += dep;
        el->eb_dep             += dep;
        el->count_trapped      += 1;
        return FALSE;
    }
}

/* ---------------------------------------------------------------- *
 * sfmd-collisions: electron-impact ionization by the cloud (mc-md's
 * collisional channel with mcmd-coll-data = lotz, ionization only).  The
 * rate itself is computed in sfionize.c, which has the ionization energies.
 * ---------------------------------------------------------------- */
double sfelec_coll_Te(const t_sfelec *el)
{
    if (el == NULL || !el->coll_on || !el->pb_ready || el->pb_N <= 0 || el->num_free_electrons < 1.0)
    {
        return 0.0;
    }
    return sfel_plasma_Te(el);
}

double sfelec_rec_Te(const t_sfelec *el)
{
    /* the same conditions as impact ionization, but at max(T_e, sfmd-te-min):
     * the rate grows as n_e^2 T^-3 and is capped at its value at the floor */
    if (el == NULL || !el->coll_on || !el->pb_ready || el->pb_N <= 0 || el->num_free_electrons < 1.0)
    {
        return 0.0;
    }
    return sfel_model_Te(el);
}

double sfelec_coll_ne(const t_sfelec *el, int k)
{
    /* the density at the atom from the last solve (nm^-3, for pb_N electrons),
     * scaled to the electrons now, in cm^-3 */
    if (el == NULL || !el->pb_ready || el->pb_N <= 0 || el->pb_ne == NULL)
    {
        return 0.0;
    }
    return el->pb_ne[k] * (el->num_free_electrons / el->pb_N) * 1e21;
}

gmx_bool sfelec_coll_event(t_sfelec *el, const t_mdatoms *md, rvec *x, int k, double P)
{
    double D_k;

    /* the cloud cannot pay more than it has: no event then (the rate is
     * vanishingly small there anyway) */
    if (el->screen_E_trapped < P)
    {
        return FALSE;
    }
    /* the ion's charge change against the cloud, as for a birth */
    if (el->led_q != NULL)
    {
        sfel_ledger_atom(el, md, x, k,
                         (el->pb_ready && el->pb_N > 0) ? el->num_free_electrons / el->pb_N : 0.0);
    }
    (void)sfel_escape_depth(el, md, x, k, &D_k);

    /* a cloud electron gives up P and frees one more, which stays in the cloud
     * at rest at its atom (potential energy -D_k) */
    el->num_free_electrons += 1;
    if (el->pol_axis >= 0) { el->pol_X += x[k][el->pol_axis]; }
    el->pb_Etot            -= P + D_k;
    el->pbd_N              -= D_k;
    el->screen_E_trapped   -= P;
    el->eb_coll            += P;
    el->count_coll         += 1;
    return TRUE;
}

/* sfmd-schedule: the cloud's step-counted intervals for the stage that
 * starts at MD step step0, time t0 (ps), with ir->delta_t already the
 * stage's: the PB solve every pb_fs; the long-range ion field keeps the tpr's
 * sfmd-pb-interval in time (gromacs-mc 6e81e9f: it is the bare push from
 * beyond rcoulomb, which falls as the sample expands; held for the schedule's
 * long PB intervals it over-drove the explosion, KE +17 to +50 % at 11k
 * atoms).  bChange: the stage starts now, so the steps since the last solve
 * and the last refresh are converted from the old step dt_prev (ps). */
void sfelec_schedule(t_sfelec *el, const t_inputrec *ir, double pb_fs, gmx_large_int_t step0, double t0,
                     gmx_bool bChange, double dt_prev)
{
    const double dt_fs = ir->delta_t * 1e3;
    int          n;

    if (el == NULL)
    {
        return;
    }
    el->sched_on    = TRUE;
    el->sched_step0 = step0;
    el->sched_t0    = t0;
    n = (int)floor(pb_fs / dt_fs + 0.5);
    n = (n < 1) ? 1 : n;
    if (bChange)
    {
        const double rr = dt_prev / ir->delta_t;

        el->pb_step_solve = el->screen_step - (gmx_large_int_t)floor((el->screen_step - el->pb_step_solve) * rr + 0.5);
        el->lr_step       = el->screen_step - (gmx_large_int_t)floor((el->screen_step - el->lr_step) * rr + 0.5);
    }
    el->pb_nstep = n;
    n = (int)floor(sfel_mdp_real(ir->sfmd_pb_interval) / dt_fs + 0.5);
    el->pbg_lr_nstep = (n < 1) ? 1 : n;
    el->te_floor_dt  = dt_fs;
    if (el->pbg_lr)
    {
        sprintf(el->sched_str, "PB every %d steps (%g fs), long-range ion field every %d steps,", el->pb_nstep,
                el->pb_nstep * dt_fs, el->pbg_lr_nstep);
    }
    else
    {
        sprintf(el->sched_str, "PB every %d steps (%g fs),", el->pb_nstep, el->pb_nstep * dt_fs);
    }
}

const char *sfelec_schedule_str(const t_sfelec *el)
{
    return (el != NULL) ? el->sched_str : "";
}

void sfelec_counts(const t_sfelec *el, double *n_coll, double *n_rec)
{
    *n_coll = (el != NULL) ? el->count_coll : 0;
    *n_rec  = (el != NULL) ? el->count_rec : 0;
}

int sfelec_rec_on(const t_sfelec *el)
{
    return (el != NULL) ? el->rec_on : 0;
}

gmx_bool sfelec_rec_event(t_sfelec *el, const t_mdatoms *md, rvec *x, int k, double P)
{
    double D_k;

    /* never capture an electron the cloud no longer has */
    if (el->num_free_electrons < 1.0)
    {
        return FALSE;
    }
    if (el->led_q != NULL)
    {
        sfel_ledger_atom(el, md, x, k,
                         (el->pb_ready && el->pb_N > 0) ? el->num_free_electrons / el->pb_N : 0.0);
    }
    (void)sfel_escape_depth(el, md, x, k, &D_k);

    /* one cloud electron is bound at its atom (it leaves the cloud from
     * potential energy -D_k, at rest), the binding energy P goes to a third */
    el->num_free_electrons -= 1;
    if (el->pol_axis >= 0) { el->pol_X -= x[k][el->pol_axis]; }
    el->pb_Etot            += P + D_k;
    el->pbd_N              += D_k;
    el->screen_E_trapped   += P;
    el->eb_rec             += P;
    el->count_rec          += 1;
    return TRUE;
}

void sfelec_step_end(t_sfelec *el, const t_mdatoms *md, rvec *x, const t_commrec *cr)
{
    if (el == NULL)
    {
        return;
    }
    if (MASTER(cr))
    {
        sfel_ledger_step(el, md, x);
    }
    /* the trapped-electron state lives on the master, but every rank
     * corrects the forces on its own home atoms */
    if (PAR(cr))
    {
        sfel_bcast_plasma(el, cr);
    }
}

/* ---------------------------------------------------------------- *
 * The forces of the trapped electrons on the ions (mcionize_screen_forces).
 *
 * The MD sees bare ion charges, as if every emitted electron had left.  Here
 * the forces do_force() just produced get the field of the trapped electron
 * cloud of the last PB solve (sfel_pbg_forces).  Also writes the energy
 * ledger and screening_log.txt.
 * ---------------------------------------------------------------- */
void sfelec_forces(t_sfelec *el, const t_inputrec *ir, t_mdatoms *mdatoms,
                   rvec *x, rvec *v, rvec *f, const t_commrec *cr)
{
    const double N_tr    = (el != NULL) ? el->num_free_electrons : 0;
    double       f_trap  = 0;
    int          applied = 0;
    double       tm_sf;

    if (el == NULL)
    {
        return;
    }
    el->screen_step++;
    tm_sf = sfel_pb_wallms();

    if (N_tr > 0)
    {
        /* lambda_D of the mean density, for the log only */
        double R  = el->radius_now, Te = sfel_plasma_Te(el);
        double ne = N_tr / (4.0 / 3.0 * M_PI * R * R * R);
        double lambda;

        f_trap               = N_tr / (N_tr + el->escaped_photoelectrons);
        lambda               = (Te > 0) ? 0.23508 * sqrt(Te / ne) : 0;
        el->screen_ND        = ne * 4.0 / 3.0 * M_PI * lambda * lambda * lambda;
        el->screen_lambda_nm = lambda;
        applied              = 1;
    }
    sfel_pbg_forces(el, ir, mdatoms, x, v, f, cr);
    if (el->pol_axis >= 0 && el->pol_E != 0)
    {
        /* sfmd-laser-polarization: the field on the ions, the one the cloud was
         * solved in, so that ions and cloud together feel Q e E */
        const int    ax = el->pol_axis;
        const double fE = el->pol_E * SFEL_EV_TO_KJMOL;
        int          a;

        for (a = mdatoms->start; a < mdatoms->start + mdatoms->homenr; a++)
        {
            f[a][ax] += mdatoms->chargeA[a] * fE;
        }
    }
    if (el->pbg_lr)
    {
        const double tm_lr = sfel_pb_wallms();

        sfel_pbg_lr(el, ir, mdatoms, x, v, f, cr);
        el->tm[6] += sfel_pb_wallms() - tm_lr;
    }
    el->pressure_h_now      = el->pb_h;
    el->tm[SFEL_TM_SCREEN] += sfel_pb_wallms() - tm_sf;

    /* energy ledger, every 100 steps */
    if (MASTER(cr) && ((el->screen_step - 1) % 100) == 0)
    {
        if (el->fled == NULL)
        {
            el->fled = sfel_open_output("energy_ledger.txt", "w");
            fprintf(el->fled, "# step t[ps] J_ii[eV] J_ie[eV] W[eV] E_kin_e[eV] N_trapped charge_changes "
                    "(cumulative: ion-ion and ion-electron energy jumps of charge changes, electron-field work on "
                    "the ions)\n");
        }
        fprintf(el->fled, "%ld %.5f %.3f %.3f %.3f %.3f %.1f %.0f\n", (long)(el->screen_step - 1),
                sfel_time(el, ir, el->screen_step - 1), el->led_Jii, el->led_Jie, el->pressure_work,
                el->screen_E_trapped, el->num_free_electrons, el->led_nchg);
        fflush(el->fled);
    }

    if (MASTER(cr) && (el->screen_step % 100) == 1)
    {
        if (el->fscreen == NULL)
        {
            el->fscreen = sfel_open_output("screening_log.txt", "w");
            fprintf(el->fscreen, "# step escaped trapped trapped_frac mean_E_trapped[eV] "
                    "R[nm] lambda_D[nm] N_Debye applied evaporated T_e[eV] pressure_work[eV] "
                    "pressure_h[nm]\n");
        }
        fprintf(el->fscreen, "%ld %.0f %.0f %.4f %.2f %.4f %.5f %.4g %d %.0f %.2f %.1f %.4f\n",
                (long)el->screen_step, el->escaped_photoelectrons, N_tr, f_trap,
                N_tr > 0 ? el->screen_E_trapped / N_tr : 0.0, el->radius_now,
                el->screen_lambda_nm, el->screen_ND, applied, el->screen_evaporated,
                sfel_plasma_Te(el), el->pressure_work, el->pressure_h_now);
    }
}

void sfelec_done(t_sfelec *el, FILE *fplog, const t_commrec *cr)
{
    FILE *fps[10];
    int   i;

    if (el == NULL)
    {
        return;
    }
    if (MASTER(cr))
    {
        FILE *out[2] = {stderr, fplog};

        for (i = 0; i < 2; i++)
        {
            if (out[i] == NULL)
            {
                continue;
            }
            fprintf(out[i], "\nsfelectrons: escaped at birth %.0f, trapped %.0f, evaporated %.0f, spilled %.0f; "
                    "laser heating %.1f eV; at the end N %.1f, Q %.1f, T_e %.2f eV, E_kin %.1f eV, %d PB solves (%.0f s), %d grid rebuilds; "
                    "ledger J_ii %.1f, J_ie %.1f, W %.1f eV, E_clamped %.1f eV\n",
                    el->count_escaped_birth, el->count_trapped, el->screen_evaporated, el->count_spilled,
                    el->eb_laser, el->num_free_electrons,
                    el->escaped_photoelectrons, sfel_plasma_Te(el), el->screen_E_trapped, el->pb_nsolve,
                    1e-3 * el->tm[SFEL_TM_SOLVE], el->pbg_nregrid, el->led_Jii, el->led_Jie, el->pressure_work,
                    el->pb_Eclamp);
            if (el->coll_on)
            {
                fprintf(out[i], "sfelectrons: electron-impact ionizations %.0f, ionization energy paid by the "
                        "cloud %.1f eV\n", el->count_coll, el->eb_coll);
                if (el->rec_on)
                {
                    fprintf(out[i], "sfelectrons: three-body recombinations %.0f, binding energy given to the cloud "
                            "%.1f eV\n", el->count_rec, el->eb_rec);
                }
            }
            if (el->te_floor_steps > 0)
            {
                fprintf(out[i], "sfelectrons: %.0f steps (%.2f fs) with the actual T_e below sfmd-te-min = %g eV "
                        "(solve, evaporation and recombination at the floor)\n", el->te_floor_steps,
                        el->te_floor_steps * el->te_floor_dt, el->te_min);
            }
            if (el->pbg_lr)
            {
                fprintf(out[i], "sfelectrons: long-range ion field beyond rcoulomb = %g nm from the grid: work %.1f eV, "
                        "%.1f s (pbg_lr.txt)\n", el->lr_rc, el->lr_W, 1e-3 * el->tm[6]);
            }
            if (el->pol_axis >= 0)
            {
                fprintf(out[i], "sfelectrons: laser polarization %c: polarization work of the field on the cloud "
                        "%.1f eV (the stored part returns after the pulse; electrons born into and spilled from the shifted "
                        "cloud leave a net gain), %.0f solves with the level at the sample surface (no saddle outside it)\n",
                        "xyz"[el->pol_axis], el->eb_pol, el->pol_nosaddle);
            }
        }
    }
    fps[0] = el->fscreen; fps[1] = el->fpbd; fps[2] = el->fpb; fps[3] = el->fpbw; fps[4] = el->ftd;
    fps[5] = el->fled; fps[6] = el->fbox; fps[7] = el->fol; fps[8] = el->fop; fps[9] = el->flr;
    if (el->flaser != NULL) { fclose(el->flaser); }
    for (i = 0; i < 10; i++)
    {
        if (fps[i] != NULL) { fclose(fps[i]); }
    }
    if (el->pbg_ne != NULL)
    {
        if (el->pbg_nplan > 0)
        {
            fftwf_destroy_plan(el->pbg_fwd);
            fftwf_destroy_plan(el->pbg_bwd);
            fftwf_destroy_plan(el->pbg_fwdE);
            fftwf_destroy_plan(el->pbg_bwdE);
        }
        sfree(el->pbg_ne); sfree(el->pbg_Phi); sfree(el->pbg_V); sfree(el->pbg_g); sfree(el->pbg_d);
        sfree(el->pbg_t); sfree(el->pbg_Kn); sfree(el->pbg_nt); sfree(el->pbg_Vt); sfree(el->pbg_cg);
        fftwf_free(el->pbg_buf); fftwf_free(el->pbg_hat);
        sfree(el->pbg_KhE); sfree(el->pbg_Kh);
    }
    sfree(el->pbg_UL); sfree(el->pb_evhX);
    sfree(el->pbg_PhiL); sfree(el->pbg_PhiLf); sfree(el->lr_E); sfree(el->lr_head); sfree(el->lr_next);
    sfree(el->pbg_Ve); sfree(el->pb_ne); sfree(el->pbd_q); sfree(el->pbg_nprev); sfree(el->led_q);
    sfree(el->pb_evlist); sfree(el->pb_evhw); sfree(el->pb_evhV); sfree(el->pb_evhVt); sfree(el->pb_evhK);
    sfree(el);
}
