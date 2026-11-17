/*
 * sfionize.c
 *
 * The MolDStruct strong-field ionization module.  Extracted from md.c, where
 * it had grown to ~1600 lines inline; md.c now holds only the four calls in
 * sfionize.h.
 *
 * ADK tunnelling ionization driven by a Gaussian laser pulse: the field is
 * evaluated per step, each atom's current charge state gives an ionization
 * potential and a set of quantum numbers, and the resulting DC tunnelling
 * rate is turned into a per-step probability tested against the module's own
 * random stream.
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
#include <sys/types.h>

#include "typedefs.h"
#include "smalloc.h"
#include "vec.h"
#include "network.h"
#include "mdrun.h"
#include "partdec.h"
#include "gmx_random.h"
#include "gmx_fatal.h"
#include "main.h"
#include "sfionize.h"
#include "sfelectrons.h"

/* Defined in gmxlib/bondfree.c, where it is read.  Raised by sfionize_init()
 * so the altered force field switches on with the ionization. */
extern int moldstruct_altered_ff;

//////////////////////////////////////////////////////////////////////////

// (approximative) Masses, used as atomic species identifiers
// If you want more atomic species you need to update this mass-list, the "Element list"
#define MASS_H 1
#define MASS_C 12
#define MASS_N 14
#define MASS_O 16
#define MASS_F 19
#define MASS_NA 23
#define MASS_MG 24
#define MASS_SI 28
#define MASS_P 31
#define MASS_S 32
#define MASS_CL 35
#define MASS_CA 40
#define MASS_FE 56
#define MASS_NI 59
#define MASS_I 127

/* Deuterium.  Not an element of its own here: it is chemically hydrogen and
 * has hydrogen's electronic structure, so it ionizes with hydrogen's ADK
 * parameters.  See species_mass() below. */
#define MASS_D 2

typedef struct
{
    int mass;
    int index;
    char symbol[3];
} Element;

#define NUM_ELEMENTS 15
// The digit is simply an index to uniquely identify elements
Element elements[NUM_ELEMENTS] = {
    {MASS_H, 0, "H"},
    {MASS_C, 1, "C"},
    {MASS_N, 2, "N"},
    {MASS_O, 3, "O"},
    {MASS_F, 4, "F"},
    {MASS_MG, 5, "MG"},
    {MASS_P, 6, "P"},
    {MASS_S, 7, "S"},
    {MASS_CL, 8, "CL"},
    {MASS_CA, 9, "CA"},
    {MASS_FE, 10, "FE"},
    {MASS_NI, 11, "NI"},
    {MASS_SI, 12, "SI"},
    {MASS_NA, 13, "NA"},
    {MASS_I, 14, "I"}};

int mass2idx(int mass)
{
    int i;
    for (i = 0; i < NUM_ELEMENTS; i++)
    {
        if (elements[i].mass == mass)
        {
            return elements[i].index;
        }
    }
    return -1;
}

int idx2mass(int idx)
{
    if (idx >= 0 && idx < NUM_ELEMENTS)
    {
        return elements[idx].mass;
    }
    return -1;
}

char *mass2char(int mass)
{
    int i;
    for (i = 0; i < NUM_ELEMENTS; i++)
    {
        if (elements[i].mass == mass)
        {
            return elements[i].symbol;
        }
    }
    return "_";
}

/* Species identifier for an atom of mass m [amu].  Everything downstream --
 * the ADK parameter tables, the ionization potentials, mass2char() -- keys off
 * this rather than off the raw mass, so this is the one place isotopes are
 * folded onto their element.
 *
 * Deuterium is mapped onto hydrogen.  Only the species identity is remapped;
 * mdatoms->massT keeps the real mass 2, so the dynamics are untouched and a
 * deuteron still recoils at its own mass.  Without this a deuterium atom
 * matches no entry in the ADK table and try_adk_ionization_precomputed()
 * silently never ionizes it. */
static int species_mass(real m)
{
    int mass = (int)round(m);

    return (mass == MASS_D) ? MASS_H : mass;
}

//////////////////////////////////////////////////////////////////////////////////////
////////////                                                            //////////////
////////////                Here lives structs we need                  //////////////
////////////                                                            //////////////
//////////////////////////////////////////////////////////////////////////////////////
// Here we can put structures and functions we need

#define SPEED_OF_LIGHT_M_S 2.99792458e8
#define EPS0_SI 8.8541878128e-12
#define ELEMENTARY_CHARGE_C 1.602176634e-19
#define H_PLANCK_J_S 6.62607015e-34
#define HBAR_J_S 1.054571817e-34

#define PI_CONST_LOCAL 3.14159265358979323846
#define PS_TO_S 1.0e-12
/* The mdp gives pulse times in femtoseconds; the module works in ps. */
#define FS_TO_PS 1e-3
/* FWHM = 2 sqrt(2 ln 2) sigma, for converting sfmd-pulse-fwhm. */
#define SFMD_FWHM_PER_SIGMA 2.3548200450309493
#define NM_TO_M 1.0e-9
#define W_M2_TO_W_CM2 1.0e-4

/* Atomic units */
#define AU_FIELD_V_M 5.14220674763e11
#define AU_TIME_S 2.4188843265857e-17
#define AU_ENERGY_EV 27.211386245988
#define AU_INTENSITY_W_CM2 3.50944506e16

typedef struct
{
    /* User inputs */
    double t0_ps;            /* Pulse peak time [ps] */
    double pulse_energy_J;   /* Total pulse energy [J] */
    double sigma_ps;         /* Gaussian intensity sigma [ps] */
    double spot_diameter_nm; /* Focal spot diameter [nm] */
    double wavelength_nm;    /* Wavelength [nm] */

    /* Derived geometry/time */
    double t0_s;            /* Pulse peak time [s] */
    double sigma_s;         /* Gaussian intensity sigma [s] */
    double spot_diameter_m; /* Focal spot diameter [m] */
    double spot_radius_m;   /* Focal spot radius [m] */
    double spot_area_m2;    /* Focal area [m^2] */
    double wavelength_m;    /* Wavelength [m] */

    /* Derived pulse/field quantities */
    double peak_intensity_W_m2;
    double peak_intensity_W_cm2;
    double E0_V_m;
    double omega_rad_s;
    double period_s;
    double frequency_Hz;

    /* Carrier phase, randomized per run in [-pi/2, pi/2] */
    double phase_rad;

    /* Photon quantities */
    double photon_energy_J;
    double photon_energy_eV;

} LaserPulse;

static LaserPulse init_laser_pulse(const t_inputrec *ir, gmx_rng_t rng)
{
    LaserPulse p;

    /* Read user inputs.  The mdp gives the pulse times in femtoseconds and
     * the width as the FWHM; everything below works in picoseconds and the
     * standard deviation, so convert once here. */
    p.t0_ps = (double)ir->sfmd_pulse_peak_time * FS_TO_PS;
    p.pulse_energy_J = (double)ir->sfmd_pulse_energy;
    p.sigma_ps = ((double)ir->sfmd_pulse_fwhm * FS_TO_PS) / SFMD_FWHM_PER_SIGMA;
    p.spot_diameter_nm = (double)ir->sfmd_pulse_focal_diameter;
    p.wavelength_nm = (double)ir->sfmd_pulse_wavelength;

    /* Unit conversions */
    p.t0_s = p.t0_ps * PS_TO_S;
    p.sigma_s = p.sigma_ps * PS_TO_S;
    p.spot_diameter_m = p.spot_diameter_nm * NM_TO_M;
    p.spot_radius_m = 0.5 * p.spot_diameter_m;
    p.spot_area_m2 = PI_CONST_LOCAL * p.spot_radius_m * p.spot_radius_m;
    p.wavelength_m = p.wavelength_nm * NM_TO_M;

    /* Peak intensity from total pulse energy:
       E_pulse = A * integral I(t) dt
               = A * I0 * sigma * sqrt(2*pi)
    */
    if (p.pulse_energy_J > 0.0 && p.spot_area_m2 > 0.0 && p.sigma_s > 0.0)
    {
        p.peak_intensity_W_m2 =
            p.pulse_energy_J /
            (p.spot_area_m2 * p.sigma_s * sqrt(2.0 * PI_CONST_LOCAL));
    }
    else
    {
        p.peak_intensity_W_m2 = 0.0;
    }

    p.peak_intensity_W_cm2 = p.peak_intensity_W_m2 * W_M2_TO_W_CM2;

    /* I = 0.5 * c * eps0 * E0^2 */
    if (p.peak_intensity_W_m2 > 0.0)
    {
        p.E0_V_m = sqrt(2.0 * p.peak_intensity_W_m2 /
                        (SPEED_OF_LIGHT_M_S * EPS0_SI));
    }
    else
    {
        p.E0_V_m = 0.0;
    }

    if (p.wavelength_m > 0.0)
    {
        p.frequency_Hz = SPEED_OF_LIGHT_M_S / p.wavelength_m;
        p.omega_rad_s = 2.0 * PI_CONST_LOCAL * p.frequency_Hz;
        p.period_s = 1.0 / p.frequency_Hz;

        p.photon_energy_J = H_PLANCK_J_S * SPEED_OF_LIGHT_M_S / p.wavelength_m;
        p.photon_energy_eV = p.photon_energy_J / ELEMENTARY_CHARGE_C;
    }
    else
    {
        p.frequency_Hz = 0.0;
        p.omega_rad_s = 0.0;
        p.period_s = 0.0;
        p.photon_energy_J = 0.0;
        p.photon_energy_eV = 0.0;
    }

    /* Random carrier phase, uniform in [-pi/2, pi/2].
     * Every rank draws its own value here; the caller broadcasts
     * laser.phase_rad from MASTER afterwards so all ranks agree on the
     * one physical field, regardless of what each rank's own RNG drew. */
    p.phase_rad = (gmx_rng_uniform_real(rng) - 0.5) * PI_CONST_LOCAL;

    return p;
}

/* Time-dependent laser quantities */
static double laser_intensity_W_m2(const LaserPulse *p, double t_s)
{
    double arg;

    if (p->sigma_s <= 0.0)
    {
        return 0.0;
    }

    arg = (t_s - p->t0_s) / p->sigma_s;

    return p->peak_intensity_W_m2 * exp(-0.5 * arg * arg);
}

static double laser_intensity_W_cm2(const LaserPulse *p, double t_s)
{
    return laser_intensity_W_m2(p, t_s) * W_M2_TO_W_CM2;
}

static double laser_field_amplitude_V_m(const LaserPulse *p, double t_s)
{
    double arg;

    if (p->sigma_s <= 0.0)
    {
        return 0.0;
    }

    arg = (t_s - p->t0_s) / p->sigma_s;

    /* Field envelope because I(t) proportional to E(t)^2 */
    return p->E0_V_m * exp(-0.25 * arg * arg);
}

static double laser_field_V_m(const LaserPulse *p, double t_s)
{
    if (p->omega_rad_s <= 0.0)
    {
        return 0.0;
    }

    return laser_field_amplitude_V_m(p, t_s) *
           cos(p->omega_rad_s * (t_s - p->t0_s) + p->phase_rad);
}

/* Ionization model constants */
#define MAX_CHARGE_STATES 8
#define MAX_SHELLS 3

typedef struct
{
    int mass;
    char symbol[3];

    /* Number of supported charge states.
       ionization_potential_eV[q] is the energy needed for q -> q+1.
       Example: q = 0 means neutral -> +1.
    */
    int n_ip;
    double ionization_potential_eV[MAX_CHARGE_STATES];

} ElementIonizationData;

static ElementIonizationData ion_data[] = {
    {MASS_H, "H", 1, {13.598}},
    {MASS_C, "C", 4, {11.260, 24.383, 47.887, 64.494}},
    {MASS_N, "N", 5, {14.534, 29.601, 47.448, 77.473, 97.890}},
    {MASS_O, "O", 6, {13.618, 35.117, 54.936, 77.413, 113.899, 138.119}},
    {MASS_S, "S", 6, {10.360, 23.330, 34.830, 47.300, 72.600, 88.000}}};

#define NUM_ION_DATA ((int)(sizeof(ion_data) / sizeof(ion_data[0])))

static ElementIonizationData *get_ion_data_from_mass(int mass)
{
    int k;

    for (k = 0; k < NUM_ION_DATA; k++)
    {
        if (ion_data[k].mass == mass)
        {
            return &ion_data[k];
        }
    }

    return NULL;
}

static double get_ionization_potential_eV(int mass, int charge)
{
    ElementIonizationData *d;

    d = get_ion_data_from_mass(mass);

    if (d == NULL)
    {
        return 0.0;
    }

    if (charge < 0 || charge >= d->n_ip)
    {
        return 0.0;
    }

    return d->ionization_potential_eV[charge];
}

static double field_V_m_to_au(double E_V_m)
{
    return fabs(E_V_m) / AU_FIELD_V_M;
}

static double energy_eV_to_au(double E_eV)
{
    return E_eV / AU_ENERGY_EV;
}

static double factorial_int(int n)
{
    int i;
    double out = 1.0;

    if (n < 0)
    {
        return 0.0;
    }

    for (i = 2; i <= n; i++)
    {
        out *= (double)i;
    }

    return out;
}

typedef struct
{
    int n; /* principal quantum number of active electron */
    int l; /* orbital angular momentum: s=0, p=1, d=2 */
    int m; /* magnetic quantum number; default use m=0 */
} ADKQuantumNumbers;

typedef struct
{
    double rate_s_inv;
    double probability;
    double random_value;
    int ionized;
} IonizationTrial;

/*
 * Approximate atomic shell model for ADK ionization.
 *
 * The quantum numbers identify the orbital from which the next electron is
 * assumed to be removed. This is not a full electronic-structure calculation.
 * We use neutral-like shell ordering and charge-state-dependent ionization
 * potentials. We remove electrons based on a simplified Hund-like order.
 *
 * H:  1s
 * C:  2p^2 then 2s^2
 * N:  2p^3 then 2s^2
 * O:  2p^4 then 2s^2
 * S:  3p^4 then 3s^2
 *
 */
static ADKQuantumNumbers get_adk_quantum_numbers(int mass, int charge)
{
    ADKQuantumNumbers q;

    /* Default fallback */
    q.n = 1;
    q.l = 0;
    q.m = 0;

    switch (mass)
    {
    case MASS_H:
        /*
         * H: 1s1
         */
        q.n = 1;
        q.l = 0;
        q.m = 0;
        break;

    case MASS_C:
        /*
         * C: 2s2 2p2
         *
         *   charge 0: remove 2p, m = 0
         *   charge 1: remove 2p, |m| = 1
         *   charge >= 2: remove 2s, m = 0
         */
        q.n = 2;

        if (charge < 2)
        {
            q.l = 1; /* 2p */
            q.m = (charge == 0) ? 0 : 1;
        }
        else
        {
            q.l = 0; /* 2s */
            q.m = 0;
        }
        break;

    case MASS_N:
        /*
         * N: 2s2 2p3
         *
         *   charge 0: remove 2p, m = 0
         *   charge 1-2: remove 2p, |m| = 1
         *   charge >= 3: remove 2s, m = 0
         */
        q.n = 2;

        if (charge < 3)
        {
            q.l = 1; /* 2p */
            q.m = (charge == 0) ? 0 : 1;
        }
        else
        {
            q.l = 0; /* 2s */
            q.m = 0;
        }
        break;

    case MASS_O:
        /*
         * O: 2s2 2p4
         *
         *   charge 0: remove 2p, m = 0
         *   charge 1-3: remove 2p, |m| = 1
         *   charge >= 4: remove 2s, m = 0
         */
        q.n = 2;

        if (charge < 4)
        {
            q.l = 1; /* 2p */
            q.m = (charge == 0) ? 0 : 1;
        }
        else
        {
            q.l = 0; /* 2s */
            q.m = 0;
        }
        break;

    case MASS_P:
        /*
         * P: 3s2 3p3
         *
         *   charge 0: remove 3p, m = 0
         *   charge 1-2: remove 3p, |m| = 1
         *   charge >= 3: remove 3s, m = 0
         */
        q.n = 3;

        if (charge < 3)
        {
            q.l = 1; /* 3p */
            q.m = (charge == 0) ? 0 : 1;
        }
        else
        {
            q.l = 0; /* 3s */
            q.m = 0;
        }
        break;

    case MASS_S:
        /*
         * S: 3s2 3p4
         *
         *   charge 0: remove 3p, m = 0
         *   charge 1-3: remove 3p, |m| = 1
         *   charge >= 4: remove 3s, m = 0
         */
        q.n = 3;

        if (charge < 4)
        {
            q.l = 1; /* 3p */
            q.m = (charge == 0) ? 0 : 1;
        }
        else
        {
            q.l = 0; /* 3s */
            q.m = 0;
        }
        break;

    default:
        /*
         * Unsupported element fallback.
         */
        q.n = 1;
        q.l = 0;
        q.m = 0;
        break;
    }

    return q;
}
/*
 * Current supported ADK transitions:
 * H: 1, C: 4, N: 5, O: 6, S: 6 -> total 22 states.
 * Keep extra room for future species.
 */
#define MAX_ADK_PRECOMPUTED_STATES 32

typedef struct
{
    int valid;

    int mass;
    int charge;

    int n;
    int l;
    int m_abs;

    double Ip_eV;
    double Ip_au;
    double Z;
    double nstar;

    /*
     * Precomputed ADK constants.
     *
     * Runtime form:
     *
     * W_au = prefactor
     *        * pow(base / E_au, power)
     *        * exp(-exp_coeff / E_au)
     *
     * W_s^-1 = W_au / AU_TIME_S
     */
    double prefactor;
    double base;
    double power;
    double exp_coeff;

} ADKPrecomputedState;

typedef struct
{
    int nstates;
    ADKPrecomputedState states[MAX_ADK_PRECOMPUTED_STATES];
} ADKPrecomputedTable;

static ADKPrecomputedState build_adk_precomputed_state(int mass, int charge)
{
    ADKPrecomputedState s;
    ADKQuantumNumbers q;

    double Cnl2;
    double prefactor_lm;
    double two_Ip_32;

    s.valid = 0;

    s.mass = mass;
    s.charge = charge;

    s.n = 1;
    s.l = 0;
    s.m_abs = 0;

    s.Ip_eV = 0.0;
    s.Ip_au = 0.0;
    s.Z = 0.0;
    s.nstar = 0.0;

    s.prefactor = 0.0;
    s.base = 0.0;
    s.power = 0.0;
    s.exp_coeff = 0.0;

    s.Ip_eV = get_ionization_potential_eV(mass, charge);

    if (s.Ip_eV <= 0.0)
    {
        return s;
    }

    s.Ip_au = energy_eV_to_au(s.Ip_eV);

    if (s.Ip_au <= 0.0)
    {
        return s;
    }

    /*
     * Residual ion charge after ionization:
     * neutral -> +1 uses Z = 1, +1 -> +2 uses Z = 2, etc.
     */
    s.Z = (double)(charge + 1);

    /*
     * Effective principal quantum number:
     * n* = Z / sqrt(2 Ip), with Ip in atomic units.
     */
    s.nstar = s.Z / sqrt(2.0 * s.Ip_au);

    if (s.nstar <= 0.0 || !isfinite(s.nstar))
    {
        return s;
    }

    q = get_adk_quantum_numbers(mass, charge);

    s.n = q.n;
    s.l = q.l;
    s.m_abs = q.m;

    if (s.m_abs < 0)
    {
        s.m_abs = -s.m_abs;
    }

    if (s.m_abs > s.l)
    {
        return s;
    }

    /*
     * Eq. (4)
     */
    Cnl2 =
        (1.0 / (2.0 * PI_CONST_LOCAL * s.nstar)) *
        pow((2.0 * exp(1.0)) / s.nstar, 2.0 * s.nstar);

    /*
     * |m|-dependent angular factor:
     * (2l+1)(l+|m|)! / [2^|m| |m|! (l-|m|)!]
     */
    prefactor_lm =
        ((2.0 * (double)s.l + 1.0) *
         factorial_int(s.l + s.m_abs)) /
        (pow(2.0, (double)s.m_abs) *
         factorial_int(s.m_abs) *
         factorial_int(s.l - s.m_abs));

    two_Ip_32 = pow(2.0 * s.Ip_au, 1.5);

    s.prefactor = Cnl2 * prefactor_lm * s.Ip_au;
    s.base = 2.0 * two_Ip_32;
    s.exp_coeff = s.base / 3.0;
    s.power = 2.0 * s.nstar - (double)s.m_abs - 1.0;

    if (!isfinite(s.prefactor) ||
        !isfinite(s.base) ||
        !isfinite(s.exp_coeff) ||
        !isfinite(s.power))
    {
        s.valid = 0;
        return s;
    }

    s.valid = 1;

    return s;
}

static void init_adk_precomputed_table(ADKPrecomputedTable *table)
{
    int k;
    int charge;

    table->nstates = 0;

    for (k = 0; k < NUM_ION_DATA; k++)
    {
        for (charge = 0; charge < ion_data[k].n_ip; charge++)
        {
            ADKPrecomputedState s;

            if (table->nstates >= MAX_ADK_PRECOMPUTED_STATES)
            {
                return;
            }

            s = build_adk_precomputed_state(ion_data[k].mass, charge);

            if (s.valid)
            {
                table->states[table->nstates] = s;
                table->nstates++;
            }
        }
    }
}

static void print_adk_precomputed_table(FILE *fp,
                                        const ADKPrecomputedTable *table)
{
    int i;

    if (fp == NULL || table == NULL)
    {
        return;
    }

    fprintf(fp, "\nPrecomputed ADK states: %d\n", table->nstates);
    fprintf(fp, " mass charge n l |m| Ip_eV nstar prefactor base power exp_coeff\n");

    for (i = 0; i < table->nstates; i++)
    {
        const ADKPrecomputedState *s = &table->states[i];

        fprintf(fp,
                " %4d %6d %1d %1d %3d %10.4f %10.4g %10.4g %10.4g %10.4g %10.4g\n",
                s->mass,
                s->charge,
                s->n,
                s->l,
                s->m_abs,
                s->Ip_eV,
                s->nstar,
                s->prefactor,
                s->base,
                s->power,
                s->exp_coeff);
    }

    fprintf(fp, "\n");
}

static const ADKPrecomputedState *
find_adk_precomputed_state(const ADKPrecomputedTable *table,
                           int mass,
                           int charge)
{
    int i;

    if (table == NULL)
    {
        return NULL;
    }

    for (i = 0; i < table->nstates; i++)
    {
        if (table->states[i].mass == mass &&
            table->states[i].charge == charge)
        {
            return &table->states[i];
        }
    }

    return NULL;
}

static double adk_dc_rate_precomputed_s_inv(const ADKPrecomputedState *s,
                                            double E_V_m)
{
    double E_au;
    double exponent;
    double field_factor;
    double W_au;

    if (s == NULL || !s->valid)
    {
        return 0.0;
    }

    E_au = field_V_m_to_au(E_V_m);

    if (E_au <= 0.0)
    {
        return 0.0;
    }

    exponent = -s->exp_coeff / E_au;

    if (exponent < -700.0)
    {
        return 0.0;
    }

    field_factor = pow(s->base / E_au, s->power);

    W_au = s->prefactor * field_factor * exp(exponent);

    if (W_au < 0.0 || !isfinite(W_au))
    {
        return 0.0;
    }

    return W_au / AU_TIME_S;
}

static IonizationTrial try_adk_ionization_precomputed(
    const ADKPrecomputedTable *table,
    int mass,
    int charge,
    double E_V_m,
    double dt_s,
    gmx_rng_t rng)
{
    IonizationTrial trial;
    const ADKPrecomputedState *s;

    trial.rate_s_inv = 0.0;
    trial.probability = 0.0;
    trial.random_value = 1.0;
    trial.ionized = 0;

    if (dt_s <= 0.0)
    {
        return trial;
    }

    s = find_adk_precomputed_state(table, mass, charge);

    if (s == NULL)
    {
        return trial;
    }

    trial.rate_s_inv = adk_dc_rate_precomputed_s_inv(s, E_V_m);

    if (trial.rate_s_inv <= 0.0)
    {
        return trial;
    }

    trial.probability = 1.0 - exp(-trial.rate_s_inv * dt_s);

    if (trial.probability > 1.0)
    {
        trial.probability = 1.0;
    }

    trial.random_value = (double)gmx_rng_uniform_real(rng);

    if (trial.random_value < trial.probability)
    {
        trial.ionized = 1;
    }

    return trial;
}

/* ---------------------------------------------------------------- *
 * sfmd-collisions: electron-impact ionization by the trapped cloud.
 *
 * Lotz (1968) Maxwellian rate coefficient, b = 0 form, as mc-md's built-in
 * channel (mcmd-coll-data = lotz):
 *   <sigma v> = 6.7e-7 a q T^-3/2 E1(P/T) / (P/T)   cm^3 s^-1,  a = 4.5,
 * q the occupation of the valence shell (the shell ADK ionizes: n = 1 for H,
 * 2 for C N O, 3 for S), P the ionization energy of the charge state, T the
 * cloud's T_e (eV).  The rate of atom k is n_e(k) <sigma v>, n_e(k) the cloud
 * density at the atom.  Inner shells (K: P > 280 eV) are left out, as are
 * the charge states without data (the same as ADK's).
 * ---------------------------------------------------------------- */
#define SFMD_LOTZ_K 6.7e-7
#define SFMD_LOTZ_A 4.5
#define SFMD_SAHA_K 1.656e-22   /* (h^2/2 pi m_e)^3/2 / 2 at 1 eV, cm^3 eV^3/2 (mc-md MCMD_SAHA_K) */

/* exp(x) E1(x), x > 0 (mc-md's mcmd_e1x, Numerical Recipes expint n = 1) */
static double sfmd_e1x(double x)
{
    int i;

    if (x <= 0)
    {
        return 0;
    }
    if (x <= 1.0)
    {
        double sum = 0, term = 1;

        for (i = 1; i < 60; i++)
        {
            term *= -x / i;
            sum  -= term / i;
            if (fabs(term / i) < 1e-17 * fabs(sum)) { break; }
        }
        return exp(x) * (-0.5772156649015329 - log(x) + sum);
    }
    else
    {
        double b = x + 1.0, c = 1e300, d = 1.0 / b, h = d;

        for (i = 1; i < 200; i++)
        {
            double an = -(double)i * i, del;

            b  += 2.0;
            d   = 1.0 / (an * d + b);
            c   = b + an / c;
            del = c * d;
            h  *= del;
            if (fabs(del - 1.0) < 1e-15) { break; }
        }
        return h;
    }
}

/* electrons left in the valence shell of (mass, charge); 0 if none / unknown */
static int sfmd_valence_q(int mass, int charge)
{
    int nval;

    switch (mass)
    {
        case MASS_H: nval = 1; break;
        case MASS_C: nval = 4; break;
        case MASS_N: nval = 5; break;
        case MASS_O: nval = 6; break;
        case MASS_S: nval = 6; break;
        default: return 0;
    }
    return (charge >= 0 && charge < nval) ? nval - charge : 0;
}

/* valence shell size: 2 for H, 8 for C N O (L), 18 for S (M), as mc-md's
 * statistical weights C(2,K) C(8,L) C(18,M) */
static int sfmd_valence_shell(int mass)
{
    switch (mass)
    {
        case MASS_H: return 2;
        case MASS_S: return 18;
        default: return 8;
    }
}

static double sfmd_binom(int n, int k)
{
    double r = 1;
    int    i;

    if (k < 0 || k > n) { return 0; }
    for (i = 1; i <= k; i++) { r = r * (n - k + i) / i; }
    return r;
}

/* three-body recombination rate (s^-1) of (mass, charge) -> charge - 1: the
 * inverse of the impact ionization charge - 1 -> charge by detailed balance,
 * as mc-md (mcmd-coll-recombination):
 *   n_e^2 6.7e-7 a q T^-3/2 e^b E1(b)/b g_lo/g_hi 1.656e-22 T^-3/2,
 * q and g_lo of the lower charge state, b = P/T, P its ionization energy */
static double sfmd_recomb_rate_s_inv(int mass, int charge, double T, double ne, double *P_out)
{
    const double P  = (charge > 0) ? get_ionization_potential_eV(mass, charge - 1) : 0.0;
    const int    q  = (charge > 0) ? sfmd_valence_q(mass, charge - 1) : 0;
    const int    ns = sfmd_valence_shell(mass);
    double       b, glo, ghi;

    *P_out = P;
    if (P <= 0 || q <= 0 || T <= 0 || ne <= 0)
    {
        return 0.0;
    }
    b   = P / T;
    glo = sfmd_binom(ns, q);
    ghi = sfmd_binom(ns, q - 1);
    return ne * ne * SFMD_LOTZ_K * SFMD_LOTZ_A * q * pow(T, -1.5) * sfmd_e1x(b) / b * (glo / ghi) *
           SFMD_SAHA_K * pow(T, -1.5);
}

/* impact-ionization rate (s^-1) of (mass, charge) at T (eV), n_e (cm^-3) */
static double sfmd_lotz_rate_s_inv(int mass, int charge, double T, double ne, double *P_out)
{
    const double P = get_ionization_potential_eV(mass, charge);
    const int    q = sfmd_valence_q(mass, charge);
    double       b;

    *P_out = P;
    if (P <= 0 || q <= 0 || T <= 0 || ne <= 0)
    {
        return 0.0;
    }
    b = P / T;
    if (b > 700)
    {
        return 0.0;
    }
    /* E1(b) / b = e^-b (e^b E1(b)) / b */
    return ne * SFMD_LOTZ_K * SFMD_LOTZ_A * q * pow(T, -1.5) * exp(-b) * sfmd_e1x(b) / b;
}

#define LASER_SIGMA_CUTOFF 6.0

static int laser_is_active(const LaserPulse *p, double t_s)
{
    double arg;

    if (p == NULL || p->sigma_s <= 0.0)
    {
        return 0;
    }

    arg = fabs(t_s - p->t0_s) / p->sigma_s;

    return (arg <= LASER_SIGMA_CUTOFF);
}

typedef struct
{
    double E_now_V_m;
    double I_now_W_cm2;
    double mean_charge_local;
    int n_local;
    int n_ionized_local;
    int laser_active;
} IonizeStepStats;

static int touch_empty_file(const char *filename)
{
    FILE *fp;

    fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error: could not open %s.\n", filename);
        return 1;
    }

    fclose(fp);
    return 0;
}

static int ionize_first_step_init(t_mdatoms *mdatoms, int read_charges, int coll, const t_commrec *cr)
{
    int i;

    /* Only one rank should truncate these shared output files; every rank
     * still zeroes/reads its own local chargeA below. */
    if (MASTER(cr))
    {
        if (touch_empty_file("./simulation_output/pulse_profile.txt"))
        {
            return 1;
        }

        if (touch_empty_file("./simulation_output/transition_log.txt"))
        {
            return 1;
        }

        if (touch_empty_file("./simulation_output/mean_charge_vs_time.txt"))
        {
            return 1;
        }

        /* sfmd-collisions: one line per impact ionization,
         * t[ps] atom element q_before q_after T_e[eV] n_e[cm^-3] rate[1/s] probability random */
        if (coll && touch_empty_file("./simulation_output/collisional_log.txt"))
        {
            return 1;
        }
    }

    /* Under -pd the other ranks append their step-0 ionization events to
     * transition_log.txt in this same step; without a barrier the master's
     * truncation above could land after some of those appends and delete
     * them (seen 2026-11-02: 4-10 of the t=0 lines lost per lysozyme run). */
    if (PAR(cr))
    {
        gmx_barrier(cr);
    }

    /* No partial charges: set initial charges to zero. */
    for (i = mdatoms->start; i < mdatoms->start + mdatoms->homenr; i++)
    {
        mdatoms->chargeA[i] = 0.0;
    }

    if (read_charges)
    {
        int charge_index;
        int temp_charge;
        char line[100];
        FILE *finit;

        finit = fopen("charges.txt", "r");
        if (finit == NULL)
        {
            printf("Error: could not open charge file for reading.\n");
            return 1;
        }

        while (fgets(line, sizeof(line), finit))
        {
            if (sscanf(line, "%d %d", &charge_index, &temp_charge) == 2)
            {
                if (charge_index >= 1 && charge_index <= mdatoms->nr)
                {
                    mdatoms->chargeA[charge_index - 1] = (real)temp_charge;
                    printf("Atom %d charge set to %f\n",
                           charge_index - 1,
                           (double)mdatoms->chargeA[charge_index - 1]);
                }
                else
                {
                    printf("Warning: charge index %d out of range [1, %d]\n",
                           charge_index, mdatoms->nr);
                }
            }
        }

        fclose(finit);
    }

    return 0;
}

/* Ring-gathers a per-atom array laid out atom-major with `stride` reals per
 * atom (stride=1 for a plain per-atom scalar snapshot; stride=N to move a
 * whole window of N buffered per-step values for each atom in a single
 * pass). This is the same particle-decomposition ring communication
 * move_rvecs()/move_reals() (mdlib/mvxvf.c) use -- the same primitive
 * pd_collect_state() (kernel/repl_ex.c) relies on to correctly reassemble
 * full-system positions/velocities from each rank's local PD subset --
 * generalized the same way move_rvecs generalizes move_reals by a fixed
 * DIM=3 stride, except here the stride is a runtime parameter. After
 * `shift` = cr->nnodes - cr->npmenodes - 1 hops (one full lap of the ring),
 * every rank's array holds every other rank's contribution, i.e. the true
 * full-system array instead of just this rank's local PD slice. Only
 * meaningful/safe to call when PAR(cr) && !DOMAINDECOMP(cr) (this module
 * only ever runs under particle decomposition, never domain decomposition
 * -- see gromacs_vmc_pd_required). */
static void move_reals_strided(const t_commrec *cr, gmx_bool bForward,
                               int left, int right,
                               real reals[], int stride, int shift)
{
    int i, cur;
    int *index;
#define next ((cur + 1) % cr->nnodes)
#define prev ((cur - 1 + cr->nnodes) % cr->nnodes)
#define HOMENRI(ind, i) ((ind)[(i) + 1] - (ind)[(i)])

    index = pd_index(cr);
    cur = cr->nodeid;

    for (i = 0; i < shift; i++)
    {
        if (bForward)
        {
            gmx_tx_rx_real(cr,
                           GMX_RIGHT, reals + index[cur] * stride,  HOMENRI(index, cur) * stride,
                           GMX_LEFT,  reals + index[prev] * stride, HOMENRI(index, prev) * stride);
            gmx_wait(cr, right, left);
            cur = prev;
        }
        else
        {
            gmx_tx_rx_real(cr,
                           GMX_LEFT,  reals + index[cur] * stride,  HOMENRI(index, cur) * stride,
                           GMX_RIGHT, reals + index[next] * stride, HOMENRI(index, next) * stride);
            gmx_wait(cr, left, right);
            cur = next;
        }
    }
#undef next
#undef prev
#undef HOMENRI
}

/* Gathers the first n_valid columns of the atom-major charge_buf (stride =
 * interval reals per atom) across PD ranks, transposes them into the
 * on-disk time-major layout (one full n_atoms-length frame per logged
 * step -- the same layout a single-rank run always wrote directly, one
 * fwrite per step), and appends them to fcharges in a single fwrite. Called
 * both at each charge_log_interval window boundary (n_valid ==
 * charge_log_interval) and once more after the main loop for any leftover
 * partial window (n_valid == nsteps % charge_log_interval). No-op if
 * n_valid <= 0 (nothing buffered since the last flush). */
/* Gather one frame of charges across the PD ranks and append it to
 * charges_over_time.bin.  One frame, not a window: the module now writes
 * every sfmd-charge-output-stride'th step rather than buffering every step
 * and batching the writes, which matches the sibling gromacs-mc build and
 * keeps the file to 1/stride of its former size.  The cross-rank
 * communication is just as infrequent as before, since it now happens only
 * on the steps that are actually written. */
static void flush_charge_window(const t_commrec *cr, t_mdatoms *mdatoms,
                                real *charge_buf, FILE *fcharges)
{
    if (PAR(cr) && !DOMAINDECOMP(cr))
    {
        move_reals_strided(cr, FALSE, GMX_LEFT, GMX_RIGHT,
                           charge_buf, 1,
                           cr->nnodes - cr->npmenodes - 1);
    }

    if (MASTER(cr) && fcharges != NULL)
    {
        fwrite(charge_buf, sizeof(charge_buf[0]),
               (size_t)mdatoms->nr, fcharges);
    }
}

static IonizeStepStats do_ionize_step(t_mdatoms *mdatoms,
                                      const LaserPulse *laser,
                                      const ADKPrecomputedTable *adk_table,
                                      double t_ps,
                                      double dt_ps,
                                      int logging,
                                      FILE *fcharges,
                                      real *charge_buf,
                                      int charge_log_interval,
                                      gmx_large_int_t step,
                                      const t_commrec *cr,
                                      gmx_rng_t rng,
                                      int i_begin,
                                      int i_end,
                                      t_sfelec *el,
                                      rvec *x)
{
    IonizeStepStats stats;
    int i;

    stats.E_now_V_m = 0.0;
    stats.I_now_W_cm2 = 0.0;
    stats.mean_charge_local = 0.0;
    stats.n_local = 0;
    stats.n_ionized_local = 0;
    stats.laser_active = 0;

    {
        double t_s;
        double dt_s;

        t_s = t_ps * PS_TO_S;
        dt_s = dt_ps * PS_TO_S;

        const double T_coll = sfelec_coll_Te(el);   /* 0: no impact ionization this step */
        /* recombination at max(T_e, sfmd-te-min); = T_coll without the floor */
        const double T_rec  = sfelec_rec_Te(el);
        /* sfmd-coll-recombination = 2: only once the pulse is over (past the peak,
         * below 1 % of the peak intensity) */
        const int    rec_mode = sfelec_rec_on(el);
        const int    rec_now  = (rec_mode == 1) ||
            (rec_mode == 2 && t_s > laser->t0_s &&
             laser_intensity_W_m2(laser, t_s) < 0.01 * laser->peak_intensity_W_m2);

        stats.laser_active = laser_is_active(laser, t_s);

        if (stats.laser_active)
        {
            stats.E_now_V_m = laser_field_V_m(laser, t_s);
            stats.I_now_W_cm2 = laser_intensity_W_cm2(laser, t_s);
        }

        if (stats.laser_active || T_coll > 0 || (rec_now && T_rec > 0))
        {

            /* [i_begin, i_end) is this rank's home range, or with the
             * electron cloud every atom on the master and none elsewhere */
            for (i = i_begin; i < i_end; i++)
            {
                int mass;
                int charge;
                IonizationTrial trial;

                mass = species_mass(mdatoms->massT[i]);
                charge = (int)round(mdatoms->chargeA[i]);

                trial.ionized = 0;
                if (stats.laser_active)
                {
                    trial = try_adk_ionization_precomputed(
                        adk_table,
                        mass,
                        charge,
                        stats.E_now_V_m,
                        dt_s,
                        rng);
                }

                /* impact ionization, for an atom ADK left alone this step */
                if (!trial.ionized && (T_coll > 0 || (rec_now && T_rec > 0)))
                {
                    const double ne = sfelec_coll_ne(el, i);
                    double       P, Pr = 0, rate, rrec = 0, prob, u;

                    rate = (ne > 0) ? sfmd_lotz_rate_s_inv(mass, charge, T_coll, ne, &P) : 0.0;
                    if (ne > 0 && rec_now)
                    {
                        rrec = sfmd_recomb_rate_s_inv(mass, charge, T_rec, ne, &Pr);
                    }
                    if (rate + rrec > 0)
                    {
                        /* one draw: an event with the total rate, then which one
                         * (with recombination off this is the ionization draw alone) */
                        prob = 1.0 - exp(-(rate + rrec) * dt_s);
                        u    = (double)gmx_rng_uniform_real(rng);
                        if (u < prob && rrec > 0 && u >= prob * rate / (rate + rrec))
                        {
                            mdatoms->chargeA[i] = (real)(charge - 1);
                            if (sfelec_rec_event(el, mdatoms, x, i, Pr))
                            {
                                if (logging)
                                {
                                    FILE *fp;

                                    fp = fopen("./simulation_output/collisional_log.txt", "a");
                                    if (fp != NULL)
                                    {
                                        fprintf(fp, "%g %d %s %d %d %g %g %g %g %g\n", t_ps, i,
                                                mass2char(mass), charge, charge - 1, T_rec, ne, rrec,
                                                prob, u);
                                        fclose(fp);
                                    }
                                }
                            }
                            else
                            {
                                mdatoms->chargeA[i] = (real)charge;
                            }
                        }
                        else if (u < prob)
                        {
                            mdatoms->chargeA[i] = (real)(charge + 1);
                            if (sfelec_coll_event(el, mdatoms, x, i, P))
                            {
                                stats.n_ionized_local++;
                                if (logging)
                                {
                                    FILE *fp;

                                    fp = fopen("./simulation_output/collisional_log.txt", "a");
                                    if (fp != NULL)
                                    {
                                        fprintf(fp, "%g %d %s %d %d %g %g %g %g %g\n", t_ps, i,
                                                mass2char(mass), charge, charge + 1, T_coll, ne, rate,
                                                prob, u);
                                        fclose(fp);
                                    }
                                }
                            }
                            else
                            {
                                mdatoms->chargeA[i] = (real)charge;
                            }
                        }
                    }
                    continue;
                }

                if (trial.ionized)
                {
                    mdatoms->chargeA[i] = (real)(charge + 1);
                    stats.n_ionized_local++;

                    /* the electron is born at rest at its atom (tunnel
                     * exit): it escapes or joins the cloud */
                    if (el != NULL)
                    {
                        sfelec_birth(el, mdatoms, x, i, 0.0);
                    }

                    if (logging)
                    {
                        FILE *fp;

                        fp = fopen("./simulation_output/transition_log.txt", "a");
                        if (fp != NULL)
                        {
                            fprintf(fp,
                                    "%g %d %s %d %d %g %g %g %g\n",
                                    t_ps,
                                    i,
                                    mass2char(mass),
                                    charge,
                                    charge + 1,
                                    stats.E_now_V_m,
                                    trial.rate_s_inv,
                                    trial.probability,
                                    trial.random_value);
                            fclose(fp);
                        }
                    }
                }
            }
        }
    }

    /* With the cloud the master has ionized every atom: publish the charges
     * now, so that the per-rank sums below and the charge window see them */
    if (el != NULL && PAR(cr))
    {
        gmx_bcast(mdatoms->nr * sizeof(mdatoms->chargeA[0]), mdatoms->chargeA, cr);
    }

    for (i = mdatoms->start; i < mdatoms->start + mdatoms->homenr; i++)
    {
        stats.mean_charge_local += (double)mdatoms->chargeA[i];
        stats.n_local++;
    }

    if (stats.n_local > 0)
    {
        stats.mean_charge_local /= (double)stats.n_local;
    }

    /* mean_charge_vs_time.txt / transition counts previously only reflected
     * MASTER's local PD subset (see gromacs_vmc_pd_charge_logging_bug) --
     * this is a true cross-rank reduction, cheap enough (3 doubles) to do
     * every logged step rather than throttling it like the per-atom array
     * below. */
    if (logging && PAR(cr) && !DOMAINDECOMP(cr))
    {
        double sums[3];

        sums[0] = stats.mean_charge_local * (double)stats.n_local;
        sums[1] = (double)stats.n_local;
        sums[2] = (double)stats.n_ionized_local;

        gmx_sumd(3, sums, cr);

        stats.mean_charge_local = (sums[1] > 0.0) ? sums[0] / sums[1] : 0.0;
        stats.n_ionized_local = (int)sums[2];
    }

    /* charges_over_time.bin previously wrote only MASTER's local PD subset,
     * every single step (see gromacs_vmc_pd_charge_logging_bug). Fix:
     * buffer every step's full local subset locally (cheap, no
     * communication), and only gather + fwrite once per
     * charge_log_interval steps -- so every step is still recorded, but the
     * cross-rank ring communication and disk I/O happen 1/charge_log_interval
     * as often. */
    if (logging && charge_buf != NULL &&
        (step % charge_log_interval) == 0)
    {
        for (i = mdatoms->start; i < mdatoms->start + mdatoms->homenr; i++)
        {
            charge_buf[i] = mdatoms->chargeA[i];
        }

        flush_charge_window(cr, mdatoms, charge_buf, fcharges);
    }

    return stats;
}

static void log_ionize_step(const IonizeStepStats *stats,
                            double t_ps,
                            int logging,
                            const t_commrec *cr)
{
    FILE *fp;

    if (!logging)
    {
        return;
    }

    if (MASTER(cr))
    {
        fp = fopen("./simulation_output/pulse_profile.txt", "a");
        if (fp != NULL)
        {
            fprintf(fp,
                    "%g %g %g %d\n",
                    t_ps,
                    stats->I_now_W_cm2,
                    stats->E_now_V_m,
                    stats->laser_active);
            fclose(fp);
        }
    }

    /* Guarded like pulse_profile.txt above: without this, every rank would
     * race to append its own *local* mean_charge_local/n_ionized_local to
     * the same file. Note this still only reports MASTER's local subset of
     * atoms under domain decomposition -- a true global mean would need an
     * MPI reduction of stats across ranks, which is not done here. */
    if (MASTER(cr))
    {
        fp = fopen("./simulation_output/mean_charge_vs_time.txt", "a");
        if (fp != NULL)
        {
            fprintf(fp,
                    "%g %g %d %d\n",
                    t_ps,
                    stats->mean_charge_local,
                    stats->n_ionized_local,
                    stats->laser_active);
            fclose(fp);
        }
    }
}

static void print_laser_pulse_info(FILE *fp,
                                   const LaserPulse *laser,
                                   const ADKPrecomputedTable *adk_table)
{
    if (fp == NULL || laser == NULL)
    {
        return;
    }

    fprintf(fp,
            "\nLaser pulse:\n"
            "  t0              = %g ps\n"
            "  sigma           = %g ps\n"
            "  pulse energy    = %g J\n"
            "  spot diameter   = %g nm\n"
            "  wavelength      = %g nm\n"
            "  peak intensity  = %g W/cm^2\n"
            "  peak field      = %g V/m\n"
            "  carrier phase   = %g rad\n"
            "  active window   = %g ps to %g ps\n"
            "  ADK states      = %d\n\n",
            laser->t0_ps,
            laser->sigma_ps,
            laser->pulse_energy_J,
            laser->spot_diameter_nm,
            laser->wavelength_nm,
            laser->peak_intensity_W_cm2,
            laser->E0_V_m,
            laser->phase_rad,
            laser->t0_ps - LASER_SIGMA_CUTOFF * laser->sigma_ps,
            laser->t0_ps + LASER_SIGMA_CUTOFF * laser->sigma_ps,
            adk_table != NULL ? adk_table->nstates : 0);

    fflush(fp);
}

//////////////////////////////////////////////////////////////////////////////////////
////////////                                                            //////////////
////////////                     Structs ends here                      //////////////
////////////                                                            //////////////
//////////////////////////////////////////////////////////////////////////////////////

static unsigned int sfmd_rng_seed(const t_commrec *cr)
{
    const char *env = getenv("GMX_SFMD_SEED");

    if (env != NULL)
    {
        unsigned int seed = (unsigned int)strtoul(env, NULL, 10);

        /* Mixed with a large odd constant so neighbouring ranks get
         * well-separated seeds rather than adjacent ones. */
        return seed + (unsigned int)(cr->nodeid) * 2654435761u;
    }

    return gmx_rng_make_seed();
}

/* ---------------------------------------------------------------- *
 * Module state
 *
 * Everything here used to be a local of do_md().
 * ---------------------------------------------------------------- */

struct t_sfionize
{
    LaserPulse          laser;
    ADKPrecomputedTable adk_table;
    gmx_rng_t           rng;

    /* Per-step output.  All of it is behind sfmd-detailed-output. */
    gmx_bool bDetailedOutput;
    FILE    *fcharges;          /* charges_over_time.bin              */
    real    *charge_buf;        /* one gathered frame, see the stride */
    int      charge_stride;     /* sfmd-charge-output-stride          */

    /* The trapped-electron cloud (sfmd-electrons = pbgrid), NULL without */
    t_sfelec *el;
    FILE     *fplog;

    /* sfmd-autostop = 2: ds_u holds the unit velocity of each home heavy ion
     * at the last check */
    int       ds_nint;              /* steps between checks */
    gmx_bool  ds_have_ref;          /* a previous check exists */
    rvec     *ds_u;                 /* natoms, only home heavy ions filled */
    FILE     *ds_fp;                /* simulation_output/autostop_dirs.txt (master) */

    /* sfmd-schedule: the current stage's first step, and that step counted in
     * the stage's dt from tinit (sched_k0), so the charge dumps stay on the
     * grid tinit + k x interval; both 0 without a schedule */
    gmx_large_int_t sched_step0, sched_k0;
    int             charge_stride0;   /* sfmd-charge-output-stride at the tpr's dt */
    double          dt0_ps;           /* the tpr's dt */

    /* run summaries (master) */
    int             nHarmonic;        /* harmonic bonds in the topology (they do not break) */
    unsigned int    seed;             /* the master's ADK seed */
    double          q0;               /* total charge after the first-step initialisation */
    double          t_last_ps;        /* time of the last step */
    int             stop_reason;      /* 0 none, 1 energy ratio, 2 directions */
    double          stop_t_fs;
};

/* ---------------------------------------------------------------- *
 * Public entry points
 *
 * mdp options read by this module.  All of them only take effect under
 * mdrun -ionize, which is also what enables the altered force field.
 *
 *   sfmd-pulse-peak-time        centre of the Gaussian pulse [fs]
 *   sfmd-pulse-fwhm             pulse duration, FWHM [fs]
 *   sfmd-pulse-energy           pulse energy [J]
 *   sfmd-pulse-focal-diameter   focal spot diameter [nm]
 *   sfmd-pulse-wavelength       laser wavelength [nm]
 *   sfmd-autostop               1: stop once E_kin/E_tot passes the threshold;
 *                               2: once the ion directions have settled
 *   sfmd-autostop-energy-ratio     that threshold (1)
 *   sfmd-autostop-angle         predicted turn still to come, degrees (2)
 *   sfmd-autostop-interval      fs between direction checks (2)
 *   sfmd-autostop-min-time      fs after the pulse peak before a stop (2)
 *   sfmd-initial-charges        0 = start neutral, 1 = read charges.txt
 *   sfmd-detailed-output        all per-step output (heavy I/O)
 *   sfmd-charge-output-stride   steps between charges_over_time.bin frames
 * ---------------------------------------------------------------- */

t_sfionize *sfionize_init(FILE *fplog, const t_inputrec *ir,
                          t_mdatoms *mdatoms, const t_commrec *cr, int nHarmonicBonds)
{
    t_sfionize  *sf;
    unsigned int seed;

    snew(sf, 1);
    sf->fplog = fplog;

    /* The module is active, so the altered force field is too.  This used to
     * be a separate mdp switch (userint1) that was only ever assigned inside
     * do_md's if (bIonize) block, so it was already implied by -ionize. */
    moldstruct_altered_ff = 1;

    sf->bDetailedOutput = (ir->sfmd_detailed_output != 0);
    sf->charge_stride   = (ir->sfmd_charge_output_stride > 0) ?
                          ir->sfmd_charge_output_stride : 1;
    sf->charge_stride0  = sf->charge_stride;
    sf->dt0_ps          = ir->delta_t;

    /* Each rank/thread gets its own private RNG state, so the ranks do not
     * draw identical ionization sequences and no mutable state is shared
     * across the threads that thread-MPI runs its "ranks" as. */
    seed     = sfmd_rng_seed(cr);
    sf->rng  = gmx_rng_init(seed);
    sf->seed      = seed;
    sf->nHarmonic = nHarmonicBonds;

    if (fplog != NULL)
    {
        fprintf(fplog, "sfmd: ADK random seed %u on rank %d%s\n",
                seed, cr->nodeid,
                getenv("GMX_SFMD_SEED") != NULL ?
                " (pinned by GMX_SFMD_SEED)" : "");
    }

    sf->laser = init_laser_pulse(ir, sf->rng);
    sf->el    = sfelec_init(fplog, ir, cr);

    sf->ds_nint = (int)floor(ir->sfmd_autostop_interval / (ir->delta_t * 1e3) + 0.5);
    if (sf->ds_nint < 1)
    {
        sf->ds_nint = 1;
    }
    if (ir->sfmd_autostop == 2 && fplog != NULL && MASTER(cr))
    {
        fprintf(fplog, "sfmd: autostop 2: stop once the median heavy-ion direction turn still to come is < %g deg "
                "(checked every %g fs, extrapolated as 1/(t - t_peak)^2), not before %g fs after the pulse peak "
                "(%g fs); nsteps is the upper limit (simulation_output/autostop_dirs.txt)\n",
                (double)ir->sfmd_autostop_angle, sf->ds_nint * ir->delta_t * 1e3,
                (double)ir->sfmd_autostop_min_time, (double)ir->sfmd_pulse_peak_time);
    }

    /* The carrier phase is a property of the physical laser field and must be
     * identical on every rank, even though each just drew its own independent
     * value.  gmx_bcast() may only be called when actually running in
     * parallel: with -nt 1 no thread-MPI environment is started, so
     * cr->mpi_comm_mygroup is never set up and MPI_Bcast segfaults. */
    if (PAR(cr))
    {
        gmx_bcast(sizeof(sf->laser.phase_rad), &sf->laser.phase_rad, cr);
    }

    init_adk_precomputed_table(&sf->adk_table);

    if (MASTER(cr))
    {
        print_laser_pulse_info(stderr, &sf->laser, &sf->adk_table);
        print_adk_precomputed_table(stderr, &sf->adk_table);

        if (fplog != NULL)
        {
            print_laser_pulse_info(fplog, &sf->laser, &sf->adk_table);
            print_adk_precomputed_table(fplog, &sf->adk_table);
        }
    }

    /* Output setup is gated on -ionize, not on sfmd-detailed-output: reaching
     * this function already means -ionize was given.  The per-step text files
     * are truncated later, by ionize_first_step_init().
     *
     * NOTE nothing sized by the atom count can be set up here: do_md() calls
     * sfionize_init() before atoms2md() has populated mdatoms, so mdatoms->nr
     * is still 0.  Both such things -- charge_buf and the masses.bin dump --
     * are done on the first step instead, by sfionize_first_step_alloc().
     * Doing them here wrote an empty masses.bin and left charge_buf NULL,
     * which the guard in do_ionize_step() then read as "logging off", so
     * charges_over_time.bin came out empty too.  All three failures were
     * silent. */
    if (MASTER(cr))
    {
        struct stat st = {0};

        if (stat("./simulation_output", &st) == -1 &&
            mkdir("./simulation_output", 0700) != 0)
        {
            gmx_fatal(FARGS,
                      "Could not create the 'simulation_output' directory.");
        }

        sf->fcharges = fopen("./simulation_output/charges_over_time.bin", "wb");
        if (sf->fcharges == NULL)
        {
            gmx_fatal(FARGS,
                      "Could not open 'charges_over_time.bin' for writing.");
        }
    }

    return sf;
}

/* Everything whose size depends on the atom count.  Called on the first step,
 * once atoms2md() has filled mdatoms in -- see the note in sfionize_init().
 *
 * charge_buf must exist on every rank: each fills its own local slice before
 * flush_charge_window() gathers them onto MASTER. */
/* an mdp real as written (7 digits), not its float value */
static double sfmd_mdp_real(real r)
{
    char buf[32];

    sprintf(buf, "%.7g", (double)r);
    return atof(buf);
}

/* The run summary (as gromacs-mc's): what the run will do, from the mdp
 * choices, and warnings for settings outside what was tested.  Printed on the
 * first step (the atoms are not known before), master only. */
#define SFMD_REF_DT_AS 2.0
#define SFMD_LR_AUTO_ATOMS 10000
static void sfionize_print_summary(FILE *fp, const t_sfionize *sf, const t_inputrec *ir, const t_mdatoms *md,
                                   const t_commrec *cr)
{
    const char  *bar   = "==================================================================================";
    const double dt_as = ir->delta_t * 1e6;
    const double Ipk   = sf->laser.peak_intensity_W_cm2;
    int          count[NUM_ELEMENTS] = { 0 }, nother = 0, i, n = 0;
    char         warn[20][360], buf[64], as[256];
    int          nw = 0;
    const int    cloud = (ir->sfmd_electrons != 0);

    if (fp == NULL)
    {
        return;
    }
#define SFMD_WARN(...) do { if (nw < 20) { snprintf(warn[nw++], sizeof(warn[0]), __VA_ARGS__); } } while (0)
    for (i = 0; i < md->nr; i++)
    {
        const int k = mass2idx(species_mass(md->massT[i]));

        if (k >= 0 && k < NUM_ELEMENTS) { count[k]++; } else { nother++; }
    }
    fprintf(fp, "\n%s\n MolDStruct SF (strong-field ionization, ADK)%s\n", bar,
            cloud ? " with the trapped-electron cloud - DEVELOPMENT BUILD" : "");
    fprintf(fp, " System        %d atoms (", md->nr);
    for (i = 0; i < NUM_ELEMENTS; i++)
    {
        if (count[i] > 0)
        {
            fprintf(fp, "%s%s %d", n++ ? ", " : "", elements[i].symbol, count[i]);
        }
    }
    if (nother > 0)
    {
        fprintf(fp, "%sother %d", n ? ", " : "", nother);
    }
    fprintf(fp, "), ionization charge %+.0f at the start%s\n", sf->q0,
            ir->sfmd_initial_charges ? " (sfmd-initial-charges = 1)" : "");
    fprintf(fp, " Pulse         %g J, %g nm, FWHM %g fs, peak at %g fs, focus %g nm\n"
            "               peak intensity %.3g W/cm^2, peak field %.3g V/nm\n",
            (double)ir->sfmd_pulse_energy, (double)ir->sfmd_pulse_wavelength, (double)ir->sfmd_pulse_fwhm,
            (double)ir->sfmd_pulse_peak_time, (double)ir->sfmd_pulse_focal_diameter, Ipk, sf->laser.E0_V_m * 1e-9);
    if (ir->sfmd_sched_n > 0)
    {
        gmx_large_int_t s0, nend = ir->init_step + ir->nsteps;
        double          t0;
        const int       k = sfmd_sched_stage(ir, ir->delta_t, nend, &s0, &t0);

        fprintf(fp, " Run           %s steps to %g fs (sfmd-schedule); seed %u (%s); %d rank%s\n",
                gmx_step_str(ir->nsteps, buf), (t0 + (nend - s0) * sfmd_sched_dt(ir, k, ir->delta_t)) * 1e3,
                sf->seed, getenv("GMX_SFMD_SEED") != NULL ? "GMX_SFMD_SEED" : "drawn at start", cr->nnodes,
                cr->nnodes > 1 ? "s" : "");
        fprintf(fp, " Schedule      from %g fs: dt %g as, PB every %g fs\n", ir->init_t * 1e3, dt_as,
                sfmd_mdp_real(ir->sfmd_pb_interval));
        for (i = 0; i < ir->sfmd_sched_n; i++)
        {
            fprintf(fp, "               from %g fs: dt %g as, PB every %g fs\n", sfmd_mdp_real(ir->sfmd_sched_t[i]),
                    sfmd_mdp_real(ir->sfmd_sched_dt[i]) * 1e3, sfmd_mdp_real(ir->sfmd_sched_pb[i]));
        }
    }
    else
    {
        fprintf(fp, " Run           %s steps x %g as = %g fs; seed %u (%s); %d rank%s\n",
                gmx_step_str(ir->nsteps, buf), dt_as, ir->nsteps * dt_as * 1e-3, sf->seed,
                getenv("GMX_SFMD_SEED") != NULL ? "GMX_SFMD_SEED" : "drawn at start", cr->nnodes,
                cr->nnodes > 1 ? "s" : "");
    }

    fprintf(fp, "\n Processes\n");
    fprintf(fp, "   %-40s on (always)\n", "ADK tunnel ionization");
    if (cloud)
    {
        fprintf(fp, "   %-40s on   sfmd-electrons = pbgrid\n", "trapped electrons (Poisson-Boltzmann)");
        fprintf(fp, "   %-40s s_e %g nm, %g cells/s_e, box <= %g R, <= %d cells, solve every %g fs\n", "    grid",
                sfmd_mdp_real(ir->sfmd_pb_smoothing), sfmd_mdp_real(ir->sfmd_pb_cells_per_smoothing),
                sfmd_mdp_real(ir->sfmd_pb_box_max), ir->sfmd_pb_grid_limit, sfmd_mdp_real(ir->sfmd_pb_interval));
        fprintf(fp, "   %-40s %s\n", "    evaporation", ir->sfmd_evaporation ? "on" : "off");
        fprintf(fp, "   %-40s %s\n", "    laser barrier",
                ir->sfmd_laser_barrier == 1 ? "on: envelope field (sfmd-laser-barrier = 1)" :
                ir->sfmd_laser_barrier == 2 ? "on: instantaneous field for spill/evaporation (sfmd-laser-barrier = 2)" :
                "off");
        if (ir->sfmd_laser_polarization > 0)
        {
            fprintf(fp, "   %-40s %c   field E(t) in the cloud, saddle escape level, q E on the ions\n",
                    "    laser polarization", "xyz"[ir->sfmd_laser_polarization - 1]);
        }
        if (ir->sfmd_laser_heating)
        {
            fprintf(fp, "   %-40s on   Drude %s, nu_ei <= %g omega_p, surface damping A = %g\n", "    laser heating",
                    ir->sfmd_heating_shape == 1 ? "ellipsoid" : "sphere", sfmd_mdp_real(ir->sfmd_heating_nu_cap),
                    sfmd_mdp_real(ir->sfmd_heating_surface));
        }
        else
        {
            fprintf(fp, "   %-40s off\n", "    laser heating");
        }
        fprintf(fp, "   %-40s %s\n", "    electron-impact ionization", ir->sfmd_collisions ? "on   Lotz" : "off");
        fprintf(fp, "   %-40s %s\n", "    three-body recombination",
                ir->sfmd_coll_recombination == 1 ? "on   always (sfmd-coll-recombination = 1)" :
                ir->sfmd_coll_recombination == 2 ? "on   after the pulse (sfmd-coll-recombination = 2)" : "off");
        if (ir->sfmd_te_min > 0)
        {
            fprintf(fp, "   %-40s %g eV: solve, evaporation, recombination at >= it\n", "    temperature floor",
                    sfmd_mdp_real(ir->sfmd_te_min));
        }
        else
        {
            fprintf(fp, "   %-40s old model (sfmd-te-min = 0): solve at >= 1 eV, E_kin clamped at 0\n",
                    "    temperature floor");
        }
    }
    else
    {
        fprintf(fp, "   %-40s none (sfmd-electrons = none: every electron leaves)\n", "trapped electrons");
    }
    if (ir->sfmd_pb_long_range == 1)
    {
        fprintf(fp, "   %-40s %g nm cutoff + grid long range\n", "ion-ion forces", (double)ir->rcoulomb);
    }
    else if (ir->rcoulomb > 0)
    {
        fprintf(fp, "   %-40s %g nm cutoff, nothing beyond\n", "ion-ion forces", (double)ir->rcoulomb);
    }
    else
    {
        fprintf(fp, "   %-40s all pairs\n", "ion-ion forces");
    }
    if (ir->sfmd_autostop == 1)
    {
        snprintf(as, sizeof(as), "E_kin/E_tot > %g (sfmd-autostop = 1)", (double)ir->sfmd_autostop_energy_ratio);
    }
    else if (ir->sfmd_autostop == 2)
    {
        snprintf(as, sizeof(as), "ion directions settled (sfmd-autostop = 2): predicted remaining heavy-ion "
                 "turn < %g deg, checked every %g fs, from %g fs after the peak; at most nsteps",
                 (double)ir->sfmd_autostop_angle, (double)ir->sfmd_autostop_interval,
                 (double)ir->sfmd_autostop_min_time);
    }
    else
    {
        snprintf(as, sizeof(as), "off");
    }
    fprintf(fp, "\n Output        simulation_output/: %s, charges_over_time.bin every %d steps;\n"
            "               autostop %s\n",
            sf->bDetailedOutput ? "event logs (sfmd-detailed-output = 1)" : "no event logs (sfmd-detailed-output = 0)",
            sf->charge_stride, as);

    /* warnings: settings outside what was tested (lysozyme, 850 nm, 15 fs, 1e15-1e17 W/cm^2) */
    if (dt_as > SFMD_REF_DT_AS * 1.001 && ir->sfmd_sched_n == 0)
    {
        SFMD_WARN("dt = %g as: the runs were made at %g as", dt_as, SFMD_REF_DT_AS);
    }
    if (sf->nHarmonic > 0)
    {
        SFMD_WARN("%d harmonic bonds: they neither weaken nor break under -ionize; set morse = yes in the mdp",
                  sf->nHarmonic);
    }
    if (ir->sfmd_pulse_energy <= 0 || ir->sfmd_pulse_wavelength <= 0 || ir->sfmd_pulse_fwhm <= 0)
    {
        SFMD_WARN("sfmd-pulse-energy, -wavelength or -fwhm is 0: nothing will be ionized");
    }
    if (cloud)
    {
        if (Ipk > 0 && (Ipk < 0.5e15 || Ipk > 2e17))
        {
            SFMD_WARN("peak intensity %.2g W/cm^2: the cloud was tested at 1e15-1e17 W/cm^2 (lysozyme, 850 nm, 15 fs)", Ipk);
        }
        if (ir->sfmd_laser_polarization > 0)
        {
            SFMD_WARN("sfmd-laser-polarization: development, not validated (escape and forces depend on the axis)");
        }
        if (ir->sfmd_laser_barrier == 0)
        {
            SFMD_WARN("sfmd-laser-barrier = 0: the field-free cloud traps ~98 %% of the electrons at any intensity");
        }
        else if (ir->sfmd_laser_barrier == 2 && ir->sfmd_laser_polarization == 0)
        {
            SFMD_WARN("sfmd-laser-barrier = 2: same physics as 1 but more seed noise and clamped energy; 1 is recommended");
        }
        if (ir->sfmd_laser_heating == 0)
        {
            SFMD_WARN("sfmd-laser-heating = 0: the cloud cools to the floor and is seed-bistable at 1e15 W/cm^2");
        }
        else if (ir->sfmd_heating_surface <= 0)
        {
            SFMD_WARN("sfmd-heating-surface = 0: the Mie resonance is undamped (15-19 MeV heating outliers in ~1 of 8 "
                      "runs at 1e16-1e17)");
        }
        if (ir->sfmd_laser_heating && fabs(ir->sfmd_heating_nu_cap - 1) > 1e-6)
        {
            SFMD_WARN("sfmd-heating-nu-cap = %g: tested at 1 (0.5: -8 %% ion energy at 1e15)", (double)ir->sfmd_heating_nu_cap);
        }
        if (ir->sfmd_coll_recombination == 1)
        {
            SFMD_WARN("sfmd-coll-recombination = 1: capture during the pulse pumps ADK (~9x the events, +22 %% ion "
                      "energy); diagnostic only, use 2");
        }
        if (ir->sfmd_te_min <= 0)
        {
            SFMD_WARN("sfmd-te-min = 0: the old model (E_kin clamped at 0, recombination unbounded as T_e -> 0)");
        }
        if (sfmd_mdp_real(ir->sfmd_pb_interval) > 0.2 * 1.001 || fabs(sfmd_mdp_real(ir->sfmd_pb_smoothing) - 0.25) > 1e-6)
        {
            SFMD_WARN("sfmd-pb-interval %g fs, sfmd-pb-smoothing %g nm: tested at 0.2 fs and 0.25 nm (the ion energy "
                      "depends on the smoothing)", sfmd_mdp_real(ir->sfmd_pb_interval), sfmd_mdp_real(ir->sfmd_pb_smoothing));
        }
        if (ir->sfmd_pb_long_range == 0 && ir->rcoulomb <= 0 && md->nr > SFMD_LR_AUTO_ATOMS)
        {
            SFMD_WARN("all pairs with %d atoms: O(N^2); sfmd-pb-long-range = auto would use the grid long range",
                      md->nr);
        }
        if (ir->sfmd_pb_long_range == 0 && ir->rcoulomb > 0)
        {
            SFMD_WARN("a %g nm cutoff without the long range: the net charge's push is lost and the sample "
                      "under-explodes", (double)ir->rcoulomb);
        }
    }
#undef SFMD_WARN
    if (nw > 0)
    {
        fprintf(fp, "\n Warnings (settings outside what was tested)\n");
        for (i = 0; i < nw; i++)
        {
            fprintf(fp, "   - %s\n", warn[i]);
        }
    }
    fprintf(fp, "%s\n\n", bar);
    fflush(fp);
}

static void sfionize_first_step_alloc(t_sfionize *sf, t_mdatoms *mdatoms,
                                      const t_commrec *cr)
{
    if (sf->charge_buf == NULL)
    {
        snew(sf->charge_buf, mdatoms->nr);
    }

    if (MASTER(cr))
    {
        FILE *fmasses = fopen("./simulation_output/masses.bin", "wb");

        if (fmasses == NULL)
        {
            gmx_fatal(FARGS, "Could not open 'masses.bin' for writing.");
        }
        fwrite(mdatoms->massT, sizeof(mdatoms->massT[0]), mdatoms->nr, fmasses);
        fclose(fmasses);
    }
}

void sfionize_step(t_sfionize *sf, const t_inputrec *ir, t_mdatoms *mdatoms,
                   rvec *x, double t, gmx_large_int_t step,
                   gmx_bool bFirstStep, const t_commrec *cr, t_nrnb *nrnb)
{
    IonizeStepStats stats;
    int             i_begin, i_end;

    if (sf == NULL)
    {
        return;
    }

    if (bFirstStep)
    {
        sfionize_first_step_alloc(sf, mdatoms, cr);

        if (ionize_first_step_init(mdatoms, ir->sfmd_initial_charges, ir->sfmd_collisions, cr))
        {
            gmx_fatal(FARGS, "Could not initialise the ionization charges.");
        }
    }
    sf->t_last_ps = t;

    /* Without the cloud every rank ionizes its own home atoms with its own
     * stream.  With it every event needs the depth of the whole sample and
     * changes the cloud, so the master ionizes all atoms, in order, with its
     * stream (as gromacs-mc's Monte Carlo), and broadcasts the charges. */
    if (sf->el == NULL)
    {
        i_begin = mdatoms->start;
        i_end   = mdatoms->start + mdatoms->homenr;
    }
    else
    {
        /* ionize_first_step_init() zeroes only the home atoms; the master is
         * about to ionize every atom, and the rest still hold force-field
         * partial charges (-0.83 would round to -1).  sfmd-initial-charges
         * is refused with the cloud, so every atom starts neutral. */
        if (bFirstStep)
        {
            int a;

            for (a = 0; a < mdatoms->nr; a++)
            {
                mdatoms->chargeA[a] = 0;
            }
        }
        /* the envelope, not the oscillating field: the cloud sees the
         * cycle-averaged laser (sfmd-laser-barrier) */
        sfelec_set_laser(sf->el, laser_field_amplitude_V_m(&sf->laser, t * PS_TO_S));
        /* the carrier too, for sfmd-laser-barrier = 2 (spill, evaporation) */
        sfelec_set_laser_now(sf->el, laser_field_V_m(&sf->laser, t * PS_TO_S));
        sfelec_step_begin(sf->el, ir, mdatoms, x, t, bFirstStep, cr, nrnb);
        i_begin = 0;
        i_end   = MASTER(cr) ? mdatoms->nr : 0;
    }

    /* the run summary, once the charges are the run's starting ones (the
     * cloud's first step replaces the topology's partial charges above) */
    if (bFirstStep && MASTER(cr))
    {
        int a;

        sf->q0 = 0;
        for (a = 0; a < mdatoms->nr; a++)
        {
            sf->q0 += mdatoms->chargeA[a];
        }
        sfionize_print_summary(stderr, sf, ir, mdatoms, cr);
        sfionize_print_summary(sf->fplog, sf, ir, mdatoms, cr);
    }

    stats = do_ionize_step(mdatoms, &sf->laser, &sf->adk_table, t,
                           ir->delta_t, sf->bDetailedOutput, sf->fcharges,
                           sf->charge_buf, sf->charge_stride, step - sf->sched_step0 + sf->sched_k0, cr,
                           sf->rng, i_begin, i_end, sf->el, x);

    log_ionize_step(&stats, t, sf->bDetailedOutput, cr);

    if (sf->el != NULL)
    {
        /* charges already broadcast in do_ionize_step() */
        sfelec_step_end(sf->el, mdatoms, x, cr);
        return;
    }

    /* do_ionize_step() only updates this rank's own home range of
     * mdatoms->chargeA.  Under PD every rank keeps a full-length array, but
     * nothing else keeps the other ranks' slices in sync during the run, and
     * charges change every step while the pulse is on.  Without this
     * re-sync, do_force() would evaluate every cross-rank pair against a
     * stale charge for the far atom - wrong forces, not just wrong output. */
    if (PAR(cr) && !DOMAINDECOMP(cr))
    {
        move_reals_strided(cr, FALSE, GMX_LEFT, GMX_RIGHT,
                           mdatoms->chargeA, 1,
                           cr->nnodes - cr->npmenodes - 1);
    }
}

void sfionize_screen_forces(t_sfionize *sf, const t_inputrec *ir,
                            t_mdatoms *mdatoms, rvec *x, rvec *v, rvec *f,
                            const t_commrec *cr)
{
    if (sf == NULL)
    {
        return;
    }
    sfelec_forces(sf->el, ir, mdatoms, x, v, f, cr);
}

gmx_bool sfionize_autostop(t_sfionize *sf, const t_inputrec *ir,
                           double E_kin, double E_tot, gmx_large_int_t step,
                           gmx_bool bEnergiesGlobal, const t_commrec *cr)
{
    int stop = 0;

    /* sf == NULL already means -ionize was not given, which is what gates the
     * whole module.  All three of these tests are rank-independent, so every
     * rank returns here together and none is left behind at the broadcast. */
    if (sf == NULL || ir->sfmd_autostop != 1)
    {
        return FALSE;
    }

    /* enerd->term[] is only summed across ranks on global-communication
     * steps.  On any other step it carries this rank's own partial energies,
     * and the ratio computed from them is not the system's - under particle
     * decomposition that makes the test fire on one rank and not the others,
     * which then deadlock against each other. */
    if (!bEnergiesGlobal)
    {
        return FALSE;
    }

    if (MASTER(cr))
    {
        double ratio = E_kin / E_tot;

        if (ratio > ir->sfmd_autostop_energy_ratio && step > 10000)
        {
            fprintf(stderr,
                    "\nSimulation terminated: ratio %f is over threshold %f\n",
                    ratio, (double)ir->sfmd_autostop_energy_ratio);
            stop = 1;
            sf->stop_reason = 1;
            sf->stop_t_fs   = sf->t_last_ps * 1e3;
        }
    }

    /* Decide once and tell everyone: the threshold is crossed by a hair, so
     * even identical inputs cannot be relied on to give every rank the same
     * answer. */
    if (PAR(cr))
    {
        gmx_bcast(sizeof(stop), &stop, cr);
    }

    return (stop != 0);
}

/* sfmd-autostop = 2: stop once the ion directions have settled (gromacs-mc's
 * mcionize_dirstop, as of 6e81e9f).
 *
 * Every ds_nint steps each rank measures, for its home heavy ions (hydrogen
 * is left out: the protons leave first and are not what the maps are made
 * of), the angle between the velocity direction now and at the last check,
 * and bins it into a log-spaced histogram that is summed over the ranks.  The
 * master takes the median turn s over the interval dt_c and extrapolates it:
 * late in the explosion the turning rate falls like 1/(t - t_peak)^2, so the
 * turn still to come is
 *
 *     rest = s * (t - dt_c - t_peak) / dt_c,
 *
 * which on gromacs-mc's 250 fs reference runs was 1.7-3x the turn actually
 * left: an overestimate, so the stop is on the safe side.  The run stops once
 * rest < sfmd-autostop-angle and t - t_peak >= sfmd-autostop-min-time (early
 * on the turning has not reached the 1/t^2 regime and the extrapolation
 * undershoots).  Decided on the master and broadcast, like the energy test:
 * a rank that leaves the MD loop alone hangs the others.
 *
 * The velocity in the state is the leap-frog half step; for a direction that
 * makes no difference at the resolution this needs. */
#define SFMD_DS_NB  512        /* histogram bins: [0, lo], then log-spaced lo..180 deg */
#define SFMD_DS_LO  1e-4       /* degrees */

gmx_bool sfionize_dirstop(t_sfionize *sf, const t_inputrec *ir, const t_mdatoms *md, rvec *v,
                          gmx_large_int_t step, double t, const t_commrec *cr)
{
    double       hist[SFMD_DS_NB + 1];      /* last entry: the number of ions counted */
    const double lr = log(180.0 / SFMD_DS_LO);
    int          stop = 0, a, k;

    /* All of these are rank-independent, so every rank returns together. */
    if (sf == NULL || ir->sfmd_autostop != 2 || (step - sf->sched_step0) % sf->ds_nint != 0)
    {
        return FALSE;
    }
    if (sf->ds_u == NULL)
    {
        snew(sf->ds_u, md->nr);
    }

    for (k = 0; k <= SFMD_DS_NB; k++)
    {
        hist[k] = 0;
    }
    for (a = md->start; a < md->start + md->homenr; a++)
    {
        double vn, u[3];

        if (species_mass(md->massT[a]) == 1)
        {
            continue;
        }
        vn = sqrt((double)v[a][XX] * v[a][XX] + (double)v[a][YY] * v[a][YY] + (double)v[a][ZZ] * v[a][ZZ]);
        if (vn <= 0)
        {
            clear_rvec(sf->ds_u[a]);
            continue;
        }
        u[0] = v[a][XX] / vn;
        u[1] = v[a][YY] / vn;
        u[2] = v[a][ZZ] / vn;
        if (sf->ds_have_ref && norm2(sf->ds_u[a]) > 0)
        {
            double c   = u[0] * sf->ds_u[a][XX] + u[1] * sf->ds_u[a][YY] + u[2] * sf->ds_u[a][ZZ];
            double ang = acos(c > 1 ? 1 : (c < -1 ? -1 : c)) * 180.0 / M_PI;

            k = (ang <= SFMD_DS_LO) ? 0 : 1 + (int)(log(ang / SFMD_DS_LO) / lr * (SFMD_DS_NB - 1));
            if (k > SFMD_DS_NB - 1)
            {
                k = SFMD_DS_NB - 1;
            }
            hist[k] += 1;
            hist[SFMD_DS_NB] += 1;
        }
        sf->ds_u[a][XX] = u[0];
        sf->ds_u[a][YY] = u[1];
        sf->ds_u[a][ZZ] = u[2];
    }
    if (!sf->ds_have_ref)
    {
        sf->ds_have_ref = TRUE;     /* the first check only sets the reference */
        return FALSE;
    }
    if (PAR(cr))
    {
        gmx_sumd(SFMD_DS_NB + 1, hist, cr);
    }

    if (MASTER(cr))
    {
        const double n    = hist[SFMD_DS_NB];
        const double t_fs = t * 1e3;
        const double tp   = ir->sfmd_pulse_peak_time;
        const double dtc  = sf->ds_nint * ir->delta_t * 1e3;
        const double lev  = (t_fs - dtc - tp) / dtc;
        double       med  = 0, cum = 0, pred;

        if (n > 0)
        {
            for (k = 0; k < SFMD_DS_NB; k++)
            {
                if (cum + hist[k] >= 0.5 * n && hist[k] > 0)
                {
                    const double f  = (0.5 * n - cum) / hist[k];
                    const double e0 = (k == 0) ? 0 : SFMD_DS_LO * exp(lr * (k - 1) / (SFMD_DS_NB - 1));
                    const double e1 = SFMD_DS_LO * exp(lr * (k == 0 ? 0 : k) / (SFMD_DS_NB - 1));

                    med = (k == 0) ? f * e1 : e0 * pow(e1 / e0, f);
                    break;
                }
                cum += hist[k];
            }
        }
        pred = (lev > 0 && n > 0) ? med * lev : -1;   /* -1: no prediction yet */

        if (sf->ds_fp == NULL)
        {
            sf->ds_fp = fopen("./simulation_output/autostop_dirs.txt", "w");
            if (sf->ds_fp != NULL)
            {
                fprintf(sf->ds_fp, "# sfmd-autostop = 2: every %g fs, the median turn of the heavy-ion velocity "
                        "directions since the last check, and the turn predicted still to come (median * (t - %g fs - "
                        "t_peak) / %g fs; -1 before there is one)\n# stop once it is < %g deg and t - t_peak >= %g fs "
                        "(t_peak = %g fs)\n# t[fs] turn[deg] predicted_rest[deg] heavy_ions\n",
                        dtc, dtc, dtc, (double)ir->sfmd_autostop_angle, (double)ir->sfmd_autostop_min_time, tp);
            }
        }
        if (sf->ds_fp != NULL)
        {
            fprintf(sf->ds_fp, "%.3f %.5f %.4f %.0f\n", t_fs, med, pred, n);
            fflush(sf->ds_fp);
        }

        if (pred >= 0 && pred < ir->sfmd_autostop_angle && t_fs - tp >= ir->sfmd_autostop_min_time)
        {
            fprintf(stderr, "\nSimulation terminated: ion directions settled at %.1f fs (%.1f fs after the pulse "
                    "peak): median heavy-ion turn %.3f deg in the last %g fs, %.2f deg predicted still to come, under "
                    "sfmd-autostop-angle = %g\n", t_fs, t_fs - tp, med, dtc, pred, (double)ir->sfmd_autostop_angle);
            if (sf->fplog != NULL)
            {
                fprintf(sf->fplog, "\nsfmd: autostop 2 at %.1f fs (%.1f fs after the peak): median turn %.3f deg in "
                        "%g fs, %.2f deg predicted still to come\n", t_fs, t_fs - tp, med, dtc, pred);
            }
            stop = 1;
            sf->stop_reason = 2;
            sf->stop_t_fs   = t_fs;
        }
    }
    if (PAR(cr))
    {
        gmx_bcast(sizeof(stop), &stop, cr);
    }

    return (stop != 0);
}

/* ---------------------------------------------------------------- *
 * sfmd-schedule (gromacs-mc's mcmd-schedule, 6e81e9f): the MD step and the
 * PB interval change at given times.  Stage 0 runs with the tpr's dt and
 * sfmd-pb-interval from tinit; stage k >= 1 from sfmd_sched_t[k-1] (fs) with
 * sfmd_sched_dt / _pb[k-1].  The stage boundaries are steps that follow from
 * the tpr alone (grompp makes every stage start on a step of the one before),
 * so the stage of any step, and with it the time, is known without saving
 * anything.
 * ---------------------------------------------------------------- */

/* Stage of MD step `step`, with its first step and start time (ps); dt0 is the tpr's dt (ps) */
int sfmd_sched_stage(const t_inputrec *ir, double dt0, gmx_large_int_t step, gmx_large_int_t *step0, double *t0)
{
    gmx_large_int_t s = 0, sn;
    double          ta = ir->init_t, dta = dt0;
    int             k;

    for (k = 0; k < ir->sfmd_sched_n; k++)
    {
        const double tb = sfmd_mdp_real(ir->sfmd_sched_t[k]) * 1e-3;

        sn = s + (gmx_large_int_t)floor((tb - ta) / dta + 0.5);
        if (step < sn)
        {
            break;
        }
        s   = sn;
        ta  = tb;
        dta = sfmd_sched_dt(ir, k + 1, dt0);
    }
    *step0 = s;
    *t0    = ta;
    return k;
}

/* Stage k's MD step (ps, as single-precision dt) and PB interval (fs); stage 0 is the tpr's */
double sfmd_sched_dt(const t_inputrec *ir, int k, double dt0)
{
    return (k == 0) ? dt0 : (double)(real)(sfmd_mdp_real(ir->sfmd_sched_dt[k - 1]) * 1e-3);
}

double sfmd_sched_pb(const t_inputrec *ir, int k)
{
    return sfmd_mdp_real(k == 0 ? ir->sfmd_pb_interval : ir->sfmd_sched_pb[k - 1]);
}

/* MD steps from step 0 to t_end (ps) through the schedule */
gmx_large_int_t sfmd_sched_steps_to(const t_inputrec *ir, double dt0, double t_end)
{
    gmx_large_int_t s = 0;
    double          ta = ir->init_t, dta = dt0;
    int             k;

    for (k = 0; k < ir->sfmd_sched_n; k++)
    {
        const double tb = sfmd_mdp_real(ir->sfmd_sched_t[k]) * 1e-3;

        if (t_end <= tb)
        {
            break;
        }
        s  += (gmx_large_int_t)floor((tb - ta) / dta + 0.5);
        ta  = tb;
        dta = sfmd_sched_dt(ir, k + 1, dt0);
    }
    return s + (gmx_large_int_t)floor((t_end - ta) / dta + 0.5);
}

/* The module's step-counted intervals for the stage that starts at MD step
 * step0 and time t0 (ps), with ir->delta_t already the stage's: the direction
 * checks every sfmd-autostop-interval, the charge dumps at the time interval
 * they had at the tpr's step, and the cloud's (sfelec_schedule).  bChange:
 * the stage starts now (dt_prev, ps, the step before it). */
void sfionize_schedule(t_sfionize *sf, FILE *fplog, const t_inputrec *ir, int stage, double pb_fs,
                       gmx_large_int_t step0, double t0, gmx_bool bChange, double dt_prev)
{
    const double dt_fs = ir->delta_t * 1e3;
    int          n;
    char         sbuf[32];

    if (sf == NULL)
    {
        return;
    }
    sf->sched_step0 = step0;
    sf->sched_k0    = (gmx_large_int_t)floor((t0 - ir->init_t) / ir->delta_t + 0.5);
    n               = (int)floor(ir->sfmd_autostop_interval / dt_fs + 0.5);
    sf->ds_nint     = (n < 1) ? 1 : n;
    n                 = (int)floor(sf->charge_stride0 * sf->dt0_ps / ir->delta_t + 0.5);
    sf->charge_stride = (n < 1) ? 1 : n;
    sfelec_schedule(sf->el, ir, pb_fs, step0, t0, bChange, dt_prev);
    if (fplog != NULL && (bChange || stage > 0))
    {
        fprintf(fplog, "sfmd: schedule stage %d from %g fs (step %s): dt %g fs, %s charge dumps every %d steps\n",
                stage, t0 * 1e3, gmx_step_str(step0, sbuf), dt_fs,
                sfelec_schedule_str(sf->el), sf->charge_stride);
    }
}

void sfionize_done(t_sfionize *sf, t_mdatoms *mdatoms, int natoms,
                   const t_commrec *cr)
{
    if (sf == NULL)
    {
        return;
    }

    /* charges.txt used to index chargeA[0..natoms-1] as if it held every atom
     * in the system, which is only true for single-rank runs; under PD
     * everything past this rank's local slice was still the original
     * pre-ionization topology charge.  Gather the true full-system array -
     * collectively, every rank must participate - before the MASTER-only
     * dump. */
    if (PAR(cr) && !DOMAINDECOMP(cr))
    {
        move_reals_strided(cr, FALSE, GMX_LEFT, GMX_RIGHT,
                           mdatoms->chargeA, 1,
                           cr->nnodes - cr->npmenodes - 1);
    }

    if (MASTER(cr))
    {
        FILE *fp = fopen("./simulation_output/charges.txt", "w");
        int   i;

        if (fp == NULL)
        {
            gmx_fatal(FARGS, "Could not open 'charges.txt' for writing.");
        }
        for (i = 1; i < natoms + 1; i++)
        {
            fprintf(fp, "%d %d\n", i, (int)round(mdatoms->chargeA[i - 1]));
        }
        fclose(fp);
    }

    if (MASTER(cr))
    {
        FILE  *out[2] = { stderr, sf->fplog };
        double qsum[NUM_ELEMENTS] = { 0 }, ncnt[NUM_ELEMENTS] = { 0 }, qtot = 0, nco = 0, nre = 0;
        int    a, k, j;

        for (a = 0; a < natoms; a++)
        {
            k = mass2idx(species_mass(mdatoms->massT[a]));
            if (k >= 0 && k < NUM_ELEMENTS)
            {
                qsum[k] += mdatoms->chargeA[a];
                ncnt[k] += 1;
            }
            qtot += mdatoms->chargeA[a];
        }
        sfelec_counts(sf->el, &nco, &nre);
        for (j = 0; j < 2; j++)
        {
            if (out[j] == NULL)
            {
                continue;
            }
            fprintf(out[j], "\n==================================================================================\n"
                    " MolDStruct SF: end of the run at %g fs", sf->t_last_ps * 1e3);
            if (sf->stop_reason == 1)
            {
                fprintf(out[j], " (autostop 1: E_kin/E_tot over sfmd-autostop-energy-ratio)\n");
            }
            else if (sf->stop_reason == 2)
            {
                fprintf(out[j], " (autostop 2: ion directions settled, %.1f fs after the pulse peak)\n",
                        sf->stop_t_fs - sf->laser.t0_ps * 1e3);
            }
            else
            {
                fprintf(out[j], " (nsteps)\n");
            }
            fprintf(out[j], " Charge        total %+.0f, %.3f per atom; per element:", qtot, natoms > 0 ? qtot / natoms : 0.0);
            for (k = 0; k < NUM_ELEMENTS; k++)
            {
                if (ncnt[k] > 0)
                {
                    fprintf(out[j], " %s %.2f", elements[k].symbol, qsum[k] / ncnt[k]);
                }
            }
            fprintf(out[j], "\n Events        ADK %.0f", qtot - sf->q0 - nco + nre);
            if (sf->el != NULL)
            {
                fprintf(out[j], ", electron impact %.0f, three-body recombination %.0f", nco, nre);
            }
            fprintf(out[j], "\n Summary tool  tools/sfmd_output.py summary|plot simulation_output\n");
            fprintf(out[j], "==================================================================================\n");
        }
    }

    sfelec_done(sf->el, sf->fplog, cr);

    if (sf->fcharges != NULL)
    {
        fclose(sf->fcharges);
    }
    sfree(sf->charge_buf);
    sfree(sf->ds_u);
    if (sf->ds_fp != NULL)
    {
        fclose(sf->ds_fp);
    }
    if (sf->rng != NULL)
    {
        gmx_rng_destroy(sf->rng);
    }
    sfree(sf);
}
