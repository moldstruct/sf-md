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

static int ionize_first_step_init(t_mdatoms *mdatoms, int read_charges, const t_commrec *cr)
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
                                      gmx_rng_t rng)
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

        stats.laser_active = laser_is_active(laser, t_s);

        if (stats.laser_active)
        {
            stats.E_now_V_m = laser_field_V_m(laser, t_s);
            stats.I_now_W_cm2 = laser_intensity_W_cm2(laser, t_s);

            for (i = mdatoms->start; i < mdatoms->start + mdatoms->homenr; i++)
            {
                int mass;
                int charge;
                IonizationTrial trial;

                mass = (int)round(mdatoms->massT[i]);
                charge = (int)round(mdatoms->chargeA[i]);

                trial = try_adk_ionization_precomputed(
                    adk_table,
                    mass,
                    charge,
                    stats.E_now_V_m,
                    dt_s,
                    rng);

                if (trial.ionized)
                {
                    mdatoms->chargeA[i] = (real)(charge + 1);
                    stats.n_ionized_local++;

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
 *   sfmd-autostop               stop once E_kin/E_tot passes the threshold
 *   sfmd-autostop-threshold     that threshold
 *   sfmd-initial-charges        0 = start neutral, 1 = read charges.txt
 *   sfmd-detailed-output        all per-step output (heavy I/O)
 *   sfmd-charge-output-stride   steps between charges_over_time.bin frames
 * ---------------------------------------------------------------- */

t_sfionize *sfionize_init(FILE *fplog, const t_inputrec *ir,
                          t_mdatoms *mdatoms, const t_commrec *cr)
{
    t_sfionize  *sf;
    unsigned int seed;

    snew(sf, 1);

    /* The module is active, so the altered force field is too.  This used to
     * be a separate mdp switch (userint1) that was only ever assigned inside
     * do_md's if (bIonize) block, so it was already implied by -ionize. */
    moldstruct_altered_ff = 1;

    sf->bDetailedOutput = (ir->sfmd_detailed_output != 0);
    sf->charge_stride   = (ir->sfmd_charge_output_stride > 0) ?
                          ir->sfmd_charge_output_stride : 1;

    /* Each rank/thread gets its own private RNG state, so the ranks do not
     * draw identical ionization sequences and no mutable state is shared
     * across the threads that thread-MPI runs its "ranks" as. */
    seed     = sfmd_rng_seed(cr);
    sf->rng  = gmx_rng_init(seed);

    if (fplog != NULL)
    {
        fprintf(fplog, "sfmd: ADK random seed %u on rank %d%s\n",
                seed, cr->nodeid,
                getenv("GMX_SFMD_SEED") != NULL ?
                " (pinned by GMX_SFMD_SEED)" : "");
    }

    sf->laser = init_laser_pulse(ir, sf->rng);

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
     * charge_buf must exist on every rank: each fills its own local slice
     * before flush_charge_window() gathers them onto MASTER. */
    snew(sf->charge_buf, mdatoms->nr);

    if (MASTER(cr))
    {
        struct stat st = {0};
        FILE       *fmasses;

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

        fmasses = fopen("./simulation_output/masses.bin", "wb");
        if (fmasses == NULL)
        {
            gmx_fatal(FARGS, "Could not open 'masses.bin' for writing.");
        }
        fwrite(mdatoms->massT, sizeof(mdatoms->massT[0]), mdatoms->nr, fmasses);
        fclose(fmasses);
    }

    return sf;
}

void sfionize_step(t_sfionize *sf, const t_inputrec *ir, t_mdatoms *mdatoms,
                   double t, gmx_large_int_t step, gmx_bool bFirstStep,
                   const t_commrec *cr)
{
    IonizeStepStats stats;

    if (sf == NULL)
    {
        return;
    }

    if (bFirstStep &&
        ionize_first_step_init(mdatoms, ir->sfmd_initial_charges, cr))
    {
        gmx_fatal(FARGS, "Could not initialise the ionization charges.");
    }

    stats = do_ionize_step(mdatoms, &sf->laser, &sf->adk_table, t,
                           ir->delta_t, sf->bDetailedOutput, sf->fcharges,
                           sf->charge_buf, sf->charge_stride, step, cr,
                           sf->rng);

    log_ionize_step(&stats, t, sf->bDetailedOutput, cr);

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

gmx_bool sfionize_autostop(const t_sfionize *sf, const t_inputrec *ir,
                           double E_kin, double E_tot, gmx_large_int_t step)
{
    double ratio;

    /* sf == NULL already means -ionize was not given, which is what gates the
     * whole module, so no separate force-field test is needed here. */
    if (sf == NULL || ir->sfmd_autostop == 0)
    {
        return FALSE;
    }

    ratio = E_kin / E_tot;

    if (ratio > ir->sfmd_autostop_threshold && step > 10000)
    {
        printf("\nSimulation terminated: Ratio %f is over threshold %f\n",
               ratio, (double)ir->sfmd_autostop_threshold);
        return TRUE;
    }

    return FALSE;
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

    if (sf->fcharges != NULL)
    {
        fclose(sf->fcharges);
    }
    sfree(sf->charge_buf);
    if (sf->rng != NULL)
    {
        gmx_rng_destroy(sf->rng);
    }
    sfree(sf);
}
