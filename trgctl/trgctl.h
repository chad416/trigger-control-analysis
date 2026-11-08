/*
 * trgctl: per-shot trigger-control analysis for an instrumented training weapon.
 *
 * Looks at the half second before the trigger breaks and answers: did the
 * muzzle move at the break, which way, when did the movement start, and does
 * the trigger trace say the press was a snatch? From that it labels the shot
 * GOOD, UNSTEADY (large hold sway, no break fault), JERK (fast final press
 * with a disturbance at the break) or FLINCH (muzzle dips, starting well
 * before the break: anticipation of recoil), or DISTURBED (moved, no pattern).
 *
 * Inputs are the same signals the trainer firmware already keeps:
 *  - IMU samples (deg/s, g) in the weapon frame: x forward along the barrel,
 *    y to the left, z up. Positive gy dips the muzzle, positive gz moves it left.
 *  - trigger travel in permille at 1 kHz, the last sample being the break.
 *
 * Pure C99, no heap, no OS calls; float maths. Designed to run once per shot
 * on the weapon's MCU, after the break, from buffers it already holds.
 */
#ifndef TRGCTL_H_
#define TRGCTL_H_

#include <stdbool.h>
#include <stdint.h>

#define TRGCTL_IMU_MAX  160U  /* > 600 ms at 208 Hz */
#define TRGCTL_TRIG_MAX 1000U /* 1 s at 1 kHz */

struct trgctl_imu_sample {
	uint32_t t_us;
	float gx, gy, gz; /* deg/s */
	float ax, ay, az; /* g */
};

enum trgctl_label {
	TRGCTL_GOOD = 0,
	TRGCTL_UNSTEADY = 1,
	TRGCTL_JERK = 2,
	TRGCTL_FLINCH = 3,
	TRGCTL_DISTURBED = 4,
};

struct trgctl_cfg {
	/* windows, as age before the break in microseconds */
	uint32_t disp_window_us; /* movement at the break is measured over this window */
	uint32_t ref_from_us;    /* reference (drift) window, newer edge */
	uint32_t ref_to_us;      /* reference (drift) window, older edge */
	uint32_t hold_from_us;   /* hold-steadiness window, newer edge */
	uint32_t hold_to_us;     /* hold-steadiness window, older edge */
	/* decision thresholds */
	float disp_thr_mrad;      /* below this the break is clean */
	float disp_sway_k;        /* ... and below disp_sway_k x hold RMS (a shaky hold moves anyway) */
	float sway_thr_mrad;      /* hold RMS above this is UNSTEADY */
	float onset_k;            /* residual rate > k x reference RMS starts a movement */
	float onset_floor_dps;    /* ... and at least this */
	float jerk_rate_pm_per_ms;/* final-press rate at or above this is a snatch */
	uint32_t jerk_onset_us;   /* movement starting this close to the break is a snatch */
	float flinch_dip_share;   /* dip share of the movement for FLINCH */
	uint16_t wall_pm;         /* final press = time since travel was last below this (just past the wall) */
};

struct trgctl_result {
	enum trgctl_label label;
	float dip_mrad;        /* muzzle movement at the break, + = down */
	float left_mrad;       /* + = left */
	float disp_mrad;       /* magnitude */
	float onset_ms;        /* how long before the break the movement started; 0 if none */
	float sway_rms_mrad;   /* hold steadiness */
	float push_g;          /* forward acceleration change at the break (anticipation push) */
	float squeeze_ms;      /* wall to break */
	float peak_rate_pm_per_ms; /* fastest trigger travel in the last 50 ms */
	uint16_t imu_samples;  /* samples used, 0 if the analysis failed */
};

void trgctl_default_cfg(struct trgctl_cfg *cfg);

/*
 * imu: samples in time order (oldest first), n of them; t_break_us on the
 * same clock. trig: travel in permille at 1 kHz, oldest first, trig[n_trig-1]
 * at the break. Returns false (result zeroed) if the IMU window is too thin.
 */
bool trgctl_analyze(const struct trgctl_cfg *cfg,
		    const struct trgctl_imu_sample *imu, uint16_t n_imu,
		    uint32_t t_break_us,
		    const uint16_t *trig, uint16_t n_trig,
		    struct trgctl_result *out);

const char *trgctl_label_name(enum trgctl_label l);

#endif /* TRGCTL_H_ */
