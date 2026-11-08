#include "trgctl.h"

#include <math.h>
#include <string.h>

#define DEG_TO_MRAD 17.45329252f
#define PEAK_RATE_WINDOW 50U /* samples (ms) before the break */
#define PEAK_RATE_STEP   5U  /* rate over 5 ms, so one noisy ADC sample does not count */

/*
 * Windows from the fault timings; thresholds from a grid search on a simulated
 * training set (tools/evaluate.py --tune, seed 1), taken from the flat top of the
 * score rather than the single best cell. Not tuned on real shooters.
 */
void trgctl_default_cfg(struct trgctl_cfg *cfg)
{
	cfg->disp_window_us = 150000U;
	cfg->ref_from_us = 160000U;
	cfg->ref_to_us = 320000U;
	cfg->hold_from_us = 200000U;
	cfg->hold_to_us = 600000U;
	cfg->disp_thr_mrad = 1.5f;
	cfg->disp_sway_k = 6.0f;
	cfg->sway_thr_mrad = 0.3f;
	cfg->onset_k = 4.0f;
	cfg->onset_floor_dps = 0.3f;
	cfg->jerk_rate_pm_per_ms = 2.5f;
	cfg->jerk_onset_us = 25000U;
	cfg->flinch_dip_share = 0.5f;
	cfg->wall_pm = 620U;
}

const char *trgctl_label_name(enum trgctl_label l)
{
	switch (l) {
	case TRGCTL_GOOD:
		return "GOOD";
	case TRGCTL_UNSTEADY:
		return "UNSTEADY";
	case TRGCTL_JERK:
		return "JERK";
	case TRGCTL_FLINCH:
		return "FLINCH";
	case TRGCTL_DISTURBED:
		return "DISTURBED";
	default:
		return "?";
	}
}

/* Age of a sample before the break; negative if after it. Wrap-safe. */
static int32_t age_us(uint32_t t_break_us, uint32_t t_us)
{
	return (int32_t)(t_break_us - t_us);
}

/* Rectangle rule, as in the firmware's imu_metrics: a sample holds until the next one (or the break). */
static float dt_s(const struct trgctl_imu_sample *imu, uint16_t n, uint16_t i, uint32_t t_break_us)
{
	uint32_t until = t_break_us;

	if ((i + 1U) < n && age_us(t_break_us, imu[i + 1U].t_us) >= 0) {
		until = imu[i + 1U].t_us;
	}
	return (float)(until - imu[i].t_us) * 1e-6f;
}

static bool in_window(int32_t age, uint32_t from, uint32_t to)
{
	return (age >= (int32_t)from) && (age <= (int32_t)to);
}

static void analyze_trigger(const struct trgctl_cfg *cfg, const uint16_t *trig, uint16_t n,
			    struct trgctl_result *out)
{
	uint16_t last = (uint16_t)(n - 1U);
	uint16_t i = last;
	float peak = 0.0f;

	/* Final press: from the last sample below the wall to the break. */
	while (i > 0U && trig[i] >= cfg->wall_pm) {
		i--;
	}
	out->squeeze_ms = (trig[i] >= cfg->wall_pm) ? (float)n : (float)(last - i);

	for (uint16_t k = 0U; k < PEAK_RATE_WINDOW && k + PEAK_RATE_STEP <= last; k++) {
		uint16_t j = (uint16_t)(last - k);
		float rate = ((float)trig[j] - (float)trig[j - PEAK_RATE_STEP]) / (float)PEAK_RATE_STEP;

		if (rate > peak) {
			peak = rate;
		}
	}
	out->peak_rate_pm_per_ms = peak;
}

bool trgctl_analyze(const struct trgctl_cfg *cfg,
		    const struct trgctl_imu_sample *imu, uint16_t n_imu,
		    uint32_t t_break_us,
		    const uint16_t *trig, uint16_t n_trig,
		    struct trgctl_result *out)
{
	float ref_y = 0.0f, ref_z = 0.0f, ref_ax = 0.0f, ref_ss = 0.0f, ref_x = 0.0f;
	float sxx = 0.0f, sxy = 0.0f, sxz = 0.0f, slope_y = 0.0f, slope_z = 0.0f;
	float hold_y = 0.0f, hold_z = 0.0f;
	float dip = 0.0f, left = 0.0f, disp_ax = 0.0f;
	uint16_t n_ref = 0U, n_hold = 0U, n_disp = 0U;

	(void)memset(out, 0, sizeof(*out));
	if (imu == NULL || trig == NULL || n_imu == 0U || n_trig < 2U) {
		return false;
	}

	/* Pass 1: means of the reference and hold windows. */
	for (uint16_t i = 0U; i < n_imu; i++) {
		int32_t age = age_us(t_break_us, imu[i].t_us);

		if (in_window(age, cfg->ref_from_us, cfg->ref_to_us)) {
			ref_y += imu[i].gy;
			ref_z += imu[i].gz;
			ref_ax += imu[i].ax;
			ref_x += -(float)age * 1e-6f;
			n_ref++;
		}
		if (in_window(age, cfg->hold_from_us, cfg->hold_to_us)) {
			hold_y += imu[i].gy;
			hold_z += imu[i].gz;
			n_hold++;
		}
	}
	if (n_ref < 3U || n_hold < 10U) {
		return false;
	}
	ref_y /= (float)n_ref;
	ref_z /= (float)n_ref;
	ref_ax /= (float)n_ref;
	ref_x /= (float)n_ref;

	/* Reference rate is a straight-line fit over the reference window, extrapolated
	 * to the break: it follows the slow hold sway and the gyro bias together.
	 */
	for (uint16_t i = 0U; i < n_imu; i++) {
		int32_t age = age_us(t_break_us, imu[i].t_us);

		if (in_window(age, cfg->ref_from_us, cfg->ref_to_us)) {
			float dx = (-(float)age * 1e-6f) - ref_x;

			sxx += dx * dx;
			sxy += dx * (imu[i].gy - ref_y);
			sxz += dx * (imu[i].gz - ref_z);
		}
	}
	if (sxx > 0.0f) {
		slope_y = sxy / sxx;
		slope_z = sxz / sxx;
	}
	hold_y /= (float)n_hold;
	hold_z /= (float)n_hold;

	/* Pass 2: residual spread in the reference window, movement at the break,
	 * and the hold angle trace (mean rate removed) for the steadiness RMS.
	 */
	float ang_y = 0.0f, ang_z = 0.0f;
	float sum_y = 0.0f, sum_z = 0.0f, sum_sq = 0.0f;

	for (uint16_t i = 0U; i < n_imu; i++) {
		int32_t age = age_us(t_break_us, imu[i].t_us);

		if (age < 0) {
			continue;
		}
		float dx = (-(float)age * 1e-6f) - ref_x;
		float ry = imu[i].gy - (ref_y + (slope_y * dx));
		float rz = imu[i].gz - (ref_z + (slope_z * dx));

		if (in_window(age, cfg->ref_from_us, cfg->ref_to_us)) {
			ref_ss += (ry * ry) + (rz * rz);
		}
		if (age <= (int32_t)cfg->disp_window_us) {
			float dt = dt_s(imu, n_imu, i, t_break_us);

			dip += ry * dt;
			left += rz * dt;
			disp_ax += imu[i].ax;
			n_disp++;
		}
		if (in_window(age, cfg->hold_from_us, cfg->hold_to_us)) {
			float dt = dt_s(imu, n_imu, i, t_break_us);

			ang_y += (imu[i].gy - hold_y) * dt * DEG_TO_MRAD;
			ang_z += (imu[i].gz - hold_z) * dt * DEG_TO_MRAD;
			sum_y += ang_y;
			sum_z += ang_z;
			sum_sq += (ang_y * ang_y) + (ang_z * ang_z);
		}
	}
	if (n_disp < 3U) {
		return false;
	}

	float mean_y = sum_y / (float)n_hold;
	float mean_z = sum_z / (float)n_hold;
	float var = (sum_sq / (float)n_hold) - (mean_y * mean_y) - (mean_z * mean_z);

	out->sway_rms_mrad = (var > 0.0f) ? sqrtf(var) : 0.0f;
	out->dip_mrad = dip * DEG_TO_MRAD;
	out->left_mrad = left * DEG_TO_MRAD;
	out->disp_mrad = sqrtf((out->dip_mrad * out->dip_mrad) + (out->left_mrad * out->left_mrad));
	out->push_g = (disp_ax / (float)n_disp) - ref_ax;

	/* Onset: the oldest sample in the movement window whose residual rate stands
	 * out; needs at least two such samples, so one noise spike is not a movement.
	 */
	float thr = cfg->onset_k * sqrtf(ref_ss / (float)n_ref);
	int32_t onset_age = -1;
	uint16_t n_out = 0U;

	if (thr < cfg->onset_floor_dps) {
		thr = cfg->onset_floor_dps;
	}
	for (uint16_t i = 0U; i < n_imu; i++) {
		int32_t age = age_us(t_break_us, imu[i].t_us);
		float dx = (-(float)age * 1e-6f) - ref_x;
		float ry = imu[i].gy - (ref_y + (slope_y * dx));
		float rz = imu[i].gz - (ref_z + (slope_z * dx));

		if (age < 0 || age > (int32_t)cfg->disp_window_us) {
			continue;
		}
		if (sqrtf((ry * ry) + (rz * rz)) > thr) {
			if (age > onset_age) {
				onset_age = age;
			}
			n_out++;
		}
	}
	if (n_out < 2U) {
		onset_age = -1;
	}
	out->onset_ms = (onset_age >= 0) ? (float)onset_age * 1e-3f : 0.0f;

	analyze_trigger(cfg, trig, n_trig, out);
	out->imu_samples = n_hold;

	/* Decision. */
	float disp_thr = cfg->disp_thr_mrad;

	if (cfg->disp_sway_k * out->sway_rms_mrad > disp_thr) {
		disp_thr = cfg->disp_sway_k * out->sway_rms_mrad;
	}
	if (out->disp_mrad < disp_thr) {
		out->label = (out->sway_rms_mrad > cfg->sway_thr_mrad) ? TRGCTL_UNSTEADY : TRGCTL_GOOD;
	} else if (out->peak_rate_pm_per_ms >= cfg->jerk_rate_pm_per_ms ||
		   (onset_age >= 0 && onset_age <= (int32_t)cfg->jerk_onset_us)) {
		out->label = TRGCTL_JERK;
	} else if (out->dip_mrad >= cfg->flinch_dip_share * out->disp_mrad) {
		out->label = TRGCTL_FLINCH;
	} else {
		out->label = TRGCTL_DISTURBED;
	}
	return true;
}
