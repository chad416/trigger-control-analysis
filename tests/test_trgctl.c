/* Unit tests for trgctl: hand-built signals with known answers. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../trgctl/trgctl.h"

static int failures;
static int checks;

#define CHECK(cond)                                                                \
	do {                                                                       \
		checks++;                                                          \
		if (!(cond)) {                                                     \
			failures++;                                                \
			printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
		}                                                                  \
	} while (0)
#define NEAR(a, b, tol) CHECK(fabsf((float)(a) - (float)(b)) <= (float)(tol))

#define PERIOD_US 4808U /* 208 Hz */
#define N_IMU     130U  /* 625 ms */

static struct trgctl_imu_sample imu[N_IMU];
static uint16_t trig[TRGCTL_TRIG_MAX];
static struct trgctl_cfg cfg;

/* IMU samples ending exactly at the break, gy/gz from a function of age (s). */
typedef void (*rate_fn)(float age_s, float *gy, float *gz);

static void fill_imu(uint32_t t_break, rate_fn f)
{
	for (uint16_t i = 0U; i < N_IMU; i++) {
		uint32_t age = (uint32_t)(N_IMU - 1U - i) * PERIOD_US;
		float gy = 0.0f, gz = 0.0f;

		f((float)age * 1e-6f, &gy, &gz);
		imu[i] = (struct trgctl_imu_sample){ .t_us = t_break - age, .gx = 0.0f, .gy = gy, .gz = gz,
						     .ax = 0.0f, .ay = 0.0f, .az = 1.0f };
	}
}

/* Take-up to the wall at 300 ms, then a linear press of press_ms to the break at 800. */
static void fill_trig(uint32_t press_ms)
{
	for (uint32_t i = 0U; i < TRGCTL_TRIG_MAX; i++) {
		uint32_t age = TRGCTL_TRIG_MAX - 1U - i;

		if (age > 700U) {
			trig[i] = 0U;
		} else if (age > press_ms) {
			trig[i] = 615U;
		} else {
			trig[i] = (uint16_t)(615U + ((800U - 615U) * (press_ms - age)) / press_ms);
		}
	}
}

static void rate_still(float a, float *gy, float *gz) { (void)a; *gy = 0.3f; *gz = -0.2f; } /* bias only */
static void rate_ramp(float a, float *gy, float *gz) { *gy = 0.3f + 1.0f * (0.6f - a); *gz = -0.5f * (0.6f - a); }
static void rate_flinch(float a, float *gy, float *gz)
{
	*gy = 0.3f + ((a <= 0.100f && a >= 0.020f) ? 5.0f : 0.0f);
	*gz = 0.0f;
}
static void rate_jerk(float a, float *gy, float *gz)
{
	*gy = (a <= 0.020f) ? 6.0f : 0.0f;
	*gz = (a <= 0.020f) ? 4.0f : 0.0f;
}
static void rate_side(float a, float *gy, float *gz)
{
	*gy = 0.0f;
	*gz = (a <= 0.090f && a >= 0.020f) ? 5.0f : 0.0f;
}

static struct trgctl_result run(uint32_t t_break)
{
	struct trgctl_result r;

	CHECK(trgctl_analyze(&cfg, imu, N_IMU, t_break, trig, TRGCTL_TRIG_MAX, &r));
	return r;
}

int main(void)
{
	struct trgctl_result r;
	const uint32_t tb = 50000000U;

	trgctl_default_cfg(&cfg);
	CHECK(cfg.ref_from_us > cfg.disp_window_us);
	CHECK(cfg.hold_to_us <= (TRGCTL_IMU_MAX - 1U) * PERIOD_US);

	/* bad input */
	fill_trig(400U);
	CHECK(!trgctl_analyze(&cfg, NULL, 0U, tb, trig, TRGCTL_TRIG_MAX, &r));
	CHECK(r.imu_samples == 0U);
	fill_imu(tb, rate_still);
	CHECK(!trgctl_analyze(&cfg, imu, 20U, tb, trig, TRGCTL_TRIG_MAX, &r)); /* 20 samples: no hold window */

	/* still weapon with gyro bias: nothing moves */
	r = run(tb);
	CHECK(r.label == TRGCTL_GOOD);
	NEAR(r.disp_mrad, 0.0f, 0.01f);
	NEAR(r.sway_rms_mrad, 0.0f, 0.01f);
	CHECK(r.onset_ms == 0.0f);
	NEAR(r.squeeze_ms, 390.0f, 3.0f); /* travel passes 620 about 10 ms into the press */
	NEAR(r.peak_rate_pm_per_ms, 185.0f / 400.0f, 0.15f);

	/* rate changing linearly (steady sway acceleration): the fit removes it */
	fill_imu(tb, rate_ramp);
	r = run(tb);
	CHECK(r.label == TRGCTL_GOOD);
	NEAR(r.disp_mrad, 0.0f, 0.05f);

	/* same signal, stricter hold limit: the hold reads as shaky, nothing moves at the break */
	cfg.sway_thr_mrad = 0.05f;
	r = run(tb);
	CHECK(r.label == TRGCTL_UNSTEADY);
	trgctl_default_cfg(&cfg);

	/* flinch: +5 deg/s dip from 100 to 20 ms before the break = 0.4 deg = 6.98 mrad */
	fill_imu(tb, rate_flinch);
	r = run(tb);
	CHECK(r.label == TRGCTL_FLINCH);
	NEAR(r.dip_mrad, 6.98f, 0.5f);
	NEAR(r.left_mrad, 0.0f, 0.05f);
	NEAR(r.onset_ms, 100.0f, 5.0f);

	/* jerk by trigger: fast 15 ms press with a dip-left movement in the last 20 ms */
	fill_imu(tb, rate_jerk);
	fill_trig(15U);
	r = run(tb);
	CHECK(r.label == TRGCTL_JERK);
	CHECK(r.peak_rate_pm_per_ms > 10.0f);
	NEAR(r.squeeze_ms, 15.0f, 1.0f);
	CHECK(r.dip_mrad > 0.0f && r.left_mrad > 0.0f);

	/* jerk by timing: slow press, but the movement starts 20 ms before the break */
	fill_trig(400U);
	r = run(tb);
	CHECK(r.label == TRGCTL_JERK);
	CHECK(r.onset_ms > 0.0f && r.onset_ms <= 25.0f);

	/* sideways movement, slow press, early onset: no known pattern */
	fill_imu(tb, rate_side);
	r = run(tb);
	CHECK(r.label == TRGCTL_DISTURBED);
	CHECK(r.left_mrad > 4.0f);

	/* clock wrap: same flinch with the break just after 2^32 */
	fill_imu(1000U, rate_flinch);
	r = run(1000U);
	CHECK(r.label == TRGCTL_FLINCH);
	NEAR(r.dip_mrad, 6.98f, 0.5f);

	/* samples after the break are ignored */
	fill_imu(tb, rate_flinch);
	imu[N_IMU - 1U].t_us = tb + 3000U;
	imu[N_IMU - 1U].gy = 500.0f;
	r = run(tb);
	CHECK(r.dip_mrad < 10.0f);

	/* names */
	CHECK(strcmp(trgctl_label_name(TRGCTL_FLINCH), "FLINCH") == 0);
	CHECK(strcmp(trgctl_label_name((enum trgctl_label)99), "?") == 0);

	printf("%d checks, %d failures\n", checks, failures);
	return failures == 0 ? 0 : 1;
}
