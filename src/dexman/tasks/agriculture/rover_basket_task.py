"""Evaluation of a parked rover's orange pickup and deposit, in metres and seconds."""
from dataclasses import dataclass

from .contact_quality import MAX_HOLD_LINEAR_SPEED_M_S, MIN_STABLE_HOLD_S
from .objectives import FORBIDDEN_IMPACT_N


def over_basket(position):
    x,y,z = position
    return -.10 < x < .69 and abs(y) < .39 and .45 < z < .86


@dataclass
class BasketTrial:
    grip_ticks: int = 0
    acquired: bool = False
    hold_ticks: int = 0
    longest_hold: int = 0
    retained: bool = False
    basket_ticks: int = 0
    peak_lift: float = 0.0
    peak_base_drift: float = 0.0
    peak_impact: float = 0.0
    lost_before_release: bool = False
    require_continuous_grip: bool = False
    delivery_ticks: int = 0
    held_delivery_ticks: int = 0
    opened_over_basket: bool | None = None
    release_position_m: tuple | None = None

    def observe(self, *, phase, lift, held, steady, speed, basket_position, basket_contact,
                finger_clearance, touching, drift, impact, dt):
        self.peak_base_drift = max(self.peak_base_drift, drift)
        self.peak_impact = max(self.peak_impact, impact)
        self.peak_lift = max(self.peak_lift, lift)
        self.grip_ticks = self.grip_ticks + 1 if held and abs(lift) < 0.03 else 0
        self.acquired |= self.grip_ticks * dt >= 0.10
        valid = (phase == "sosten" and self.acquired and held and steady
                 and speed < MAX_HOLD_LINEAR_SPEED_M_S and lift >= 0.15)
        self.hold_ticks = self.hold_ticks + 1 if valid else 0
        self.longest_hold = max(self.longest_hold, self.hold_ticks)
        self.retained |= self.hold_ticks * dt >= MIN_STABLE_HOLD_S
        if phase == "traslado":
            self.delivery_ticks += 1
            self.held_delivery_ticks += int(held)
        if self.acquired and phase == "traslado" and not (held if self.require_continuous_grip else touching):
            self.lost_before_release = True
        x, y, z = basket_position
        if self.require_continuous_grip and self.retained and phase == "sosten" and not held:
            self.lost_before_release = True
        if phase == "apertura" and self.opened_over_basket is None:
            self.opened_over_basket = over_basket(basket_position)
        if phase in ("apertura","retirada") and self.retained and not touching and self.release_position_m is None:
            self.release_position_m = tuple(basket_position)
        seated = -0.10 < x < 0.69 and abs(y) < 0.39 and 0.45 < z < 0.66 and basket_contact
        released = (phase == "retirada" and seated and not touching
                    and finger_clearance > 0.07 and speed < 0.045)
        self.basket_ticks = self.basket_ticks + 1 if released else 0

    def result(self, dt, *, valid_start=True):
        deposited = self.basket_ticks * dt >= 0.50
        released_inside = self.release_position_m is not None and over_basket(self.release_position_m)
        return dict(grasp=self.acquired, hold_15cm=self.retained, deposit=deposited,
                    held_delivery_ticks=self.held_delivery_ticks, delivery_ticks=self.delivery_ticks,
                    opened_over_basket=self.opened_over_basket, release_position_m=self.release_position_m,
                    peak_lift_m=self.peak_lift, stable_hold_s=self.longest_hold * dt,
                    final_basket_rest_s=self.basket_ticks * dt, max_base_drift_m=self.peak_base_drift,
                    max_forbidden_force_n=self.peak_impact, lost_in_transit=self.lost_before_release,
                    success=valid_start and self.retained and deposited and not self.lost_before_release
                    and (not self.require_continuous_grip or (self.delivery_ticks>0 and self.opened_over_basket is True and released_inside))
                    and self.peak_base_drift < 0.03 and self.peak_impact <= FORBIDDEN_IMPACT_N)
