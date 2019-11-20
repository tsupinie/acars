
import numpy as np

_p0_isa = 1089.
_T0_isa = 19.

_z0_trops_isa = -0.610
_z0_tropp_isa = 11.
_z0_strats_isa = 20.

_lr_trops_isa = 6.5
_lr_tropp_isa = 0
_lr_strats_isa = -1

def std_atmosphere_pres(p0=_p0_isa, T0=_T0_isa, z0_trops=_z0_trops_isa, lr_trops=_lr_trops_isa,
                                                z0_tropp=_z0_tropp_isa, lr_tropp=_lr_tropp_isa,
                                                z0_strats=_z0_strats_isa, lr_strats=_lr_strats_isa):
    """
    std_atmosphere_pres

    This function returns a function for computing pressure (in Pascals) in a standard atmosphere given height (in 
    meters). The default parameters are set for the International Standard Atmosphere.

    Usage:
    >>> sap = std_atmosphere_pres(p0=1089.)
    >>> p_ary_1 = sap(z_ary_1)
    >>> p_ary_2 = sap(z_ary_2)
    """
    gas_const = 287.
    accel_grav = 9.806

    def const_lapse_rate_layer(z, **params):
        scale = accel_grav / gas_const
        hght = 1000 * (z - params['z0']) / (params['T0'] + 273.15)

        if np.isclose(params['lr'], 0):
            p = params['p0'] * np.exp(-scale * hght)
        else:
            laps = params['lr'] / 1000.
            p = params['p0'] * (1 - laps * hght) ** (scale / laps)
        return p

    T0_trops = T0
    T0_tropp = T0_trops - lr_trops * (z0_tropp - z0_trops)
    T0_strats = T0_tropp - lr_tropp * (z0_strats - z0_tropp)

    p0_trops = p0 * 100.
    p0_tropp = const_lapse_rate_layer(z0_tropp, p0=p0_trops, T0=T0_trops, z0=z0_trops, lr=lr_trops)
    p0_strats = const_lapse_rate_layer(z0_strats, p0=p0_tropp, T0=T0_tropp, z0=z0_tropp, lr=lr_tropp)

    def compute_std_atmos(z):
        zkm = z / 1000.

        pres = np.ma.where(zkm < z0_tropp,
            const_lapse_rate_layer(zkm, p0=p0_trops, T0=T0_trops, z0=z0_trops, lr=lr_trops),
            np.ma.where(zkm < z0_strats,
                const_lapse_rate_layer(zkm, p0=p0_tropp, T0=T0_tropp, z0=z0_tropp, lr=lr_tropp),
                const_lapse_rate_layer(zkm, p0=p0_strats, T0=T0_strats, z0=z0_strats, lr=lr_strats)
            )
        )

        return pres

    return compute_std_atmos


