import sys

sys.path.append("..")
#sys.path.append("../programming/")

import math
import numpy as np
import battery_core as bat

def test_blank_discharge():
    duration = 3000
    battery = bat.battery18650()
    i_batt = np.random.random((duration, 1)) + 9 * np.ones((duration,1))
    for t in range(duration):
        battery.step(i_batt[t])
        
    assert duration