import numpy as np
def project_action(state, action, vmax=12.0, amax=2.0, dt=0.2):
    ax, ay = np.clip(action, -amax, amax)
    x,y,vx,vy = state
    vx2 = np.clip(vx + ax*dt, -vmax, vmax)
    vy2 = np.clip(vy + ay*dt, -vmax, vmax)
    ax2 = (vx2 - vx)/dt; ay2 = (vy2 - vy)/dt
    return np.array([ax2, ay2], dtype=np.float32)
