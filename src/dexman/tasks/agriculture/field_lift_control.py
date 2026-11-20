"""Height feedback along the wrist path of a physically loaded hand."""
import numpy as np


class LiftHeightControl:
    def __init__(self,paths,versions,target_height=.157):
        self.target_height = target_height
        self.sides = [side for _,side,_,_ in versions]
        self.paths = []
        for path in paths:
            closed = next(q for q,phase,_ in reversed(path) if phase=="cierre")
            self.paths.append(np.stack([closed[:8],*(q[:8] for q,phase,_ in path if phase=="elevacion")]))
        self.progress = np.zeros(len(paths))
        self.hold_goal = self.progress.copy()
        self.feedforward = np.full(len(paths),np.nan)
        self.rise_time = 0.
        self.hold_heights = []
        self.bias = np.zeros((len(paths),8))
        self.transfer_ticks = 0

    def apply(self,command,phase,heights,dt):
        if phase in ("elevacion","sosten"):
            if phase=="elevacion":
                # Approach from below. Reversing on instantaneous measurements
                # excites the compliant arm and the banana swings in the fingers.
                rate = np.clip(8.*(self.target_height+.008-heights),0.,.45)
                self.progress = np.minimum(1.,self.progress+dt*rate)
                self.rise_time += dt
                u = min(1.,self.rise_time/2.4)
                mask = np.isfinite(self.feedforward)
                self.progress[mask] = self.feedforward[mask]*u*u*(3.-2.*u)
                self.hold_goal = self.progress.copy()
            else:
                self.hold_heights.append(heights.copy())
                settle_ticks = max(2,round(.8/dt))
                if len(self.hold_heights)>=settle_ticks:
                    settled = np.median(self.hold_heights[-max(1,settle_ticks//3):],axis=0)
                    error = self.target_height-settled
                    correction = np.where((error>.004)|(error<-.011),np.clip(error/.22,-.04,.04),0.)
                    # The recorded elbow branch already has a stationary hold.
                    # Lift further only if the actual loaded fruit settled too low.
                    correction[np.isfinite(self.feedforward)&(settled>=.152)] = 0.
                    self.hold_goal = np.clip(self.progress+correction,0.,1.)
                    self.hold_heights.clear()
                self.progress += np.clip(self.hold_goal-self.progress,-.085*dt,.085*dt)
            for i,(side,path) in enumerate(zip(self.sides,self.paths,strict=True)):
                t = self.progress[i]*(len(path)-1)
                j = min(int(t),len(path)-2)
                q = path[j]+(t-j)*(path[j+1]-path[j])
                command[i,30*side:30*side+8] = command.new_tensor(q)
                self.bias[i] = q-path[-1]
        elif phase=="traslado":
            self.transfer_ticks += 1
            t = min(1.,self.transfer_ticks*dt/1.8)
            for i,side in enumerate(self.sides):
                command[i,30*side:30*side+8] += command.new_tensor(self.bias[i])*(1.-t*t*(3.-2.*t))
