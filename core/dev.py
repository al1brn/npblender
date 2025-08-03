import numpy as np

class Test:
    def __init__(self, is_scalar=False):
        self.is_scalar = is_scalar

    def __len__(self):
        return 4
    
    def _broadcast_t(self, t):
        """
        Broadcast parameter t according to (B, T) convention.

        If self is_scalar:
            - t.shape == ()         → result shape: ()
            - t.shape == (T,)       → result shape: (T,)
        
        If self is a batch of B splines:
            - t.shape == ()         → result shape: (B,)
            - t.shape == (T,)       → result shape: (B, T)
            - t.shape == (1, T)     → result shape: (B, T)
            - t.shape == (B, T)     → result shape: (B, T)

        Returns
        -------
        - t: broadcasted array of shape (B, T)
        - res_shape: tuple to reshape result (e.g., (B, T, 3))
        """
        if self.is_scalar:
            B = 1
        else:
            B = len(self)

        t = np.asarray(t)
        if t.shape == ():
            t = np.broadcast_to(t, (B, 1))
            res_shape = () if self.is_scalar else (B,)
        else:
            try:
                t = np.broadcast_to(t, (B, t.shape[-1]))
            except ValueError:
                if self.is_scalar:
                    msg = (f"The shape of t {t.shape} is not compatible with a single spline. "
                          f"Authorized shapes are (1, n) or (n,)")
                else:
                    msg = (f"The shape of t {t.shape} is not compatible with {len(self)} spline. "
                          f"Authorized shapes are (n,) (1, n), or ({len(self)}, n)")

                    raise ValueError(msg)
                
            if self.is_scalar:
                res_shape = (t.shape[-1],)
            else:
                res_shape = t.shape

        return t, res_shape + (3,)
        


        
    def _broadcast_tOLD(self, t):
        """
        Broadcast parameter t according to (B, T) convention.

        If self is_scalar:
            - t.shape == ()         → result shape: ()
            - t.shape == (T,)       → result shape: (T,)
        
        If self is a batch of B splines:
            - t.shape == ()         → result shape: (B,)
            - t.shape == (T,)       → result shape: (B, T)
            - t.shape == (1, T)     → result shape: (B, T)
            - t.shape == (B, T)     → result shape: (B, T)

        Returns
        -------
        - t: broadcasted array of shape (B, T)
        - res_shape: tuple to reshape result (e.g., (B, T, 3))
        """
        t = np.asarray(t)
        B = 1 if self.is_scalar else len(self)
        res_shape = None

        if t.ndim == 0:
            t = np.full((B, 1), t)
            res_shape = () if self.is_scalar else (B,)

        elif t.ndim == 1:
            T = t.shape[0]
            if self.is_scalar:
                t = t[None]  # (1, T)
                res_shape = (T,)
            else:
                t = np.broadcast_to(t, (B, T))  # (B, T)
                res_shape = (B, T)

        elif t.ndim == 2:
            B2, T = t.shape
            if self.is_scalar:
                if B2 != 1:
                    raise ValueError(
                        f"The shape of t {t.shape} is not compatible with a scalar spline. "
                        f"Perhaps you want to use shape (1, {B2})"
                        )
                res_shape = (1, T)
            else:
                if B2 in [1, B]:
                    t = np.broadcast_to(t, (B, T))  # (B, T)
                    res_shape = (B, T)
                else:
                    raise ValueError(
                        f"The shape of t {t.shape} is not compatible with {B} splines. "
                        f"First dim must be 1 or the number of splines: (1, {B2}) or ({B}, {B2})."
                        )
        else:
            raise ValueError(f"Unsupported shape for t: {t.shape}")

        assert t.shape == (B, t.shape[-1]), f"t should have shape (B, T), got {t.shape}"

        print(f"DEBUG BC: {self.is_scalar=}, {t.shape=} {res_shape=}")

        return t, res_shape + (3,)
    
test = Test(True)

t, rs = test.newbc(.5)
assert t.shape == (1, 1)
assert rs == (3,)


t, rs = test.newbc([.5]*10)
assert t.shape == (1, 10)
assert rs == (10, 3)


test = Test(False)
t, rs = test.newbc(.5)
assert t.shape == (4, 1)
assert rs == (4, 3)

t, rs = test.newbc([.5]*10)
assert t.shape == (4, 10)
assert rs == (4, 10, 3)

t, rs = test.newbc(np.array([.5]*4)[:, None])
assert t.shape == (4, 1)
assert rs == (4, 1, 3)



