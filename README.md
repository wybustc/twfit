# twfit
Full spectra fitting with non-negative least-square method. 
This code models the pseudo-continuum of galaxy spectra with a non-negative linear combination 
of host stellar templates and an optional power law or black body component. To reduce the computational 
time of the fit, we used the six non-negative independent components (ICs) compressed from the library
of simple stellar populations (Bruzual & Charlot 2003) by Lu et al. 2006. We also took into account the 
potential dust attenuation effects by the host galaxy and the broadening of the stellar template caused 
by the stellar velocity dispersion.


By (Tinggui Wang) twang@ustc.edu.cn and (Yibo Wang) wybustc@mail.ustc.edu.cn 

