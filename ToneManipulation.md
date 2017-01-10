# Tone Manipulation
manipulate large-scale variations by defining a point-wise function modifying
the edge manipulation.

$$
\begin{eqnarray}
f_e(\alpha) &=& \beta \alpha, \ \text{where}, \beta \ge 0
\end{eqnarray}
$$

## Image Intensity
compute an intensity image $I_i = \dfrac{20I_r + 40I_g + I_b}{61}$ and
color ratios $(\rho_r, \rho_g, \rho_b) = \dfrac{1}{I_i}(I_r, I_g, I_b)$ and
apply filter on the log intensities $\log(I_i)$.

Here, set $\alpha \le 1$ and $\beta < 1$.
Remap the result $\log(I_i^{\rm '})$ by first offsetting its values to make its
maximum 0, then scaling them so that they cover a user-defined range.
Also, set max $99.5$th, min $0.5$th
