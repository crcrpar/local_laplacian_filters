# Detail Manipulation
## S-shaped function as $f_d$
__S-shaped function__ : $f_d(\Delta) = \Delta^{\alpha}$
In above function, $\alpha > 0$ is defined by users.
If $\alpha > 1$, smooth the details out, otherwise, increase contrast.

## Reduction of Noise Amplification
To avoid noise and artifacts from lossy image compression,
limit the smallest $\Delta$ amplified.

when $\alpha < 1$, $f_d(\alpha) = \tau \Delta^{\alpha} + (1-\tau)\Delta$.
$\tau$ is a smooth step function equal to 0
if $\Delta$ is less than 1\% of maximum intensity, 1 if it is more than 2\%.

*This function is used every computation.*
