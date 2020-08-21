# Competition

From 24.08.2020 - TODO we are hosting a Kaggle competition about predicting future movements of other traffic participants.
This page serves as introduction point for it and gives additional information.

# Scoring
When taking part in the competition, you will be asked to submit predictions for a private test set (no ground truth is available),
and your solutions will be scored by Kaggle. The winning team will receive 30.000 USD as price!
As traffic scenes can contain a large amount of ambiguity and uncertainty, we encourage the submission of multi-modal predictions.
For scoring, we calculate the *negative log-likelihood* of these multi-modal predictions compared to the ground truth.
Let us take a closer look at this.
Assume, ground truth positions of a sample trajectory are ![equation](http://www.sciweavers.org/tex2img.php?eq=x_1%2C%20%5Cldots%2C%20x_N&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0),
and we predicted K hypotheses, represented by means ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cmu_1%5Ek%2C%20%5Cldots%2C%20%5Cmu_N%5Ek&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0).
In addition, we predict confidences c of these K hypotheses.
Using a standard Normal distribution, our likelihood is given by
![equation](http://www.sciweavers.org/tex2img.php?eq=%24p%28x_1%2C%20%5Cldots%2C%20x_N%7Cc%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cmu_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%29%20%3D%20%5Csum_k%20c%5Ek%20%5Cmathcal%7BN%7D%28x_1%2C%20%5Cldots%2C%20x_N%7C%5Cmu_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5CSigma%3D1%29%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
![equation](http://www.sciweavers.org/tex2img.php?eq=%3D%20%5Csum_k%20c%5Ek%20%5Cprod_t%20%5Cmathcal%7BN%7D%28x_t%7C%5Cmu_t%5Ek%2C%20%5Csigma%3D1%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
yielding the loss
![equation](http://www.sciweavers.org/tex2img.php?eq=L%20%3D%20-%20%5Clog%20p%28x_1%2C%20%5Cldots%2C%20x_N%7Cc%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cmu_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
![equation](http://www.sciweavers.org/tex2img.php?eq=%24%3D%20-%20%5Clog%20%5Csum_k%20e%5E%7B%5Clog%28c%5Ek%29%20%2B%20%5Csum_t%20%5Clog%20%5Cmathcal%7BN%7D%28x_t%7C%5Cmu_t%5Ek%2C%20%5Csigma%3D1%29%7D%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
![equation](http://www.sciweavers.org/tex2img.php?eq=%24L%20%3D%20-%20%5Clog%20%5Csum_k%20c%5Ek%20e%5E%7B-%5Cfrac%7B1%7D%7B2%7D%20%28%5Cmu_t%5Ek%20-%20x_t%29%5E2%7D%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

In our [code](https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py#L4) we use the form

![equation](http://www.sciweavers.org/tex2img.php?eq=%24L%20%3D%20-%20%5Clog%20%5Csum_k%20%20e%5E%7Bc%5Ek%20-%5Cfrac%7B1%7D%7B2%7D%20%28%5Cmu_t%20-%20x_t%29%5E2%7D%3D%20-%20%5Clog%20%5Csum_k%20%20e%5E%7B%5Ctexttt%7Berror%7D%7D%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

and for numeral stability further apply the [log-sum-exp trick](https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations):
Assume, we need to calculate the logarithm of a sum of exponentials:
![equation](http://www.sciweavers.org/tex2img.php?eq=%24LSE%28x_1%2C%20%5Cldots%2C%20x_n%29%20%3D%20%5Clog%28e%5E%7Bx_1%7D%20%2B%20%5Cldots%20%2B%20e%5E%7Bx_n%7D%29%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
Then, we rewrite this by substracting the maximum value x^* from each exponent, resulting in much increased numerical stability:
![equation](http://www.sciweavers.org/tex2img.php?eq=%24LSE%28x_1%2C%20%5Cldots%2C%20x_n%29%20%3D%20x%5E%2A%20%2B%20%5Clog%28e%5E%7Bx_1%20-%20x%5E%7B%2A%7D%7D%20%2B%20%5Cldots%20%2B%20e%5E%7Bx_n%20-%20x%5E%7B%2A%7D%7D%29%24%0A%24x%5E%2A%20%3D%20%5Ctexttt%7Bmax%7D%28x_1%2C%20%5Cldots%2C%20x_n%29%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
