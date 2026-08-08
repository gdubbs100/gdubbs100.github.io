---
overview: "Learning the basics of diffusion models"
---
# Diffusion Models
Lately I've been getting really interested in Stochastic Differential Equations (SDE) and stochastic calculus. This interest came about via an interest in control systems and how they could operate probabilistically. All of that is still a work in progress, however, along the way I bumped into diffusion models, which I found to be interesting. I spent a bit of time reading about them and building them, and I thought I'd write about that here.

## What are Diffusion Models?
Diffusion models are a type of generative model used in image and video generation. They perform sampling from complex probability distributions by learning an SDE which converts a target distribution to random noise and vice-versa. I thought this was an interesting approach to inference. Rather than simply learn a mathematical function that outputs a prediction, the learnt neural network learns parameters for an SDE, which then generates the desired output. 

The process works like this. Suppose we have a target distribution $p^{\text{Target}}_0$ and an initial distribution $p^{\text{noise}}_T$. The Diffusion Model will learn how to convert the target distribution to the initial, noisy distribution over a series of steps, $0..T$. Often the target is an image comprised of RGB values and the initial distribution is random noise following $N(0, I)$. Once the Diffusion Model has learnt to convert the target distribution to noise, the process can be reversed so that noise can be converted back to a sample from the target distribution. 

To do this, we set up a *probability path* the describes how we travel between the initial distribution and the final distribution. The *Gaussian Conditional Probability Path* [GenAi w sde] is a widely used path, where we have some value $\alpha_t$ that defines a noise schedule on the parameters of a Gaussian distribution:

$$

q(x_t | x_{t-1}) = N(\sqrt{\alpha_t}x_{t-1}, (1 - \alpha_t)I)

$$


Where $q(x_t \| x_{t-1})$ is the distribution that governs the target to noise transition. As we move from $0$ to $T$ we adjust the values of $\alpha$ such that at $T$, $\alpha_{T} = 0$ giving a standard Gaussian, while at $0$, $\alpha_{0} = 1$ giving the value of the target distribution. Note that the Gaussian Probability Path models the movement of the data between the initial and target distribution, but the Gaussian distribution at $0$ is not the target distribution itself. The entire transition from target to noise can be represented with the equation:


$$

q(x_{1:T} | x_0) = \prod_{t = 1}^{T} q(x_t | x_{t-1})

$$


Noising is simple to do as we can just choose $\alpha$ and simulate the above equation. But what is useful is learning how to convert noise to the target! To do this, we get a neural network and learn $p_{\theta}(x_{t-1} \| x_{t})$, parameterised by $\theta$ which moves in the opposite direction to $q$. We describe this process using:


$$

p(x_{0:T}) = p(x_{T}) \prod_{t = 1}^{T} p_{\theta}(x_{t-1} | x_{t})

$$

Training a model learn $\theta$ is done using the Evidence Lower Bound (ELBO). This derivation is described in great detail in [diffusion models], and I'll only repeat higlights here. The ELBO value we want to maximise is:


$$

\begin{aligned}
\log p(x) \geq \text{E}_{q(x_{1:T} | x_0)}[\log \frac{p(x_{0:T})}{q(x_{1:T} | x_0)}] = \text{E}_{q(x_1 | x_0)}[\log p_{\theta}(x_0 | x_1)] \\ - D_{KL}(q(x_T \| x_0) \|\ p(x_{T})) - \sum^{T}_{t=2}\text{E}[D_{KL}(q(x_{t-1} | x_t, x_0) \| p_{\theta}(x_{t-1} | x_t))]
\end{aligned}

$$

The most interesting part of this equation is the final summation term. Essentially, we want to minimise the KL-divergence between the distributions $q$ and $p_{\theta}$, which entails making them as similar as possible (to avoid confusion, note that $q(x_{t-1} \| x_t, x_0) = \frac{q(x_t \| x_{t-1}, x_0) q(x_{t-1} \| x_0)}{q(x_t\|x_0)}$ via bayes rule, letting us match the backward trajectory described above [diffusion models]). So conceptually what we are doing is training a neural network to learn the noise process that we apply to the target distribution. This lets us simulate the forward process from noise to target and generate samples from the target distribution.

## Denoising Diffusion Probabilistic / Implicit Models (DDPM / DDIM)
So how do we do this? DDPM and DDIM are two related approaches to diffusion models. Both train neural networks to predict the noise process based on a noised sample and time index [cite DDPM, DDIM]. It turns out that this approach is equivalent to minimising the KL-divergence shown above. 

DDPM uses a fixed noise schedule of values $\beta_t = 1 - \alpha_t$. Using a fixed schedule means that the variance of $q(x_{t-1} \| x_t, x_0)$ and $p_{\theta}(x_{t-1} \| x_t)$ are equal. Consequently, we can minmise the KL-divergence between them by simply minimising the mean-squared error of the means of these Gaussian distributions. The means of $q$ and $p_{\theta}$ are, respectively:

$$
\begin{align}
\mu_{q} = \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_{t})x_{0}}{1 - \bar{\alpha_t}} \\
\mu_{\theta} = \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_{t})\hat{x}_{\theta}}{1 - \bar{\alpha_t}}
\end{align}
$$

Where $\bar{\alpha}\_t = \prod^{t}\_{s=1} \alpha_{s}$, i.e. the accumulation of the noise schedule over several timesteps. This amounts to minimising a weighted MSE loss:

$$

L = \frac{1}{2\sigma_t^2} \frac{\bar{\alpha}_{t-1} (1 - \alpha_{t})^2}{(1 - \bar{\alpha_{t}})^2} [\|\hat{x}_{\theta}(x_{t}, t) - x_0\|^{2}_2]

$$

Which is minimising the loss between the neural net and the unnoised observation from the data. But further to this, we can define $x_t$ as:

$$
    x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_0
    
$$ 

This gives:

$$

x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_0}{\sqrt{\bar{\alpha}_t}}

$$

We can then insert this into our means for $q$ and $p_{\theta}$, giving:

$$

\begin{align}
\mu_{q} = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \epsilon_0 \\
\mu_{\theta} = \frac{1}{\sqrt{\alpha_t}} x_t - \frac{(1 - \alpha_t)}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \hat{\epsilon}_{\theta}(x_t, t)
\end{align}

$$

If we then want to minimise the MSE of the means of $q$ and $p_{\theta}$, this results in a similar but differently weighted loss:

$$

L = \frac{1}{2\sigma_t^2} \frac{(1 - \alpha_{t})^2}{(1 - \bar{\alpha_{t}})\alpha_t} [\|\hat{\epsilon}_{\theta}(x_{t}, t) - \epsilon_0\|^{2}_2]

$$

This derivation has been summarised from [diffusion models] - further details, particularly around how the noise schedule terms $\alpha_t$ are accumulated can be found there. 

In DDPM [cite], they create a variant of this loss ($L_{\text{simple}}$) which is given by:

$$

L_{\text{simple}} = \text{E}_{t, x_0, \epsilon}[\|\epsilon - \hat{\epsilon}_{\theta}(x_t, t)\|^{2}_2]

$$

This means we simply need to train a neural network to predict the random noise from a noised sample of data and time index $t$.

### Inference
To generate samples from $p^{target}$ we need to simulate the forward process. DDPM and DDIM differ in their approaches here. In DDPM we iterate through $T..1$ and use the following equation:

$$

x_{t - 1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1 - \bar{\alpha}_{t}}} \hat{\epsilon}_{\theta}(x_t, t)) + \sigma_t z

$$

Where $z \sim N(0, I)$ and $\sigma_t = \sqrt{1 - \alpha_t}$.This is a discretised SDE effectively, as $T \rightarrow \infty$ the forward equation becomes [song and ermon]:

$$

dX_t = -\frac{1}{2} \beta_t X_t dt + \sqrt{\beta_t} dW_t = -\frac{1}{2} (1 - \alpha_t) X_t dt + \sqrt{1 - \alpha_t} dW_t

$$

DDPM's simulation process is based on a the fact that the forwards and backwards equations are Markov Chains, that is, the value of $x_t$ depends only on $x_{t-1}$. DDIM extends DDPM to non-markovian equations. DDIM shares the training process with DDPM, however the forward simulation equation is altered to:

$$

x_{t-1} = \sqrt{\frac{\alpha_{t - 1}}{\alpha_t}} (x_t - \sqrt{1 - \alpha_t} \hat{\epsilon}_{\theta}(x_t, t)) + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \hat{\epsilon}_{\theta}(x_t, t) + \sigma_t z

$$

In DDIM the parameter $\sigma_t$ is actually a tunable parameter. When $\sigma_t = 0$ the process becomes deterministic, while when $\sigma_t = \sqrt{\frac{1 - \alpha_{t-1}}{1 - \alpha_t}} \sqrt{1 - \frac{\alpha_t}{\alpha_{t-1}}}$ then we recover DDPM [DDIM]. 

The advantage of DDIM's approach is that the value of $T$ we use is arbitrary. In contrast, when sampling using DDPM's approach, we need to use the same $T$ as used during training. This means we can use DDIM's sampling process to generate samples from $p^{target}$ faster.

## Examples
So thats a whole bunch of math. Writing that out helps me get a grip on what is going on with Diffusion Models, but key to understanding something is implementing it. To that end, I've implemented some models on toy data using the approaches described above. 

Rather than use images, I've used a sample of points in a 2d plane. I've created four different, simple target distributions -  a circle, a straight line, spiral and three circles. They're fairly simple distributions, with dot locations only varying along a single two dimensional shape. Images are shown below.

<div style="display: flex; justify-content: center; gap: 10px;">
  <figure style="width: 250px; text-align: center;">
    <img src="{{ '/images/20260801diffusion/sample_circle.png' | relative_url }}" alt="Circle dataset" style="max-width: 100%;">
    <figcaption><b>Circle</b></figcaption>
  </figure>

  <figure style="width: 250px; text-align: center;">
    <img src="{{ '/images/20260801diffusion/sample_line.png' | relative_url }}" alt="Line dataset" style="max-width: 100%;">
    <figcaption><b>Line</b></figcaption>
  </figure>
</div>

<div style="display: flex; justify-content: center; gap: 10px;">
  <figure style="width: 250px; text-align: center;">
    <img src="{{ '/images/20260801diffusion/sample_spiral.png' | relative_url }}" alt="Spiral dataset" style="max-width: 100%;">
    <figcaption><b>Spiral</b></figcaption>
  </figure>

  <figure style="width: 250px; text-align: center;">
    <img src="{{ '/images/20260801diffusion/sample_three_circle.png' | relative_url }}" alt="Three circles dataset" style="max-width: 100%;">
    <figcaption><b>Three Circles</b></figcaption>
  </figure>
</div>

Training a neural net on these simple patterns gets pretty good results with a small network (single hidden layer with 128 neurons) and around 1024 samples of simulated data. Below we look at the impacts of some of the simulation settings. The only distribution that was challenging was the three circle distribution, where the models were able to learn the rough location of the points, but struggled to learn the circle shape clearly.

I got Claude to find the best parameters for me for each shape. It set up its own little temp script and ran some experiments, minimsing the `chamfer distance` which I'd never heard of before, but its measured like this:

$$

d_{chamfer}(A,B) = \frac{1}{2}\left(\frac{1}{N}\sum_{a \in A}\min_{b \in B}|a-b| ;+; \frac{1}{M}\sum_{b \in B}\min_{a \in A}|b-a|\right)

$$

Essentially, it takes the average distance of all points in set $A$ to the closest point in set $B$, and the average distance of all points in set $B$ to the closest point in set $A$ and divides by 2. By optimising this metric, Claude found the best diffusion coefficient and number of steps for each shape, plotted in the table below:

| shape | η (diffusion_coef) | steps | chamfer distance |
|---|---|---|---|
| circle | 1.0 | 34 | 0.0182 |
| line | 0.0 | 34 | 0.0106 |
| spiral | 1.0 | 34 | 0.0362 |
| three_circle | 0.1 | 100 | 0.0538 |

We can see that these settings generate pretty nice shapes, although three circles is a bit of a struggle.

<div style="display: flex; justify-content: center;">
  <figure style="width: 100%; text-align: center;">
    <img src="{{ '/images/20260801diffusion/best_ddim_per_shape.gif' | relative_url }}" alt="Best DDIM settings per shape" style="display: block; margin: 0 auto; max-width: 100%;">
    <figcaption><b>Best DDIM settings per shape</b></figcaption>
  </figure>
</div>


### Value of $\eta$
Different values of $\eta$ don't appear to have a large impact on the final shape of the distribution, but you can see very different behaviour in the simulation process. With $\eta = 0$ the process is deterministic and you can see the dots simply travel to their destination. The process for the circle is interesting as the dots travel to the centre then to their location on the circle. 

With $\eta=0.5$ or using the DDPM setting, we see that the simulation process is much more noisy right up to the final moment when the points congeal into the target distribution.

<div style="display: flex; justify-content: center;">
  <figure style="width: 100%; text-align: center;">
    <img src="{{ '/images/20260801diffusion/grid_comparison.gif' | relative_url }}" alt="Comparison of eta values" style="display: block; margin: 0 auto; max-width: 100%;">
    <figcaption><b>Comparison across values of &eta;</b></figcaption>
  </figure>
</div>

### Value of $T$
Values of $T$ also don't really have a large impact on the final outcome. Lower values of $T$ seem to result in a less tight distribution. I think this makes sense as I'm not sure the DDIM reparameterisation made the simulation process exact for fewer smaller $T$, just significantly more accurate.

<div style="display: flex; justify-content: center;">
  <figure style="width: 100%; text-align: center;">
    <img src="{{ '/images/20260801diffusion/ddim_T_sweep_circle.gif' | relative_url }}" alt="Sweep of T values for the circle" style="display: block; margin: 0 auto; max-width: 100%;">
    <figcaption><b>Sweep of $T$ for the circle distribution</b></figcaption>
  </figure>
</div>

### Guidance
Above we trained one model for each shape, but we don't actually need to do this. With image generators, often you will be able to provide a text prompt and generate a corresponding image. This process is called guided generation [genAI w SDE]. Essentially the output is conditioned on the text input so that certain types of conditioning inputs generate samples from a certain kind of distribution.

It is simple to do this for my toy shape examples - we add a one-hot-encoded binary variable ($l$) representing the shape of a given sample so that our noise estimator $ \hat{\epsilon}_{\theta}(x, t, l)$ predicted class specific noise. The figure below shows that this works pretty well for this problem. For more complicated problems, however, this simple approach runs the risk that the model will ignore the effect of the label or guidance for some classes. As such, generally diffusion models use a different approach called "classifier-free guidance", which I have not investigated here.

<div style="display: flex; justify-content: center;">
  <figure style="width: 100%; text-align: center;">
    <img src="{{ '/images/20260801diffusion/multiclass_ddim.gif' | relative_url }}" alt="Multiclass DDIM guidance" style="display: block; margin: 0 auto; max-width: 100%;">
    <figcaption><b>Multiclass DDIM</b></figcaption>
  </figure>
</div>

## Conclusion
So thats a basic overview of diffusion models. They're kind of cool. It was fun to read into them and build them from scratch. 

## Bibliography
TODO


### Post-script on Claude use
With this project, I decided to relax a bit with my Claude use. I've found that lately, when writing code, I've been heavily using agentic methods. To be honest, I felt like my skills were dulling and I was enjoying making things less. For this project, I still used Claude, however, I spent most of the time implementing the algorithms myself. I really took the time to think about what I wanted the code to do and how best to code it. When I'd given things a go first, I'd then ask Claude for advice on how I could improve it and if my algorithm implementations were sound. That said, while most of the code is my own, I did get Claude to write the visualisations, just because making them nice and putting them into videos is a bit tedious.

I also engaged Claude to train models once I'd built the system, as well as tuning parameters. It was cool to just set it on a task and let it do it for me. Finding hyper-params / nursing models is pretty simple to do, but tedious, so it is great we can automate that. Although I do wonder how efficient it is to do this with "raw" Claude. I would have thought Bayesian Optimisation based approaches would be optimal in some respect, so using up an expensive token budget to do something that could be done with a script might be sub-optimal.