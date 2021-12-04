
# This is the introduction for the FlatTorus game.
**_If the math format doesn't render properly, you might consider using [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)_** 
***
## Configuration:

- This world is a flat torus, which is defined by $\mathbb{Z}/m\mathbb{Z}\times\mathbb{Z}/n\mathbb{Z}$. The world has a start location $[0,0]$ and an end location $\{ (x_0,y_0), \cdots, (x_n, y_n) \}$.

- Each square has a stationary stochastic reward(could be negative, deterministic ones are seen as a special case), the reward can accumulate with discount factor $\gamma$.  

- We can control the acceleration of jumper. The horizontal and vertical components of velocity are integers. Each time we can decide to change one component by no more than 1, and cost is determined by the change of kinetic energy, bounded below by 0. We have a speed limit for both components.

- Each square has a stationary stochastic integer drift, which is also subject to the drift limits.

- Each step has a time cost $C$.

- The goal is to find a policy to get to the end location from the start location with minimal cost.
