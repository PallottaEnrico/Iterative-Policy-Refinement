# Iterative-Policy-Refinement
This repository contains the code for the Iterative Policy Refinement project.

The project involves the training of a policy for an agent in a gridworld environment and its consecutive refinement, aiming at avoiding unsafe behaviours.
In this work we follow this pipeline:
1. Train a Discrete Soft Actor Critic agent (without safety constraints)
2. Export the network model and load it in an optimization framework ([OMLT](https://github.com/cog-imperial/OMLT) & [Pyomo](https://github.com/Pyomo/pyomo))
3. Solve an optimization problem to find unsafe state transitions and store them
4. Retrain the agent with safety constraints avoiding unsafe state transitions found at the previous step
5. Run steps 3 and 4 iteratively until we don't observe any unsafe behaviour

You can find an example implementation of this pipeline in this [notebook](./iterative_policy_refinement.ipynb).

The environment we used in this example is better described [here](https://www.deepmind.com/blog/specifying-ai-safety-problems-in-simple-environments)

Unsafe behaviour:

![Unsafe Agent](images/agent.gif)

Safe behaviour:

![Safe Agent](images/safe_agent.gif)
