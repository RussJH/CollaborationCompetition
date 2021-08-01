# Collaboration Competition

---
 ## Overview

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

```
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
```

For this implementation we will be following an implementation of a Deep Deterministic Policy Gradients (DDPG) algorithm to train two agents to play table tennis against each other. 

<img src="Images/table-tennis.gif"  width=50%/>

 ## Getting Started
 To interact with this project there are a series of dependencies which you must download and install prior to running the [Tennis.ipynb](./Tennis.ipynb). Once you've installed the dependencies listed below, you can interact with the [Tennis Notebook](./Tennis.ipynb) which will contain several sections to describing the environment, training the agent, and then playing the trained agent in real time. 

 ### Dependencies
 In order to run the project you will need several things installed:
 * Python 3+
 * PyTorch
 * A DRNLD Configured environment [Instructions here](https://github.com/udacity/deep-reinforcement-learning#dependencies)
 * A way to run Jupiter Notebook like VSCode
 * A Unity Environment 
   * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

 Once you've downloaded the Environment, extract the contents in the root of the project (Example)
  * ./CollaborationCompetition/Tennis_Data/
  * ./CollaborationCompetition/UnityPlayer.dll
  * ./Tennis.exe
  
 #### Running the agent
 Once you've download all of the necessary dependencies, open the `Tennis.ipynb` file and follow the embedded instructions within.

### Details

See the description in the [Report.md](./Report.md)