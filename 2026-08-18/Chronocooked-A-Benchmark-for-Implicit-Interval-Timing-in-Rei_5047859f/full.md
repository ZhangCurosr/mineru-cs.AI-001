# Chronocooked: A Benchmark for Implicit Interval Timing in Reinforcement Learning Agents

Amrapali Pednekar Department of Information Technology IDLab, Ghent University - imec Amrapali.Pednekar@UGent.be

Yara Khaluf   
Computer Science Department   
Vrije Universiteit Amsterdam y.khaluf@vu.nl

Álvaro Garrido-Pérez Department of Information Technology IDLab, Ghent University - imec Alvaro.GarridoPerez@UGent.be

Pieter Simoens Department of Information Technology IDLab, Ghent University - imec Pieter.Simoens@UGent.be

## Abstract

This paper presents Chronocooked, a reinforcement learning (RL) benchmark suite for studying implicit interval timing in RL agents. Inspired by Overcooked, the suite comprises cooking scenarios that require temporal decision-making. The tasks and reward functions are designed such that temporal information is unobserved yet critical for optimal performance. The environment is intentionally kept simple to enable controlled experiments and support biologically plausible models. Evaluation metrics are designed to expose limitations in timing abilities of RL agents, and we report baselines using a non-recurrent, a recurrent, and a biologically plausible model. This work ultimately aims to underscore the need to incorporate time perception and temporal processing in artificial agents designed for human-robot interaction and deployment in time-dependent human societies.

## 1 Introduction

Time is a multifaceted dimension that shapes how humans perceive, coordinate, and make decisions in everyday life [57, 4, 49, 6]. Due to its ubiquity in our society, time should be a fundamental characteristic in artificial intelligence (AI) agents that may someday function alongside humans to accomplish various tasks. Although many AI systems excel in tasks that require temporal processing [51, 23], the focus is on final task performance, and it is unclear whether these systems develop an internal sense of time or merely rely on task-specific heuristics. While substantial research has focused on synchronization and coordination in human-robot interaction (HRI) tasks [1, 22, 40], it rarely considers time as an explicit dimension. These methods typically rely on observable cues from the environment or human partners to determine when to act. However, many real-world scenarios require actions solely based on elapsed time without any explicit cues. For example, we can infer that an oven timer is malfunctioning because the oven has been running longer than expected.

In psychology, this type of temporal decision-making is studied using interval timing tasks [38, 39, 7], which investigate how humans encode and perceive time and make decisions based on it. In this study, we introduce a reinforcement learning (RL) benchmark environment to study interval timing in AI agents. Rather than directly replicating psychology experiments, we introduce scenarios in which time-keeping is not the end goal but a means to achieve it. The reward functions and agent states do not carry any explicit information about time. Thus, the benchmark environment introduces implicit interval timing tasks, where time is an unobserved variable that must be inferred for optimal performance.

Each task is accompanied by a set of evaluation metrics designed to expose temporal limitations of models trained on it. These metrics broadly serve two purposes. First, they quantify the extent to which models replicate human timing biases. The goal is analogous to that in human-like robot behavior [28], where replicating such biases may enable smoother human-robot interaction. Second, they consolidate some key characteristics of a time-aware model as described in psychology and neuroscience literature [21, 24], and quantify a model's ability to capture them. Some of these characteristics include, ability to discriminate between different intervals, flexibly handle multiple intervals, generalize to out-of-distribution timing, achieve near-accurate timing performance and maintain temporal control. Finally, we report evaluation metrics for three types of model architectures, a non-recurrent model (CNN-MLP), a recurrent model (CNN-LSTM-MLP) and a biologically plausible model (CNN-CTRNN-MLP).

![](images/2ccf7ee7fa6c45127015e082c4a84e2e415409b89dc3247eb17db79c0e281544.jpg)  
(a) Bisection task

![](images/e1fb2bd78310381bd71a745412049343583839ef2fef32ef1169acd9e2e9eef3.jpg)  
(b) FI and Multi-timer

![](images/4422f95b745dc3dfcbaedac5843d1e2c8e8dbe57a913c0e62d86fe204dd18293.jpg)  
(c) Dual task

![](images/b16bd033a295b76cb5fcd3c27c147b057bb873094a256ba816dfc92335952547.jpg)  
(d) Uncertainty  
Figure 1: Implicit interval timing tasks: The bisection and fixed interval (FI) timing tasks are inspired from psychology research. All other tasks discussed in this study are modifications of these two tasks to evaluate different temporal decision-making characteristics.

## 2 Prior work

Timing research in AI can be broadly divided into two types: studying emergent timing and explicitly engineering time into models. Emergent timing has explored behavioral similarities in terms of qualitatively or quantitatively replicating human or animal timing data and conformity to different timing biases [12, 45, 29, 52, 46]. As well as, neural similarities in terms of biologically plausible neural activities such as time cells, ramping cells and oscillations[32, 41, 20, 37, 34, 35] as timekeeping mechanisms. These studies show scattered findings that depend on the task characteristics and reward functions, among other factor.

Studies related to explicitly engineering time into models show improved task performance and better replication of human timing behavior [25, 36, 33, 13, 27, 9, 10, 16]. These studies demonstrate how theoretical knowledge from psychology and neuroscience can inform the design of time-aware AI agents. However, these models are typically tested on specific tasks and specific timing aspects, and a standardized framework for evaluating and comparing them is lacking.

Timing models have also been independently developed in neuroscience and psychology to explain temporal behavior in humans and animals [21, 26, 5]. These models prioritize biological plausibility over task generalization, but could serve as useful components in AI systems seeking to incorporate a sense of time. Additionally, reinforcement learning (RL) has been linked to interval timing due to the temporal difference learning algorithm that has been successfully used to implement models of the basal ganglia [17, 43, 11]. Exploring this connection further may require integrating biologically plausible models into RL settings. The computational complexity of existing RL environments, however, can be an obstacle for such models.

Across all these research threads, a shared limitation is the absence of a simple, standardized benchmark. One prior RL environment, PsychLab[30], was designed to simulate psychology experiments including some timing tasks [12]. We take this work one step further by introducing tasks where timing is implicitly integrated in the scenarios. Moreover, for each task, we introduce evaluation metrics drawn from timing research literature that quantify both the replication of human timing biases and the key characteristics of a time-aware model. Finally, the environment is kept simple to facilitate testing of different types of models that may have a disadvantage due to complex environments.

## 3 Environment

The gymnasium [50] based environment is a simplified version of the Overcooked environment (Overcooked-AI [8]). The modified version consists of a single agent scenario with a 5X3 or 4X3 grid world featuring three counters, namely, an onion dispenser , an oven and a delivery counter. The environment includes two primary items, namely onions and soup, along with additional task-specific elements described in the subsequent sections (Figure 1).

The agent's goal is to prepare and deliver soup to the delivery counter. To do this, it has to pick up an onion from the dispenser (pick onion' phase), place it in the oven (put onion in oven' phase), and wait for the soup to cook (oven on’ or timing phase). The oven starts an internal (invisible) timer upon receiving the onion, tied to a predefined target duration (TD). The timing phase can vary slightly depending on the specific task. After TD or once the oven indicates ready' state), the agent can take the soup and deliver it to complete the episode. More details like the action and observation space can be found in the repository's documentation (https://anonymous.4open.science/r/Chronocooked-1BA6/).

## 4 Tasks

We take inspiration from two popular tasks in psychology: the bisection task [42, 3, 19] and the fixed interval timing task [15, 53, 48, 31, 14, 56]. The bisection task involves discrimination of intervals and is used to study temporal encoding and binary decision-making. Fixed interval timing is a prospective time estimation task that studies within-interval temporal control [31]. The remaining tasks are extensions designed to reflect real-world scenarios requiring either temporal bisection or duration estimation. These extensions enable targeted testing of specific timing capabilities in artificial agents.

## 4.1 Bisection task

The Bisection task is a widely used paradigm in timing research to study subjective timing in humans and animals [42, 3, 19, 54]. It involves two anchor durations, short (S) and long (L). Participants are trained on 50% short and 50% long intervals. During test, in addition to the short and long interval intermediate intervals are introduced and participants have to categorize each interval as S’ or L’. This tests temporal perception because, for intermediate durations, there is no objectively correct answer.

This task is implemented in the Chronocooked environment by using two delivery counters, one for ‘S’ and one for L’ (Figure 1a). After the agent places the onions in the oven, the oven runs for a target duration corresponding to either short or long duration with 50% probability. After the target duration, the oven changes to ready' state. The agent must then take the soup and deliver it to the correct counter. Interacting with the oven before it is ready results in no state change. The agent receives a reward of +1 for correct delivery, all other cases including all intermediate actions yield a reward of 0.

The agent is trained on two anchor durations, short and long, where difficulty can be varied by adjusting the ratio (L/S) or difference (L-S) between them. During testing, along with the anchor durations, the agent is evaluated on intermediate durations as well as durations slightly beyond the range of the anchor intervals.

## 4.1.1 Psychometric curve

Task performance is evaluated using a psychometric curve, a standard method in psychology. The percentage of long responses (P(long)) is plotted against the corresponding test (probe) duration (Figure 2). For an agent that performs the task successfully, the percentage of long responses should be 0% for durations at or below the short anchor and 100% for durations at or above the long anchor.

![](images/ae79a8d141bd7986d3e4972abd13b2b51a0724536402e47de686ec3713136bae.jpg)  
Figure 2: Psychometric curves of the three model types (rows) for the different anchors (columns). Y-axis shows percentage of long - P(long) and x-axis shows probe (test) durations along the the arithmetic mean (AM) and geometric mean (GM) of each anchor. The gray dots represent p(long) of each run (with 11 seeds each). The black dots is the average P(long) across all runs. The red line shows the sigmoid fit on the average P(long). The horizontal blue dotted line corresponds to P(long)=50%. The bisection points (BP) are shown in the respective plots.

The psychometric curve also reveals how the agent perceives intermediate durations, for which there is no objectively correct answer. Research consistently shows that in humans, the percentage of long responses increases systematically with increasing objective duration.

## 4.1.2 Bisection point

The bisection point (BP), the duration at which a participant is equally likely to respond long or short, is used to understand temporal encoding. A BP at the geometric mean (GM) of the anchor durations indicates logarithmic encoding of time, while a BP at the arithmetic mean (AM) suggests linear encoding [42]. Studies show that for humans, the BP typically falls near the GM [19, 3, 2], though some studies suggest it may fall closer to the AM [54]. Similarity to human temporal perception and encoding can thus be assessed by fitting a sigmoid to the agent's psychometric curve and computing the BP relative to the GM and AM.

## 4.1.3 Weber fraction

Weber's law is a well-established regularity in human timing [18], stating that the ability to discriminate between two durations is proportional to their magnitude. It is measured using the Weber fraction (WF), calculated by dividing the slope (DL) of the psychometric curve by the BP, which should remain approximately constant across different anchor durations. Additionally, studies show that the slope of the psychometric curve decreases as the ratio $( \mathrm { L } / \mathrm { S } )$ decreases, reflecting better task performance for difficult ratios. Assessing whether an agent conforms to Weber's law and exhibits similar sensitivity patterns provides a principled measure of its similarity to human timing biases. For example, replication of Weber's law would imply that the robot perceives changes in stimulus intensity in a manner similar to humans.

## 4.2 Fixed interval timing

The fixed interval (FI) timing task was originally developed to study reinforcement learning in animals [14]. The task requires the participants to respond after a fixed interval. Responses made before the interval ends, yield no reinforcement. While responses made after the interval receive a positive reward. The distribution of responses is analyzed to assess timing accuracy and within-interval temporal control.

This task is implemented in Chronocook using the oven duration as the fixed interval (Figure 1b). After placing the onion in the oven, an internal (invisible) timer associated with a target duration (TD)

(fixed interval) begins. Interacting with the oven before TD results in no state change. The agent can collect the soup at or any time after TD. It received a reward of +1 for successful soup delivery. All other cases and intermediate actions yield a reward of 0. The reward function is designed such that the agent's emergent timing behavior can be studied. There is an explicit lower bound on response time (agent cannot take the soup before TD) and an implicit upper bound, as the discounting factor incentivizes the agent to deliver the soup as soon as possible.

![](images/cc2a444f09f834e3ee474ac87e9cd037df4fd873e3af8ba6ff698aad420db31f.jpg)  
Figure 3: First oven check (FOC) distribution across model types and target durations. The standard deviation is shown as error bars on corresponding barplots.

## 4.2.1 First oven check

In addition to the average reward (a basic check of whether the agent can perform the task under different target durations), task performance can also be defined in terms of accuracy of timing.

To measure this, we define a metric called first oven check' (FOC). It represents the oven timer value at which the agent initiates the Interact' action with an onion carrying oven. If the oven timer is more than or equal to the target duration, agent can take the soup out and the corresponding oven time is recorded as the FOC for that trial. In contrast, if the oven timer is less than the target duration, the agent cannot take the soup out. In this case, if the agent consecutively continues to Interact' with the oven until it can take the soup out (i.e., until the target duration), the oven timer corresponding to the first Interact' action is recorded as the the FOC. However, if at least one of the consecutive actions is not Interact' the oven timer is not recorded. The intuition is that the agent checks the oven because it considers the target duration to be reached and continues to check it until it can get the soup out. Thus, FOC allows us to examine how agent policies vary across different agents in the timing phase.

FOC also reveals the degree of temporal control and the prevalence of unnecessary oven checks. An implicit sense of time helps agents avoid such redundant actions, which is particularly valuable in HRI scenarios where unnecessary actions may be costly. For instance, repeatedly interacting with the oven may interfere with the cooking process.

## 4.2.2 Peak interval timing

A variant of the fixed interval timing task, known as the peak interval (PI) task, omits reinforcement after the target duration and instead records the participant's response rate over time. In both humans and animals, the response rate is observed to increase gradually, peak near the target duration, and subsequently decline. To test this behavior in agents, the target duration is extended to 2× or 5× its training value to examine the agent's timing behavior beyond the trained interval.

## 4.2.3 Scalar property

The scalar property of timing, analogous to Weber's law discussed above, states that both the mean and variance of temporal responses scale linearly with the target duration [55]. It comprises two properties: the mean accuracy, which requires that the mean response time increases linearly and in most cases accurately with the target duration. And the scalar property of variance, which requires that the ratio of variance to mean remains approximately constant across durations.

To evaluate conformity to the scalar property, agents should be trained over a broad range of target durations. Conformity can then be assessed by examining whether the mean responses are close to the corresponding TDs, whether the mean-TD and mean-standard deviation (std) relationship is linear (quantified using the $R ^ { 2 }$ metric) and whether the coefficient of variation (CV = std/mean) is constant.

## 4.3 Multi-Timer

The motivation for this task comes from some limitations of timing models identified in the timing literature [21, 24]. Specifically, the ability to accurately time intervals, the capacity to store multiple intervals using a flexible time representation, and generalization to out-of-distribution durations.

This task is a modified version of the fixed interval timing task. Rather than allowing a flexible timing policy with an explicit lower bound, it introduces a timing buffer, the agent must retrieve the soup within a window of TD ± buffer duration to receive the full reward of +1. Retrieving the soup too early or too late results in a reduced reward (0.1). The task also supports multiple target durations simultaneously, with a distinct oven state for each TD (analogous to different colored indicator lights), providing the agent with a cue as to which duration must be timed.

## 4.3.1 Accurate timing

The agent performance is tested with a single timer with different buffer duration ranging from 0 to 75% of TD. This checks the accuracy of timing in agents. Increasing the target duration, can also reveal the memory capacity, agents with greater memory capacity are expected to perform better at longer target durations. This provides a controlled way to compare memory capacity across model architectures in the context of interval timing.

![](images/44427a829aa82ec1ed8e68ab5c211add8212bc4691ed443db73c759ccf260eac.jpg)  
Figure 4: Exact timer: Average reward as a function of different timing buffers for each target duration (TD). Shaded regions represent the standard deviation across runs. The red dotted line indicates the minimum possible reward (0.1) for a successful task completion.

## 4.3.2 Capacity to store multiple intervals

To test the number of TDs that can be accurately learned by an agent, the number of TD associated with the oven are varied. Target durations are kept short to isolate the factor of flexible temporal representation from that of long-term memory capacity. The buffer duration is 0, making accurate timing essential for maximum reward. An agent with a compact but efficient temporal representation may outperform a larger model with an inefficient one, making this metric a measure of how efficiently a model stores and retrieves multiple temporal durations.

## 4.3.3 Out-of-distribution timing

This metric evaluates generalization to Out-of-distribution (OOD) TDs. The agent is trained on a subset of TDs and tested on held-out durations. Since the value associated with the oven state scales with TD, it is interesting to examine whether the agent's timing behavior also scales with the oven state (for example, higher oven state value results in more waiting), even for TDs not encountered during training.An agent may not achieve high accuracy, but it can still exhibit meaningful temporal generalization through its timing behavior, for example, waits longer for higher OOD TD as compared to a lower one.

![](images/23f84ad24355f4c8ef882aec22c27334786dfccbc227ec40e7fa6e81fcc3f492.jpg)  
Target Durations (TD)

(a) Average reward by target duration  
![](images/6bda34ee10bf7e63830e245538f346f7a4f537f2bdb1395e15dda3d39bbe44c8.jpg)  
Target Durations (TD)  
(b) Average oven off time (including OOD durations)  
Figure 5: Multi-timer performance across metrics: reward and oven off time as a function of target duration.

## 5 Dual task and uncertainty task

These tasks are designed as combinations of the bisection and fixed interval timing tasks. In the dual task, an additional reward sink is introduced during the timing phase. The reward associated with the sink is kept small enough that the optimal policy requires the agent to leave the sink in time to reach the oven and collect the soup. Retrieving the soup too early or too late results in a reduced reward.

The uncertainty task introduces a manual oven-off button that transitions the oven to a ready state regardless of the target duration, but with a reduced reward. During training, the oven is assigned a small probability of never transitioning to the ready state autonomously, in which case the optimal policy is to use the manual oven-off button. It is of interest to examine how quickly agents trained on the standard timing task, without uncertainty, adapt to this condition, and whether a stronger sense of time facilitates faster adaptation.

## 6 Results

We report the results for three types of model architectures, CNN-MLP (non-recurrent), CNN-LSTM-MLP (recurrent) and CNN-CTRNN-MLP (biologically plausible recurrent). All models have been trained using the Proximal policy optimization (PPO) [47] algorithm, implemented using stable baselines3 [44]. For the CNN-CTRNN-MLP model some modifications were made to the original policy implementation of recurrent PPO. All recurrent architecture share the actor and critic network.

The recurrent models were trained with three memory sizes: 8,125,256 (referred as CTRNN8. CTRNN125, CTRNN256 and LSTM8, LSTM125, LSTM256 respectively). We report the results for memory size 125 in the main text and that of 8 and 256 in the Appendix. For a given task, all models have the same training hyperparameter settings. For each task, two runs of the corresponding models are conducted, and the reported results are aggregated over 11 or 12 seeds.

## 6.1 Bisection task

Figure 2 shows the psychometric curves of the three model types. LSTM125 performs better than its biologically plausible recurrent counterpart CTRNN125. However, for a higher memory size (256, shown in Appendix Figure 6), the two models have comparable performance. Interestingly, the weber fraction (WF) (Table 1a) is almost constant for CTRNN125, in contrast to LSTM125, though this consistency shows some discrepancy across other memory sizes (see Appendix Table 4. The sensitivity of LSTM125 appears to decrease with increasing task difficulty in most cases, whereas CTRNN125 shows less consistent sensitivity trends (Table 1b and Appendix Table 3). In conclusion, while both recurrent models are able to discriminate most intervals, including the out-of-distribution (OOD) test intervals, their similarity to human timing behavior cannot be conclusively established. Incorporating additional anchor durations and varying memory capacities may help strengthen some of the findings.

<table><tr><td rowspan="2">anchor (L,S)</td><td rowspan="2">L/S</td><td rowspan="2">L-S</td><td colspan="2">CTRNN125</td><td colspan="2">LSTM125</td></tr><tr><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td></tr><tr><td>6-8</td><td>1.33</td><td>2</td><td>NaN</td><td>NaN</td><td>0.04</td><td>0.05</td></tr><tr><td>4-8</td><td>2.00</td><td>4</td><td>NaN</td><td>NaN</td><td>0.09</td><td>0.03</td></tr><tr><td>6-12</td><td>2.00</td><td>6</td><td>0.05</td><td>0.06</td><td>0.07</td><td>0.01</td></tr><tr><td>4-12</td><td>3.00</td><td>8</td><td>0.04</td><td>0.05</td><td>0.13</td><td>0.02</td></tr><tr><td>6-18</td><td>3.00</td><td>12</td><td>0.04</td><td>0.00</td><td>0.15</td><td>0.04</td></tr><tr><td>4-18</td><td>4.50</td><td>14</td><td>0.06</td><td>0.00</td><td>0.14</td><td>0.06</td></tr></table>

(a) Weber Fraction (WF)

<table><tr><td rowspan="2">anchor (L,S)</td><td rowspan="2">L/S</td><td rowspan="2">L-S</td><td colspan="2">CTRNN125</td><td colspan="2">LSTM125</td></tr><tr><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td></tr><tr><td>6-8</td><td>1.33</td><td>2</td><td>NaN</td><td>NaN</td><td>0.29</td><td>0.30</td></tr><tr><td>4-8</td><td>2.00</td><td>4</td><td>NaN</td><td>NaN</td><td>0.50</td><td>0.12</td></tr><tr><td>6-12</td><td>2.00</td><td>6</td><td>0.46</td><td>0.53</td><td>0.62</td><td>0.10</td></tr><tr><td>4-12</td><td>3.00</td><td>8</td><td>0.26</td><td>0.33</td><td>1.04</td><td>0.28</td></tr><tr><td>6-18</td><td>3.00</td><td>12</td><td>0.48</td><td>0.02</td><td>1.69</td><td>0.23</td></tr><tr><td>4-18</td><td>4.50</td><td>14</td><td>0.59</td><td>0.08</td><td>1.19</td><td>0.42</td></tr></table>

(b) Sensitivity (DL)  
Table 1: Weber Fraction and sensitivity across models and anchor durations.

## 6.2 Fixed interval

In most cases, the CNN exhibits an FOC of zero, meaning it begins interacting with the oven immediately upon activation and continues until a state change occurs. While this strategy is not incorrect with respect to the final goal of soup delivery, it reflects a lack of time-keeping behavior. Since the models were not explicitly incentivized to exhibit timing behavior, it is noteworthy that introducing recurrence leads to emergent time-keeping-like behavior. Overall, LSTM125 produces FOC values closer to the target duration compared to CTRNN125, which tends to check the oven earlier (Figure 3). At a higher memory size, the performance of CTRNN256 improves (Appendix Figure 7).

With respect to the scalar property, while CTRNN125 exhibited an approximately constant Weber fraction in the bisection task, the coefficient of variation (CV) does not show conformity to the scalar property in the fixed interval task (Table 2a). Although the mean FOC for both recurrent models increases linearly with the target duration, consistent with the mean property of scalar timing, the relationship between mean and standard deviation is not linear in either models, indicating a failure to conform to the variance property of scalar timing (Table 2b).

<table><tr><td>TD</td><td>CTRNN125</td><td>LSTM125</td></tr><tr><td>4</td><td>NaN</td><td>0.42</td></tr><tr><td>6</td><td>NaN</td><td>0.41</td></tr><tr><td>9</td><td>1.14</td><td>0.33</td></tr><tr><td>12</td><td>2.34</td><td>0.27</td></tr><tr><td>18</td><td>0.27</td><td>0.11</td></tr><tr><td>24</td><td>0.05</td><td>0.24</td></tr></table>

<table><tr><td>Model</td><td>Mean vs TD (R2)</td><td>Mean vs Std. (R2)</td></tr><tr><td>CTRNN125</td><td>0.9</td><td>0.02</td></tr><tr><td>LSTM125</td><td>0.97</td><td>0.43</td></tr></table>

(a) Coefficient of Variation (CV = std/mean) for the different model types. A constant CV means conformity to scalar property.  
(b) Linear regression results assessing scalar timing properties across models.  
Table 2: Fixed interval scalar property

## 6.3 Multi-timing

For the exact timing case, LSTM125 generally outperforms the other two models (Figure 4), which is expected given its larger memory capacity. This is further supported by the results at the shortest target duration $\mathrm { ( T D } = 4 )$ , where LSTM125 and CTRNN125 perform equally well. Interestingly, at the shortest target duration $\mathrm { ( T D } = 4 )$ , the CNN, despite lacking recurrence, achieves an average reward above the minimum possible value of 0.1. This suggests that non-recurrent models can perform time-dependent tasks to a limited degree, likely due to the temporal difference learning underlying the RL training process [12]. However, as shown in Figure 4, this capacity is limited and does not generalize to longer target durations.

The performance for the multi-timer task is shown in Figure 5a. The x-axis represents the target duration on which the model was tested, and each plot title indicates the number and specific TDs used during training. Some additional out-of-distribution (OOD) durations are also shown. For example, in the ${ \mathrm { T i m e r s } } = 2$ case, the model was trained only on $\mathrm { T D } = 2$ and $\mathrm { T D } = 4 .$ , and all other TDs are OOD. While LSTM125 generally performs better than other models, none of the models perform all the tasks successfully.

To further investigate model behavior, we analyze the average oven-off time (Figure 5b), including additional OOD durations. As before, the plot title indicates the TDs used during training, with all others being OOD. In general, models settle at an average oven-off time corresponding to one of the trained TDs or a value in between. The number of distinct shifts in these curves serves as an indicator of the number of approximate timers stored by the model. In this respect, CTRNN125 generally shows more shifts than LSTM125. For example, in the Timers = 3 case (Figure 5b), CTRNN125 shows an average oven-off time of approximately 2 when presented with $\mathrm { T D } = 2$ , which increases to approximately 5 for $\mathrm { T D } = 3$ , indicating that it can distinguish between different oven states for the OOD durations. However, from $\mathrm { T D } = \bar { 3 }$ to $\mathrm { T D } = 8 ,$ no meaningful change in behavior is observed. At $\mathrm { T D } = 9$ , the model shifts back to an average oven-off time of 2, which indicates that it stores 2 timers. Overall, based on this analysis, none of the models demonstrate a strong capacity to store multiple intervals or generalize reliably to OOD durations.

## 7 Discussion

From the analysis of the different model types, we conclude that all models fall short in several aspects of temporal awareness. Some of these shortcomings are expected, as certain evaluations are inherently challenging given the models’ training conditions. For instance, the peak interval timing metric produced uniform oven-checking behavior across all models, which is unsurprising since the models were not trained for this scenario and the comparison is therefore unfair. Similarly out-of-distribution generalization is a known challenge for standard RL models, and meta-RL may be a more appropriate framework for evaluating OOD timing. We include these metrics for completeness, as a model with a genuine sense of time should be able to modulate its actions based on elapsed time even in novel conditions. It would also be interesting to evaluate models in which time is explicitly engineered, to assess their performance across the full suite of metrics. While no single model performs well across all metrics, the benchmark provides a standardized framework for quantifying temporal awareness from multiple angles.

This study has several limitations. The proposed metrics are not an exhaustive characterization of time-aware models but rather a starting point. Future work could expand the benchmark with additional tasks from the psychology literature, such as a temporal reproduction task, or extend it to multi-agent scenarios. Currently, all models are implemented within Stable Baselines3 (SB3), extending support to other libraries would broaden accessibility. Finally, evaluating a wider range of model and training hyperparameter settings would provide a more comprehensive view of the temporal capabilities of different model types.

## 8 References

## References

[1] Arash Ajoudani, Andrea Maria Zanchettin, Serena Ivaldi, Alin Albu-Schäffer, Kazuhiro Kosuge, and Oussama Khatib. Progress and prospects of the human-robot collaboration. Autonomous robots, 42(5):957–975, 2018.

[2] Lorraine G Allan. The location and interpretation of the bisection point. The Quarterly Journal of Experimental Psychology: Section B, 55(1):43–60, 2002.

[3] Lorraine G Allan and John Gibbon. Human bisection at the geometric mean. Learning and motivation, 22(1-2):39–58, 1991.

[4] Dan Ariely and Dan Zakay. A timely account of the role of duration in decision making. Acta psychologica, 108(2):187–207, 2001.

[5] Hamit Basgol, Inci Ayhan, and Emre Ugur. Time perception: A review on psychological, computational, and robotic models. IEEE Transactions on Cognitive and Developmental Systems, 14(2):301–315, 2021.

[6] Richard A Block and Ronald P Gruber. Time perception, attention, and memory: A selective review. Acta psychologica, 149:129–133, 2014.

[7] Catalin V Buhusi and Warren H Meck. What makes us tick? functional and neural mechanisms of interval timing. Nature reviews neuroscience, 6(10):755–765, 2005.

[8] Micah Carroll, Rohin Shah, Mark K Ho, Tom Griffiths, Sanjit Seshia, Pieter Abbeel, and Anca Dragan. On the utility of learning about humans for human-ai coordination. Advances in neural information processing systems, 32, 2019.

[9] Gabriele Cimolino and François Rivest. Latent time-adaptive drift-diffusion model. arXiv: Learning, 2021.

[10] Lorenzo Cominelli, Roberto Garofalo, and Danilo De Rossi. The influence of emotions on time perception in a cognitive system for social robotics. In 2019 28th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN), pages 1–6. IEEE, 2019.

[11] Nathaniel D Daw, Aaron C Courville, and David S Touretzky. Representation and timing in theories of the dopamine system. Neural computation, 18(7):1637–1677, 2006.

[12] Ben Deverett, Ryan Faulkner, Meire Fortunato, Gregory Wayne, and Joel Z Leibo. Interval timing in deep reinforcement learning agents. Advances in Neural Information Processing Systems, 32, 2019.

[13] Zafeirios Fountas, Anastasia Sylaidi, Kyriacos Nikiforou, Anil K. Seth, Murray Shanahan, Warrick Roseboom, Zafeirios Fountas, Anastasia Sylaidi, Kyriacos Nikiforou, Anil K. Seth, Murray Shanahan, and Warrick Roseboom. A predictive processing model of episodic memory and time perception. Neural Computation, 2022.

[14] Adam E Fox and Elizabeth GE Kyonka. Timing in response-initiated fixed intervals. Journal of the experimental analysis of behavior, 103(2):375–392, 2015.

[15] David Freestone and Fuat Balcı. Temporal decision-making: Common procedures and contemporary approaches. In Timing and Time Perception: Procedures, Measures, & Applications, pages 128–148. Brill, 2018.

[16] Felix A Gers, Nicol N Schraudolph, and Jürgen Schmidhuber. Learning precise timing with 1stm recurrent networks. Journal of machine learning research, 3(Aug):115–143, 2002.

[17] Samuel J Gershman, Ahmed A Moustafa, and Elliot A Ludvig. Time representation in reinforcement learning models of the basal ganglia. Frontiers in computational neuroscience, 7:194, 2014.

[18] John Gibbon. Scalar expectancy theory and weber's law in animal timing. Psychological review, 84(3):279, 1977.

[19] John Gibbon. On the form and location of the psychometric bisection function for time. Journal of Mathematical Psychology, 24(1):58–87, 1981.

[20] Vishwa Goudar and Dean V Buonomano. Encoding sensory and motor patterns as time-invariant trajectories in recurrent neural networks. Elife, 7:e31134, 2018.

[21] Joachim Hass and Daniel Durstewitz. Neurocomputational models of time perception. Neurobiology of interval timing, pages 49–71, 2014.

[22] Guy Hoffman. Evaluating fluency in human-robot collaboration. IEEE Transactions on Human-Machine Systems, 49(3):209–218, 2019.

[23] Max Jaderberg, Wojciech M Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio Garcia Castaneda, Charles Beattie, Neil C Rabinowitz, Ari S Morcos, Avraham Ruderman, et al. Human-level performance in 3d multiplayer games with population-based reinforcement learning. Science, 364(6443):859–865, 2019.

[24] Mehrdad Jazayeri and Michael N Shadlen. Temporal context calibrates interval timing. Nature neuroscience, 13(8):1020–1026, 2010.

[25] Md Rysul Kabir, James Mochizuki-Freeman, and Zoran Tiganj. Deep reinforcement learning with time-scale invariant memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1345–1354, 2025.

[26] Uma R Karmarkar and Dean V Buonomano. Timing in the absence of clocks: encoding time in neural network states. Neuron, 53(3):427–438, 2007.

[27] Maciej Komosiński and Adam Kupś. Time-order error and scalar variance in a computational model of human timing: simulations and predictions. Computational cognitive science, 2015.

[28] Rinaldo Kühne and Jochen Peter. Anthropomorphism in human-robot interactions: a multidimensional conceptualization. Communication Theory, 33(1):42–52, 2023.

[29] Aqeel Labash, Florian Stelzer, Daniel Majoral, and Raul Vicente Zafra. Emergence of adaptive circadian rhythms in deep reinforcement learning. In International Conference on Machine Learning, pages 18153–18170. PMLR, 2023.

[30] Joel Z Leibo, Cyprien de Masson d'Autume, Daniel Zoran, David Amos, Charles Beattie, Keith Anderson, Antonio García Castañeda, Manuel Sanchez, Simon Green, Audrunas Gruslys, et al. Psychlab: a psychology laboratory for deep reinforcement learning agents. arXiv preprint arXiv:1801.08116, 2018.

[31] Helga Lejeune and JH Wearden. The comparative psychology of fixed-interval responding: Some quantitative analyses. Learning and Motivation, 22(1-2):84–111, 1991.

[32] Dongyan Lin, Ann Zixiang Huang, and Blake Aaron Richards. Temporal encoding in deep reinforcement learning agents. Scientific Reports, 13(1):22335, 2023.

[33] Inês Lourenço, Rodrigo Ventura, and Bo Wahlberg. Teaching robots to perceive time: A twofold learning approach. Joint IEEE International Conference on Development and Learning and on Epigenetic Robotics, 2020.

[34] Michail Maniadakis and Panos Trahanias. Experiencing and processing time with neural networks. In Proc. 4th Int. Conf. on Advanced Cogn. Tech. and App, 2012.

[35] Michail Maniadakis and Panos Trahanias. Artificial agents perceiving and processing time. In 2015 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2015.

[36] Michail Maniadakis and Panos Trahanias. When and how-long: A unified approach for time perception. Front. Psychol., 2016.

[37] Valerio Mante, David Sussillo, Krishna V. Shenoy, and William T. Newsome. Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature, 2013.

[38] Matthew S Matell and Warren H Meck. Neuropsychological mechanisms of interval timing behavior. Bioessays, 22(1):94–103, 2000.

[39] Hugo Merchant and Victor De Lafuente. Introduction to the neurobiology of interval timing. Neurobiology of interval timing, pages 1–13, 2014.

[40] Stefanos Nikolaidis, Ramya Ramakrishnan, Keren Gu, and Julie Shah. Efficient model learning from joint-action demonstrations for human-robot collaborative tasks. In Proceedings of the tenth annual ACM/IEEE international conference on human-robot interaction, pages 189–196, 2015.

[41] Amrapali Pednekar, Alvaro Garrido, Pieter Simoens, and Yara Khaluf. Emergent time-keeping mechanisms in a deep reinforcement learning agent performing an interval timing task. In Artificial Life Conference Proceedings 37, volume 2025, page 51. MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info . .. , 2025.

[42] Trevor B Penney and Xiaoqin Cheng. Duration bisection: a user's guide. In Timing and time perception: Procedures, measures, & applications, pages 98–127. Brill, 2018.

[43] Elijah A Petter, Samuel J Gershman, and Warren H Meck. Integrating models of interval timing and reinforcement learning. Trends in cognitive sciences, 22(10):911–922, 2018.

[44] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of machine learning research, 22(268):1–8, 2021.

[45] Warrick Roseboom, Zafeirios Fountas, Kyriacos Nikiforou, David Bhowmik, Murray Shanahan, and Anil K Seth. Activity in perceptual classification networks as a basis for human subjective time perception. Nature communications, 10(1):267, 2019.

[46] Mostafa Safaie, Maria-Teresa Jurado-Parras, Stefania Sarno, Jordane Louis, Corane Karoutchi, Ludovic F Petit, Matthieu O Pasquet, Christophe Eloy, and David Robbe. Turning the body into a clock: Accurate timing is facilitated by simple stereotyped interactions with the environment. Proceedings of the National Academy of Sciences, 117(23):13084–13093, 2020.

[47] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[48] Burrhus Frederic Skinner. The behavior of organisms: An experimental analysis. BF Skinner Foundation, 2019.

[49] Sven Thönes and Kurt Stocker. A standard conceptual framework for the study of subjective time. Consciousness and cognition, 71:114–122, 2019.

[50] Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.

[51] Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. nature, 575(7782):350–354, 2019.

[52] Jing Wang, Yousuf El-Jayyousi, and Ilker Ozden. A neural network model for timing control with reinforcement. Frontiers in Computational Neuroscience, 16:918031, 2022.

[53]Steuart T Watson and Christine Griffes. Fixed interval schedule. Encyclopedia of Child Behavior and Development, pages 659–660, 2011.

[54] JH Wearden. Human performance on an analogue of an interval bisection task. The Quarterly Journal of Experimental Psychology Section B, 43(1b):59–81, 1991.

[55] John H Wearden and Helga Lejeune. Scalar properties in human timing: Conformity and violations. The Quarterly Journal of Experimental Psychology, 61(4):569–587, 2008.

[56] Michael D Zeiler and David G Powell. Temporal control in fixed-interval schedules. Journal of the Experimental Analysis of Behavior, 61(1):1–9, 1994.

[57] Philip G Zimbardo and John N Boyd. Putting time in perspective: A valid, reliable individualdifferences metric. In Time perspective theory; review, research and application: Essays in honor of Philip G. Zimbardo, pages 17–55. Springer, 2014.

## A Appendix

## B Bisection task

![](images/e36c0d2aea81447a2b00c7c7b49906bc6162bdf7c3d7573c9faaaf0ebcbfbc37.jpg)  
Figure 6: Psychometric curves of the three model types (rows) for the different anchors (columns). Y-axis shows percentage of long - P(long) and x-axis shows probe (test) durations along with the arithmetic mean (AM) and geometric mean (GM) of each anchor. The gray dots represent p(long) of each run (with 11 seeds each). The black dots is the average P(long) across all runs. The red line shows the sigmoid fit on the average P(long). The horizontal blue dotted line corresponds to P(long)=50%. The bisection points (BP) are shown in the respective plots.

<table><tr><td>anchor</td><td>L/S</td><td>L-S</td><td>CTRNN256 mean</td><td>std</td><td>CTRNN8 mean</td><td>std</td><td>LSTM256 mean</td><td>std</td><td>LSTM8 mean</td><td>std</td></tr><tr><td>6-8</td><td>1.33</td><td>2</td><td>Nan</td><td>NaN</td><td>Nan</td><td>NaN</td><td>Nan</td><td>Nan</td><td>NaN</td><td>NaN</td></tr><tr><td>4-8</td><td>2.00</td><td>4</td><td>0.68</td><td>0.61</td><td>NaN</td><td>NaN</td><td>0.46</td><td>0.06</td><td>Nan</td><td>NaN</td></tr><tr><td>6-12</td><td>2.00</td><td>6</td><td>0.28</td><td>0.20</td><td>NaN</td><td>NaN</td><td>0.50</td><td>0.07</td><td>Nan</td><td>NaN</td></tr><tr><td>4-12</td><td>3.00</td><td>8</td><td>0.63</td><td>0.18</td><td>Nan</td><td>NaN</td><td>0.70</td><td>0.24</td><td>Nan</td><td>NaN</td></tr><tr><td>6-18</td><td>3.00</td><td>12</td><td>0.73</td><td>0.45</td><td>NaN</td><td>NaN</td><td>1.30</td><td>0.95</td><td>NaN</td><td>NaN</td></tr><tr><td>4-18</td><td>4.50</td><td>14</td><td>0.46</td><td>0.06</td><td>NaN</td><td>NaN</td><td>1.75</td><td>0.04</td><td>NaN</td><td>NaN</td></tr></table>

Table 3: Difference Limen across models and anchor durations.

<table><tr><td>anchor</td><td>L/S</td><td>L-S</td><td>CTRNN256 mean</td><td>std</td><td>CTRNN8 mean</td><td>std</td><td>LSTM256 mean</td><td>std</td><td>LSTM8 mean</td><td>std</td></tr><tr><td>6-8</td><td>1.33</td><td>2</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td></tr><tr><td>4-8</td><td>2.00</td><td>4</td><td>0.29</td><td>0.34</td><td>NaN</td><td>NaN</td><td>0.08</td><td>0.01</td><td>NaN</td><td>NaN</td></tr><tr><td>6-12</td><td>2.00</td><td>6</td><td>0.03</td><td>0.02</td><td>NaN</td><td>NaN</td><td>0.06</td><td>0.01</td><td>NaN</td><td>NaN</td></tr><tr><td>4-12</td><td>3.00</td><td>8</td><td>0.11</td><td>0.03</td><td>NaN</td><td>NaN</td><td>0.09</td><td>0.03</td><td>NaN</td><td>NaN</td></tr><tr><td>6-18</td><td>3.00</td><td>12</td><td>0.06</td><td>0.04</td><td>NaN</td><td>NaN</td><td>0.14</td><td>0.09</td><td>NaN</td><td>NaN</td></tr><tr><td>4-18</td><td>4.50</td><td>14</td><td>0.07</td><td>0.00</td><td>NaN</td><td>NaN</td><td>0.18</td><td>0.02</td><td>NaN</td><td>NaN</td></tr></table>

Table 4: Weber fraction across models and anchor durations.

C Fixed interval

D Multi-timer

D.1 Exact timer

D.2 Multiple timers

![](images/275df516cab5f6b39abd1459b77b6fefd00844ce93cc28aede2e716accc9a6d4.jpg)  
Figure 7: First oven check (FOC) distribution across model types and target durations. The standard deviation is shown as error bars on corresponding barplots.

![](images/46fa4a3ac0b0146a8dc7820e8f1f65cde91bddbafa7c2ef8e8c4a762e1accd67.jpg)

![](images/e420288bdee67afc85973a75baee15e9bfb69c1c15b2f7c690b0a0f9040d1bce.jpg)  
Timing buffer

![](images/d28938fafe5a8e1c2791e488194db2d00ebe6fe7cb9e1329d41c9710a9bfcddb.jpg)

![](images/235450d89aa0ffd0da812637f79465e009c0a0909b8e6b21856353fd07ebf69a.jpg)

Figure 8: Exact timer: Average reward as a function of different timing buffers for each target duration (TD). Shaded regions represent the standard deviation across runs. The red dotted line indicates the minimum possible reward (0.1) for a successful task completion. Note: The plots are missing data from CTRNN.  
![](images/1655915f1e50c196da934c3a051327461ce6c4bc76585332025380fcf113a113.jpg)

![](images/d3560d24c574ef49403d2625a336a03406a7a1b702fbc6b0dee718d7fba27cf8.jpg)  
Target Durations (TD)

![](images/c19867d6494e307dd1b764f4a54ebd6119a63a718ff1f48bae06da70db4b1434.jpg)

![](images/aa3e3e32ab4ed60588f693512eabba8c28fcc53d5ec9eebe08e0db1ffd948ac8.jpg)

![](images/fd4fc18937d47e9b062086ee6d192dba12983af73879cd8c5c4f7d5cd950fdd0.jpg)

(a) Average reward by target duration  
![](images/56ce25cba2dbedd8c813f24f66705379171b1dc23859be009c0759def119061f.jpg)

![](images/11c39bc4acecb0cda3fe713dd4fe257fa4af89d52e98a14b3c3504483aee6c83.jpg)  
Target Durations (TD)

![](images/e5b83ad6d61ed6781b7d1ee3604bc8a48bb577f5d2fb3cbbb4aaa4b930ab52e1.jpg)  
(b) Average oven off time (including OOD durations)  
Figure 9: Multi-timer performance across metrics: reward and oven off time as a function of target duration.

![](images/6b5dcdc00e1e2ecc9818df511edef0119dc6b2787c462e16c5c45c97228a567a.jpg)  
Target Durations (TD)  
Figure 10: Average oven off time for different TDs