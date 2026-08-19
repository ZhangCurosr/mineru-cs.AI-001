# ASI-BENCH: AT THE DAWN OF ARTIFICIAL SUPERINTELLIGENCE

## Core Authors

Junwei Zhou<sup>5,†</sup>, Zhen Sun<sup>1,†</sup>, Binyu Li<sup>1</sup>, Jiangyu Zhou<sup>1</sup>, Yuexi Pan<sup>1</sup>, Hengyu Wang<sup>1</sup>, Honghe Ren<sup>1</sup>, Xiaohan Jia<sup>1</sup>, Xueyang Zhou<sup>1</sup>, Xiaoyu Cao<sup>1</sup>, Yongchao Chen<sup>1,∗</sup> <sup>†</sup>Equal contribution. <sup>∗</sup>Corresponding author.

## Contributors

Yuanning Feng<sup>1</sup>, Junhao Wu<sup>1</sup>, Cheng Zhang<sup>13</sup>, Sijia Chen<sup>10</sup>, Haoyu Xue<sup>1</sup>, Chengsong You<sup>1</sup>, Huan Wang<sup>1</sup>, Koutian Wu<sup>13</sup>, Peigan Gao<sup>9</sup>, Jiakun Wu<sup>1</sup>, Wenzhe Li<sup>1</sup>, Ergan Shang<sup>4</sup>, Qingyuan Zheng<sup>1</sup>, Jingjing Zhou<sup>1</sup>, Ruixuan Jia<sup>1</sup>, Yan Xu<sup>2</sup>, Hongrui Zhang<sup>7</sup>, Xiao-Han Ma<sup>9</sup>, Zhengxiang Cheng<sup>1</sup>, Yuexing Hao<sup>2</sup>, Liting Mai<sup>6</sup>, Xianglin Ji<sup>2</sup>, Wenjun Zhang<sup>8</sup>, Zhuofan Chen<sup>1</sup>, Yixiao Huang<sup>1</sup>, Chi Wang<sup>12</sup>, Wenyue Hua<sup>11</sup>, Yilun Hao<sup>2</sup>, Yuantao Zhai<sup>1</sup>, Ziyan Zhao<sup>1</sup>, Jingyan Xie<sup>3</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Massachusetts Institute of Technology <sup>3</sup>Harvard University <sup>4</sup>Carnegie Mellon University <sup>5</sup>University of Michigan <sup>6</sup>University of Illinois Urbana–Champaign <sup>7</sup>Boston University <sup>8</sup>University of Queensland <sup>9</sup>University of Science and Technology of China <sup>10</sup>Flatiron Institute <sup>11</sup>Microsoft Research <sup>12</sup>AG2 AI <sup>13</sup>Independent Researcher

![](images/37c2e4d7d4ac87fb809c14a8ac97be17c4fd98357f9c772c3834f74ddbb7b149.jpg)

![](images/2df00966637cb6512df2e3b73cd00b2f69dfe11810f06fdedd4ad2b918c9b63d.jpg)

![](images/8341e7381bbbd0fc0186bce68c8cdde5f89cbd6152863cdf4f6fa6ee1dcd6cbd.jpg)  
Figure 1: Overview of ASI-Bench. Left: B3 performance across agents. Right: scores from B1 to B4, where B1 provides full methods, B2 only the method name, B3 only the research goal and data, and B4 further adds distractors.

## Abstract

Artificial superintelligence (ASI) requires AI to move beyond mastering existing knowledge toward exploring the unknown, creating new knowledge, and turning new ideas into verifiable results. However, the capabilities of today’s AI systems are still largely built on learning, compressing, and applying existing human knowledge. Accordingly, existing benchmarks primarily test whether AI can produce correct answers based on learned knowledge, or whether it can complete tasks under extensive human guidance. We therefore introduce ASI-Bench, the first benchmark to jointly evaluate AI systems’ capabilities of innovative exploration and autonomous scientific execution across general research domains, and the first to progressively withdraw human methodological guidance within the same research project to test how far AI can proceed on its own. Built by over 40 experts with the cost of 31,000+ human hours, ASI-Bench contains 60 project-level research tasks across 11 scientific domains and progressively reduces methodological guidance to test whether AI can independently select methods, conduct research, and produce verifiable results. All tasks undergo expert review, AI-assisted auditing, sandbox execution, and scorer validation. Across 18 state-of-the-art agent–model configurations, the average score drops from 50.91 with full methodological guidance to 29.10 with only the method specified and 26.62 when agents must determine the method themselves. This sharp decline shows that current systems remain heavily dependent on human guidance and are still far from autonomously conducting end-to-end, project-level scientific research. ASI-Bench is open to the world. We invite researchers and builders everywhere to contribute new tasks, challenge the limits of today’s AI, and help accelerate humanity’s collective path toward artificial superintelligence at https://asibench.apexin.ai/submit.

## 1 Introduction

A central challenge on the path toward artificial superintelligence (ASI) is whether AI can move beyond mastering existing human knowledge to explore unfamiliar problems, develop new solutions, and turn them into verifiable results. Today’s AI systems derive much of their capability from learning, compressing, and applying the accumulated knowledge of humanity, and have made rapid progress in scientific reasoning, coding, data analysis, and agentic execution [1, 2, 3, 4, 5, 6, 7]. Yet existing evaluations largely test these capabilities either through problems with known answers or through tasks whose methods and procedures are substantially specified by humans. They therefore provide limited evidence about whether AI can autonomously conduct scientific research when both the problem and the path to a solution are open-ended. In this work, we ask a more direct question: how far can current AI systems independently explore and execute project-level scientific research as human methodological guidance is progressively withdrawn?

Built by over 40 experts with the cost of 31,000+ human hours, ASI-Bench is designed to make this question directly measurable. It consists of 60 project-level research tasks across 11 scientific domains, with each task built around a scientific objective, task-specific data, an executable environment, and verifiable research artifacts. To measure autonomy, ASI-Bench progressively reduces methodological guidance while keeping the research problem and evaluation criteria unchanged. In B1, the system receives complete methodological guidance; in B2, only the method is specified; in B3, the system must determine the method independently; and B4 introduces task-irrelevant information under the B3 setting to evaluate robustness. This design distinguishes following a complete procedure, translating a specified method into an executable workflow,

![](images/f5582baa2ab36be3fd2f75a1db0e57d97e8e636e3dc3183ccc498bf74c1a55bd.jpg)  
Figure 2: Performance comparison across HLE, SWE-bench, Terminal-Bench, and ASI-Bench.

and independently conducting the research, enabling a unified assessment of general intelligence, innovation, and autonomous execution.

We employ a multi-stage construction and validation process to ensure the scientific validity and evaluation reliability of ASI-Bench. Candidate research problems are first collected from traceable scientific sources, then screened and engineered into executable tasks by domain experts. Each task subsequently undergoes AI-assisted auditing and four rounds of human cross-review, covering scientific logic, task specifications, reference artifacts, scoring code, information leakage, and agent trajectories. Finally, every task is executed end-to-end in a sandbox environment to verify its reference solution and scoring behavior; tasks with scientific inconsistencies, unreliable evaluation, or unintended shortcuts are revised or excluded.

Table 1: Comparison of representative benchmarks for evaluating capabilities toward artificial superintelligence.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Task Setting</td><td colspan="4">Capabilities toward Autonomous Research</td></tr><tr><td>Cross-domain Generality</td><td>Method Autonomy</td><td>End-to-end Research</td><td>Guidance Gradient</td></tr><tr><td>Humanity&#x27;s Last Exam [5]</td><td>Academic QA</td><td>√</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Terminal-Bench [7]</td><td>Terminal tasks</td><td>一</td><td>△</td><td>△</td><td></td></tr><tr><td>ScienceAgentBench [8]</td><td>Scientific analysis</td><td>△</td><td>△</td><td>△</td><td></td></tr><tr><td>PaperBench [9]</td><td>Research replication</td><td></td><td>-△</td><td></td><td></td></tr><tr><td>MLE-Bench [10]</td><td>ML engineering</td><td>一</td><td></td><td></td><td></td></tr><tr><td>RE-Bench [11]</td><td>AI R&amp;D</td><td>一</td><td>V</td><td>&gt;&gt;&gt;</td><td></td></tr><tr><td>DiscoveryBench [12]</td><td>Scientific discovery</td><td>△</td><td>V</td><td>△</td><td></td></tr><tr><td>SciCode [13]</td><td>Scientific coding</td><td>V</td><td>△</td><td>一</td><td></td></tr><tr><td>ASI-Bench</td><td>Project-level research</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

✓ denotes explicit coverage; △ denotes partial or domain-limited coverage; – denotes that the capability is not explicitly evaluated.

Across 18 state-of-the-art Agent×Model configurations, average performance drops from 50.91 with full methodological guidance to 29.10 when only the method is specified and 26.62 when agents must determine the method themselves, revealing a substantial gap between scientific execution and autonomous discovery.

ASI-Bench provides a common reference point for measuring the transition from AI systems that primarily learn, compress, and apply existing human knowledge to systems capable of general intelligence, innovation, and autonomous execution. By revealing how far AI systems can progress as methodological guidance is withdrawn, ASI-Bench estab lishes a benchmark for tracking the emergence of capabilities that may define the dawn of artificial superintelligence.

## 2 Benchmark Design

## 2.1 Why Do We Need ASI-Bench?

Existing benchmarks have pushed different dimensions of advanced AI capability substantially forward. Humanity’s Last Exam (HLE) [5] tests frontier knowledge across a broad range of disciplines, while SciCode [13] focuses on research-level scientific coding. Long-horizon agent benchmarks such as Terminal-Bench [7] evaluate whether agents can use tools, resolve errors, and complete complex tasks. These benchmarks provide challenging tests of knowledge and execution, but typically operate with predefined objectives and evaluation criteria.

Other benchmarks focus on different components of scientific research. ScienceAgentBench [8] evaluates datadriven scientific analysis, DiscoveryBench [12] hypothesis-driven discovery, MLE-Bench [10] machine-learning engineering, and RE-Bench [11] open-ended AI R&D; PaperBench [9] focuses on research replication and computational reproducibility. As summarized in Table 1, these benchmarks cover different combinations of cross-domain generality, methodological autonomy, and end-to-end execution, each capturing an important dimension of advanced intelligence.

What remains missing is a benchmark that evaluates these dimensions jointly. ASI-Bench uses cross-domain projectlevel research to evaluate general intelligence, independent method selection to evaluate innovation, and end-to-end completion to evaluate autonomous execution. Its B1–B3 guidance gradient further measures whether these capabilities persist as human methodological guidance is progressively withdrawn. Together, these dimensions provide a more complete evaluation of intelligence than any individual capability alone.

## 2.2 How Is ASI-Bench Designed?

ASI-Bench consists of 60 project-level research tasks across 11 scientific domains, designed to evaluate how far AI can conduct scientific research as human methodological guidance is progressively withdrawn. Each project is evaluated under matched guidance conditions while keeping the underlying task, data, required outputs, and scoring criteria fixed.

Autonomous Research with Reduced Human Guidance. Each task in ASI-Bench is designed as a complex, project-level scientific investigation rather than an isolated question or a task with a predefined solution path. Starting from a research objective and task-specific data, agents must carry out a long-horizon, multi-stage research process that spans problem understanding, method selection, implementation, experimentation, failure diagnosis, iterative refinement, and result validation, ultimately producing verifiable scientific artifacts. Across the 60 tasks, completing these research processes involves more than 2,600 interaction turns and 2,400 execution steps, spanning over 35 hours of agent execution. Beyond their length, the tasks contain multiple interdependent and open-ended decision points: agents may need to choose among alternative methods, interpret intermediate results, recover from failed attempts, and revise subsequent actions accordingly. ASI-Bench therefore evaluates sustained end-to-end scientific investigation rather than isolated question answering or the execution of a fixed procedure.

![](images/68ae07fa0032fa5429e93b596c1faaf889a453dc0065f5225c185386c466aa5e.jpg)  
Figure 3: Representative project-level tasks in ASI-Bench across physics, astronomy, electrical engineering, and computer science.

To measure scientific autonomy, ASI-Bench progressively reduces human methodological guidance while keeping the research objective, data, and evaluation criteria fixed. A representative example is provided in Appendix D, where the same nonlinear two-dimensional dynamical system task is presented under B1–B4. In B1, the governing PDE, numerical formulation, and solver procedure are explicitly provided, where the agent mainly needs to implement and execute the prescribed approach. In B2, the full procedure is removed. The agent is only given methodological guidance about the class of PDE and suitable numerical approaches, and must turn this information into a working solution. In B3, even this methodological guidance is removed. The agent receives only the observed spatio-temporal data, the scientific objective, and the required outputs. It must determine the underlying model, choose an appropriate numerical method, implement it, and validate the resulting prediction. B4 retains the B3 setting but adds plausible yet task-irrelevant information, testing whether the agent can maintain its research direction under distraction. This progression shifts scientific responsibility from humans to AI, from executing a prescribed procedure to independently deciding how the research should be conducted.

Broad Scientific Coverage. ASI-Bench contains 60 project-level research tasks spanning 11 scientific domains: mathematics, physics, chemistry, biology, astronomy, materials science, earth science, medicine and biostatistics, computer science, robotics, and electrical engineering. These domains cover fundamental science, life science, computing, and engineering, requiring the same Agent×Model system to generalize across different data types, scientific methods, and validation criteria. This breadth tests whether autonomous research transfers across disciplines rather than remaining limited to a single domain or familiar workflow.

Rigorous Construction and Validation. ASI-Bench is built through a large-scale, iterative process rather than a single-pass collection. As shown in Figure 3, the process begins with more than 1,300 candidate research ideas collected from scientific sources. These candidates are repeatedly reviewed, revised, or removed before entering the benchmark. The construction process includes five review rounds, more than 1,100 review assignments, and over 2,000 task revisions. Reviewers examine the scientific formulation, task specification, B1–B4 information design, reference results, and evaluation criteria. They also check for information leakage and unintended shortcuts that could lead to high scores without correctly solving the task. Across this construction and validation process, more than 31,000 human-hours were invested. Each retained task is further validated through end-to-end execution in isolated sandboxes. Across development, more than 1,500 sandbox runs are conducted to verify runtime stability, reference reproducibility, artifact generation, and scoring consistency. Tasks with unresolved scientific errors, unstable execution, evaluation misalignment, or unintended solution paths are revised or excluded. After this process, 60 project-level tasks spanning 11 scientific domains remain in the final benchmark.

## 2.3 When Should ASI-Bench Be Used?

To demonstrate stronger intelligence. Improvements of several percentage points on conventional benchmarks do not necessarily indicate progress in higher-order intelligence, especially when the problems, answers, or solution procedures are already well specified. Improvements on ASI-Bench, particularly under low-guidance conditions where the system must determine how to approach the research problem itself, provide stronger evidence of progress in general intelligence, innovation, and autonomous execution. New foundation models, reasoning models, research agents, and general-purpose agents can therefore use ASI-Bench to demonstrate advances beyond knowledge recall or procedure following.

To test autonomous research. ASI-Bench is particularly suitable for systems claiming capabilities in AI for Science, AI Scientist, automated research, or highly challenging tasks. By progressively removing methodological guidance, it tests whether a system can continue a research project when a complete human-designed procedure is no longer provided. The evaluation can further reveal whether failures arise from insufficient knowledge, method selection, tool execution, result interpretation, error correction, or stability.

To compare models and agents. Different backbone models can be evaluated within the same agent framework, while the same model can be paired with different agents or harnesses to measure the contribution of system design. This allows researchers to distinguish improvements from the backbone model itself from those introduced by the agent design, and to identify under which guidance levels, scientific domains, and research stages the gains occur.

To measure frontier progress. Current performance on ASI-Bench remains low: across 18 state-of-the-art Agent×Model configurations, the average B3 score is only 26.62. The benchmark is therefore far from saturated and retains substantial room to distinguish future systems. Significant and reproducible gains, particularly under minimal methodological guidance, can provide a clear signal of progress toward more autonomous and higher-order intelligence. ASI-Bench can thus serve as a public evaluation and leaderboard for teams seeking to demonstrate such advances.

To expand the benchmark together. ASI-Bench is intended to grow with the scientific and AI communities. Researchers working at the frontier of different fields are often best positioned to identify research problems that are both scientifically meaningful and sufficiently challenging. We therefore invite scientists, engineers, model teams, agent developers, and benchmark researchers to contribute new problems, tasks, evaluation methods, and verified results, helping ASI-Bench remain challenging and relevant as AI capabilities advance.

## 3 Experiments

## 3.1 Main Results

We evaluate 18 representative Agent×Model configurations on 60 tasks in ASI-Bench. Table 2 reports their scores under B1–B4 together with the overall average.

Scientific Autonomy Remains Limited. Current systems remain far from reliable autonomous scientific discovery. Even the strongest configuration, Codex with GPT-5.6 Sol (ultra), achieves a B3 score of only 51.60 and is the sole evaluated system to exceed 50 when agents must independently select methods and construct research workflows. Stronger inference-time reasoning does improve this capability: increasing GPT-5.6 Sol from xhigh to ultra raises the B3 score from 40.86 to 51.60, a gain of 10.74 points. Yet this improvement also underscores how difficult scientific autonomy remains: substantial additional reasoning is required merely to reach moderate performance under reduced human guidance. The central challenge, therefore, is not simply whether AI can execute a research procedure once it is specified, but whether it can determine what procedure should be pursued in the first place. This distinction is critical for progress toward genuinely autonomous scientific research, where systems must move beyond following human-designed methodologies toward independently formulating, testing, and refining their own research strategies.

Dependence on Detailed Methodological Guidance. The results suggest that the main bottleneck is not method selection itself, but the ability to turn a scientific method into a complete research procedure. Average performance drops sharply from 50.91 in B1 to 29.10 in B2, a decrease of 21.82 points, when the detailed procedure is removed but the method remains available. By comparison, removing the method itself in B3 causes only a further 2.48-point drop, from 29.10 to 26.62. This large asymmetry indicates that current systems benefit far more from step-by-step methodological guidance than from being told which method to use. Performance remains nearly unchanged in B4 (26.99 versus 26.62 in B3), further showing that irrelevant contextual information has little effect relative to the loss of procedural guidance. Together, these results identify method operationalization, rather than method selection or distraction, as the primary bottleneck to autonomous scientific research.

Table 2: Main results on 60 project-level tasks without external tool access. Scores are macro-averaged over tasks and, unless otherwise specified, over three independent runs. Gray rows report ± one sample standard deviation across independent runs. The last three columns report B2−B1, B3−B1, and B4−B3, respectively, with differences computed independently for each run before aggregation. Bold values indicate the column-wise maximum result.
<table><tr><td rowspan="2">Harness</td><td rowspan="2">Backbone Model</td><td colspan="4">Scientific Score</td><td rowspan="2">Overall</td><td colspan="3">Diagnostic Metrics</td></tr><tr><td>B1</td><td>B2</td><td>B3</td><td>B4</td><td>B2-B1</td><td>B3-B1</td><td>B4-B3</td></tr><tr><td rowspan="6">Codex</td><td>GPT-5.5 (xhigh)</td><td>57.57</td><td>35.28</td><td>29.29</td><td>30.46</td><td>38.15</td><td>-22.29</td><td>-28.27</td><td>+1.16</td></tr><tr><td>± std.</td><td>±5.74</td><td>±2.23</td><td>±1.57</td><td>±1.55</td><td>±1.25</td><td>±7.37</td><td>±5.26</td><td>±3.10</td></tr><tr><td>GPT-5.6 Sol (xhigh)</td><td>62.75</td><td>42.96</td><td>40.86</td><td>40.74</td><td>46.83</td><td>-19.79</td><td>-21.89</td><td>-0.12</td></tr><tr><td>± std.</td><td>±3.05</td><td>±1.59</td><td>±2.35</td><td>±1.77</td><td>±1.10</td><td>±1.48</td><td>±5.27</td><td>±3.66</td></tr><tr><td>GPT-5.6 Sol (ultra)</td><td>71.78</td><td>49.57</td><td>51.60</td><td>50.41</td><td>55.84</td><td>-22.21</td><td>-20.18</td><td>-1.20</td></tr><tr><td>± std.</td><td>±0.46</td><td>±2.31</td><td>±3.65</td><td>±3.06</td><td>±1.58</td><td>±1.90</td><td>±3.21</td><td>±5.68</td></tr><tr><td rowspan="20">Claude Code</td><td>Claude Opus 5†</td><td>72.29</td><td>45.80</td><td>40.70</td><td>42.39</td><td>50.29</td><td>-26.49</td><td>-31.59</td><td>+1.69</td></tr><tr><td>single run</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kimi K3</td><td>55.79</td><td>35.24</td><td>27.78</td><td>29.57</td><td>37.09</td><td>-20.55</td><td>-28.02</td><td>+1.79</td></tr><tr><td>± std.</td><td>±4.86</td><td>±3.36</td><td>±2.02</td><td>±0.46</td><td>±1.74</td><td>±3.88</td><td>±5.64</td><td>±2.33</td></tr><tr><td>Claude Opus 4.8</td><td>52.48</td><td>36.25</td><td>31.73</td><td>29.59</td><td>37.51</td><td>-16.24</td><td>-20.76</td><td>-2.14</td></tr><tr><td>± std.</td><td>±6.27</td><td>±4.57</td><td>±3.44</td><td>±1.51</td><td>±0.79</td><td>±9.26</td><td>±8.71</td><td>±4.46</td></tr><tr><td>GLM-5.3</td><td>63.05</td><td>35.45</td><td>33.09</td><td>35.01</td><td>41.65</td><td>-27.60</td><td>-29.96</td><td>+1.92</td></tr><tr><td>± std.</td><td>±1.40</td><td>±3.76</td><td>±1.36</td><td>±3.40</td><td>±1.35</td><td>±4.61</td><td>±1.30</td><td>±4.74</td></tr><tr><td>GLM-5.2</td><td>54.01</td><td>30.81</td><td>30.29</td><td>27.27</td><td>35.59</td><td>-23.20</td><td>-23.72</td><td>-3.02</td></tr><tr><td>± std.</td><td>±1.65</td><td>±2.25</td><td>±1.96</td><td>±3.98</td><td>±0.95</td><td>±2.54</td><td>±3.56</td><td>±2.04</td></tr><tr><td>DeepSeek V4 Flash</td><td>51.85</td><td>26.49</td><td>24.84</td><td>24.41</td><td>31.90</td><td>-25.36</td><td>-27.01</td><td>-0.43</td></tr><tr><td>± std.</td><td>±1.71</td><td>±1.24</td><td>±2.21</td><td>±0.95</td><td>±0.47</td><td>±0.82</td><td>±3.56</td><td>±1.28</td></tr><tr><td>Kimi K2.7</td><td>44.43</td><td>23.84</td><td>19.75</td><td>21.34</td><td>27.34</td><td>-20.59</td><td>-24.68</td><td>+1.58</td></tr><tr><td>± std. MiniMax M3</td><td>±1.90</td><td>±0.40</td><td>±0.74</td><td>±2.26</td><td>±0.61</td><td>±2.29</td><td>±2.07</td><td>±2.99</td></tr><tr><td></td><td>43.53</td><td>20.86</td><td>20.61</td><td>21.21</td><td>26.55</td><td>-22.68</td><td>-22.92</td><td>+0.60</td></tr><tr><td>± std.</td><td>±3.60</td><td>±3.29</td><td>±1.02</td><td>±2.85</td><td>±1.55</td><td>±2.31</td><td>±3.20</td><td>±3.87</td></tr><tr><td>DeepSeek V4 Pro</td><td>42.97</td><td>18.79</td><td>19.20</td><td>18.94</td><td>24.98</td><td>-24.18</td><td>-23.77</td><td>-0.26</td></tr><tr><td>± std.</td><td>±0.46</td><td>±2.88</td><td>±0.22</td><td>±1.23</td><td>±0.57</td><td>±2.61</td><td>±0.68</td><td>±1.28</td></tr><tr><td>MiMo V2.5 Pro</td><td>41.70</td><td>18.49</td><td>16.49</td><td>16.33</td><td>23.25</td><td>-23.22</td><td>-25.21</td><td>-0.16</td></tr><tr><td>± std.</td><td>±1.63</td><td>±2.27</td><td>±0.07</td><td>±0.39</td><td>±1.05</td><td>±1.20</td><td>±1.56</td><td>±0.31</td></tr><tr><td rowspan="4">Kimi Code</td><td>Kimi K3</td><td>56.16</td><td>34.05</td><td>28.15</td><td>26.53</td><td>36.22</td><td>-22.11</td><td>-28.01</td><td>-1.62</td></tr><tr><td>± std.</td><td>±4.86</td><td>±5.08</td><td>±0.84</td><td>±0.99</td><td>±2.22</td><td>±2.34</td><td>±4.56</td><td>±1.45</td></tr><tr><td>Kimi K2.7</td><td>29.99</td><td>17.01</td><td>15.65</td><td>16.22</td><td>19.72</td><td>-12.98</td><td>-14.34</td><td>+0.58</td></tr><tr><td>± std.</td><td>±3.30</td><td>±0.10</td><td>±1.34</td><td>±0.85</td><td>±1.19</td><td>±3.35</td><td>±1.98</td><td>±1.56</td></tr><tr><td rowspan="2">MiMo Code</td><td>MiMo V2.5 Pro</td><td>27.73</td><td>11.33</td><td>11.50</td><td>14.11</td><td>16.17</td><td>-16.40</td><td>-16.24</td><td>+2.61</td></tr><tr><td>± std.</td><td>±4.33</td><td>±1.40</td><td>±1.36</td><td>±3.19</td><td>±1.39</td><td>±5.40</td><td>±5.37</td><td>±4.11</td></tr><tr><td rowspan="4">OpenHands</td><td>DeepSeek V4 Flash</td><td>50.60</td><td>24.89</td><td>21.91</td><td>24.99</td><td>30.60</td><td>-25.71</td><td>-28.69</td><td>+3.09</td></tr><tr><td>± std.</td><td>±2.24</td><td>±5.27</td><td>±1.75</td><td>±4.27</td><td>±3.05</td><td>±3.27</td><td>±1.53</td><td>±4.43</td></tr><tr><td>DeepSeek V4 Pro</td><td>37.78</td><td>16.66</td><td>15.78</td><td>16.28</td><td>21.63</td><td>-21.12</td><td>-22.00</td><td>+0.50</td></tr><tr><td>± std.</td><td>±2.28</td><td>±2.84</td><td>±1.91</td><td>±2.52</td><td>±0.75</td><td>±3.00</td><td>±3.42</td><td>±4.31</td></tr><tr><td colspan="2">All Models (Mean)</td><td>50.91</td><td>29.10</td><td>26.62</td><td>26.99</td><td>33.41</td><td>-21.82</td><td>-24.29</td><td>+0.36</td></tr></table>

Gray rows report ± one sample standard deviation across independent runs. For the diagnostic metrics, differences are first computed within each run and their standard deviations are then calculated across runs.  
<sup>†</sup> This result is obtained from a single run; therefore, its standard deviation cannot be estimated. All other results report the mean and standard deviation over three independent runs.

Harnesses Shape Model’s Capability. The harness can substantially change the capability expressed by the same underlying model. MiMo V2.5 Pro, for example, improves from 16.17 with MiMo Code to 23.25 with Claude Code, while Kimi K2.7 rises from 19.72 with Kimi Code to 27.34 with Claude Code. However, this effect is not uniform: Kimi K3 shows only a small difference between Kimi Code and Claude Code (36.22 vs. 37.09). These results show that scientific capability cannot be attributed to the backbone model alone; it emerges from the interaction between the mode and the harness through which it reasons, executes, and organizes research. This has an important implication for both evaluation and system design: comparing models without accounting for their harnesses can obscure where capability actually comes from, while progress toward autonomous scientific research may depend as much on improving the surrounding research system as on scaling the model itself.

## 3.2 Computational Cost

Figure 4 reports the average token consumption and execution time under different levels of methodological guidance, together with the cost profiles of individual Agent×Model combinations.

![](images/2854faf1dbf0d806c7ed839906bca40c6b3a76eb67bfe74297fa80ea5e5df98d.jpg)

![](images/aa43eb2f083054b4b1acd59c76e7150771575383cfddd43c1e7b616c1f5571b5.jpg)

![](images/5d9e4adc00058a26a06d349d401477683e2a1fb41334c32307795ed6af086dac.jpg)  
Figure 4: Computational cost under different levels of methodological guidance and cost–performance trade-offs across Agent×Model combinations. (a–b) Average per-task token consumption and execution time under B1–B4. (c) Relationship between per-run monetary cost and B3 scientific score across evaluated systems. Colors denote backbone models, while marker shapes denote agent harnesses.

Complete Guidance Reduces Compute Cost. Computational cost depends not only on how much methodological guidance is provided, but also on how complete that guidance is. B1, which specifies the method, implementation steps, and parameter choices, is the least expensive setting, requiring only 4.35M tokens and 37.8 minutes per task on average. Removing this procedural guidance increases the cost substantially: B3 and B4 consume 25% and 30% more tokens than B1 and require 22% and 18% more execution time, respectively, reflecting the additional exploration and iteration needed when agents must determine how to conduct the research themselves. More strikingly, B2 is the most expensive setting, requiring 6.91M tokens and 49.7 minutes per task—59% more tokens and 32% more time than B1. Providing only the method name therefore does not necessarily simplify research; instead, it can leave agents constrained by a prescribed direction while still requiring them to reconstruct the missing procedural details. This result reveals an important asymmetry in human guidance: complete guidance can sharply reduce search and implementation cost, whereas incomplete guidance may introduce additional overhead.

Higher Cost Does Not Guarantee Better Performance. Scientific performance varies substantially across configurations with similar or even very different computational budgets. Codex with GPT-5.6 xhigh achieves a B3 score of 40.86% at approximately \$684 per run, closely matching Claude Opus 5 with Claude Code at 40.70%, despite costing only about one quarter as much as its \$2,728 per-run cost. Spending more can still improve absolute performance: Codex with GPT-5.6 Ultra reaches the highest B3 score of 51.60% at approximately \$1,550 per run. However, the large cost differences among configurations with comparable performance show that additional spending does not translate directly into stronger scientific capability. For practical autonomous research, system selection therefore involves a clear trade-off: GPT-5.6 xhigh offers the strongest cost–performance balance, whereas GPT-5.6 Ultra is preferable when maximizing scientific performance matters more than efficiency.

## 4 Conclusion

Beyond a single leaderboard, ASI-Bench is intended to serve as shared research infrastructure for the scientific and AI communities. Its B1–B4 structure and domain-level analyses enable controlled comparisons across models and agents, while revealing where progress occurs and where human methodological guidance remains necessary. Yet no fixed benchmark—and no single research team—can fully represent the breadth of difficult, meaningful, and verifiable problems that future AI systems must confront. The continued value of ASI-Bench therefore depends on collective participation from researchers working across disciplines and at the frontier of model development.

We invite scientists, engineers, model teams, agent researchers, and benchmark developers around the world to contribute research problems, executable tasks, evaluation methods, reviews, and verified results; to test new systems against the benchmark; and to challenge the benchmark itself as AI capabilities advance. By building ASI-Bench as an open, rigorous, and continually evolving community standard, we can establish a shared foundation for identifying genuine advances in intelligence—and collectively accelerate progress toward artificial superintelligence.

## References

[1] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating language agents on machine learning experimentation. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 20271–20309, 2024.

[2] Ruochen Li, Teerth Patel, Qingyun Wang, and Xinya Du. MLR-Copilot: Autonomous machine learning research based on large language models agents. arXiv preprint arXiv:2408.14033, 2024.

[3] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, et al. Agent laboratory: Using llm agents as research assistants. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 5977–6043, 2025.

[4] Jiakang Yuan, Xiangchao Yan, Bo Zhang, Tao Chen, Botian Shi, Wanli Ouyang, Yu Qiao, Lei Bai, and Bowen Zhou. Dolphin: Moving towards closed-loop auto-research through thinking, practice, and feedback. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 21768–21789, 2025.

[5] Center for AI Safety, Scale AI, and HLE Contributors Consortium. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649:1139–1146, 2026.

[6] Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, Niklas Lauffer, Andrew Park, Nitin Pasari, Chetan Rane, Karmini Sampath, Maya Krishnan, Srivatsa Kundurthy, Sean Hendryx, Zifan Wang, Chen Bo Calvin Zhang, Noah Jacobson, Bing Liu, and Brad Kenstler. SWE-Bench Pro: Can AI agents solve long-horizon software engineering tasks? arXiv preprint arXiv:2509.16941, 2025.

[7] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, et al. Terminal-Bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868, 2026.

[8] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, et al. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery. arXiv preprint arXiv:2410.05080, 2024.

[9] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, et al. Paperbench: Evaluating ai’s ability to replicate ai research. arXiv preprint arXiv:2504.01848, 2025.

[10] Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, et al. MLE-bench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095, 2024.

[11] Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, et al. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. arXiv preprint arXiv:2411.15114, 2024.

[12] Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, et al. DiscoveryBench: Towards data-driven discovery with large language models. arXiv preprint arXiv:2407.01725, 2024.

[13] Minyang Tian, Luyu Gao, Shizhuo Dylan Zhang, Xinan Chen, Cunwei Fan, Xuefei Guo, et al. SciCode: A research coding benchmark curated by scientists. arXiv preprint arXiv:2407.13168, 2024.

[14] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature Machine Intelligence, 6:525–535, 2024.

[15] Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624:570–578, 2023.

[16] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

[17] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The AI scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

[18] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, et al. Accelerat ing scientific discovery with Co-Scientist. Nature, 655:487–496, 2026.

[19] Ludovico Mitchener, Angela Yiu, Benjamin Chang, Mathieu Bourdenx, Tyler Nadolski, Arvis Sulovari, et al. Kosmos: An AI scientist for autonomous discovery. arXiv preprint arXiv:2511.02824, 2025.

[20] Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Yamada, Shengran Hu, Jakob Foerster, David Ha, and Jeff Clune. Towards end-to-end automation of AI research. Nature, 651:914–919, 2026.

[21] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof Q&A benchmark. arXiv preprint arXiv:2311.12022, 2023.

[22] Ken Gu, Ruoxi Shang, Ruien Jiang, Keying Kuang, Richard-John Lin, Donghe Lyu, et al. BLADE: Benchmarking language model agents for data-driven science. arXiv preprint arXiv:2408.09667, 2024.

[23] Liqiang Jing, Zhehui Huang, Xiaoyang Wang, Wenlin Yao, Wenhao Yu, Kaixin Ma, et al. DSBench: How far are data science agents from becoming data science experts? arXiv preprint arXiv:2409.07703, 2024.

[24] Haokun Liu, Sicong Huang, Jingyu Hu, Yangqiaoyu Zhou, and Chenhao Tan. Hypobench: Towards systematic and principled benchmarking for hypothesis generation. arXiv preprint arXiv:2504.11524, 2025.

[25] Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, and Arvind Narayanan. Core-bench: Fostering the credibility of published research through a computational reproducibility agent benchmark. arXiv preprint arXiv:2409.11363, 2024.

[26] Christine Ye, Sihan Yuan, Suchetha Cooray, Steven Dillmann, Ian L. V. Roque, Dalya Baron, et al. Replication-Bench: Can AI agents replicate astrophysics research papers? arXiv preprint arXiv:2510.24591, 2025.

[27] Zhen Wang, Fan Bai, Zhongyan Luo, Jinyan Su, Kaiser Sun, Xinle Yu, et al. Fire-bench: Evaluating agents on the rediscovery of scientific insights. arXiv preprint arXiv:2602.02905, 2026.

[28] Wanghan Xu, Shuo Li, Tianlin Ye, Qinglong Cao, Yixin Chen, Hengjian Gao, et al. ResearchClawBench: A benchmark for end-to-end autonomous scientific research. arXiv preprint arXiv:2606.07591, 2026.

[29] Tianyu Liu, Allen Xin Wang, Antonia Panescu, Lisa Xinyi Chen, Wenxin Long, Xinyu Wei, Yueqian Jing, Ziyao Zeng, et al. Benchmarking AI agents for addressing scientific challenges across scales. arXiv preprint arXiv:2606.12736, 2026. SciAgentArena.

[30] Peter Jansen, Marc-Alexandre Côté, Tushar Khot, Erin Bransom, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, et al. DISCOVERYWORLD: A virtual environment for developing and evaluating automated scientific discovery agents. arXiv preprint arXiv:2406.06769, 2024.

[31] Jonathan Bragg, Mike D’Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, Sergey Feldman, et al. Astabench: Rigorous benchmarking of ai agents with a scientific research suite. arXiv preprint arXiv:2510.21652, 2025.

[32] Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, et al. Scienceboard: Evaluating multimodal autonomous agents in realistic scientific workflows. arXiv preprint arXiv:2505.19897, 2025.

[33] A. Lew, Y. Cao, and Markus J. Buehler. Projectionbench: Evaluating scientific hypothesis generation in llms under progressive information disclosure. arXiv preprint arXiv:2605.30284, 2026.

[34] Kaiyuan Liu, Youcheng Pan, Yang Xiang, Daojing He, Jing Li, Yexing Du, and Tianrun Gao. ProjectEval: A benchmark for programming agents automated evaluation on project-level code generation. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 20205–20221, 2025.

## A Related Work

AI Scientist Systems AI scientist systems combine language models with planning, retrieval, code execution, and scientific tools to automate increasingly complete research workflows [14, 15, 16]. Coscientist demonstrated tool-augmented planning and experimentation in chemistry [15], while The AI Scientist integrated idea generation, implementation, experimentation, visualization, writing, and review in machine-learning research [16]. The AI Scientist v2 further reduced its reliance on human-provided templates through experiment management and agentic tree search [17]. More recent systems, such as Co-Scientist and Kosmos, extend this direction toward collaborative hypothesis generation and long-horizon scientific investigation [18, 19]. Although these systems demonstrate broad workflow capabilities, their success does not necessarily indicate scientific autonomy, as objectives, methods, or experimental procedures may already be specified by humans [16, 20]. This motivates evaluations that distinguish autonomous workflow construction from reliable execution of provided methodology [8, 9].

Benchmarks for Scientific Agents Scientific-agent evaluation has progressed from knowledge-based tests to executable research tasks. GPQA and Humanity’s Last Exam assess advanced scientific knowledge without requiring data analysis or code execution [21, 5]. SciCode evaluates scientific programming [13], while DiscoveryBench, BLADE, DS Bench, and HypoBench focus on data-driven discovery and hypothesis generation [12, 22, 23, 24]. These benchmark target important capabilities but cover only selected stages of scientific research. More recent benchmarks evaluate longer workflows. ScienceAgentBench assesses data-driven scientific programming and execution [8]. MLE-Bench and RE-Bench evaluate open-ended machine-learning research engineering [10, 11], while PaperBench, CORE-Bench, and ReplicationBench measure research replication through hierarchical artifact-based or execution-based evaluation [9, 25, 26]. FIRE-Bench and ResearchClawBench extend evaluation toward full-cycle rediscovery and end-to-end scientific research [27, 28], and SciAgentArena introduces interactive cross-domain tasks with stepwise verification [29]. DiscoveryWorld, AstaBench, and ScienceBoard further expand evaluation toward simulated, suite-based, and realistic multimodal scientific workflows [30, 31, 32]. These benchmarks improve realism, execution depth, and outcome verifiability. However, most provide each task under a fixed specification. Their scores therefore show whether an agent completes the assigned workflow, but not whether the workflow was independently constructed or derived from supplied methodology [12, 8, 9, 25].

Scientific Autonomy and Guidance Dependence Several benchmarks vary the information available to a model. ScienceAgentBench compares performance with and without expert-provided knowledge, but its largely binary design does not distinguish method selection from procedural guidance [8]. ProjectionBench progressively reveals scientific information, yet focuses on hypothesis generation rather than executable project-level workflows [33]. ProjectEval varies specification detail for software projects [34], while contextual-robustness benchmarks examine sensitivity to irrelevant information. However, none evaluates how agents transition from following a prescribed method to independently constructing and validating a workflow within the same scientific project, or how this ability is affected by plausible distractors. ASI-Bench fills this gap through matched project instances that keep the research objective, data, required outputs, and evaluation criteria fixed while progressively withdrawing methodological guidance. Its four levels disentangle procedural execution, method identification, autonomous workflow construction, and distractor robustness, thereby directly measuring agents’ dependence on human-provided methodology.

## B Authors

ASI-Bench is a collaborative benchmark built through broad participation in scientific task construction and independent expert review. The final benchmark contains 60 project-level research tasks spanning 11 scientific domains. In total, 21 researchers contributed tasks retained in the final benchmark. These tasks further underwent five rounds of human review, resulting in more than 1,100 task-review assignments and repeated revisions throughout benchmark construction. Task contributors are ordered primarily by the number of tasks retained in the final benchmark.

## B.1 Task Contributors

The following researchers contributed scientific tasks retained in the final ASI-Bench benchmark.

Yuexi Pan<sup>1</sup>, Hengyu Wang<sup>1</sup>, Honghe Ren<sup>1</sup>, Peigan Gao<sup>9</sup>, Jiangyu Zhou<sup>1</sup>, Sijia Chen<sup>10</sup>, Junhao Wu<sup>1</sup>, Huan Wang<sup>1</sup>, Koutian Wu<sup>13</sup>, Cheng Zhang<sup>13</sup>, Yuanning Feng<sup>1</sup>, Qingyuan Zheng<sup>1</sup>, Wenzhe Li<sup>1</sup>, Jiakun Wu<sup>1</sup>, Ruixuan Jia<sup>1</sup>, Junwei Zhou<sup>5</sup>, Ergan Shang<sup>4</sup>, Jingjing Zhou<sup>1</sup>, Yan Xu<sup>2</sup>, Hongrui Zhang<sup>7</sup>, and Liting Mai<sup>6</sup>.

## B.2 Human Reviewers

Across five rounds of review, more than 1,100 task-review assignments were completed. The following researchers participated in the human review process.

Jiangyu Zhou<sup>1</sup>, Yuexi Pan<sup>1</sup>, Hengyu Wang<sup>1</sup>, Honghe Ren<sup>1</sup>, Xiaohan Jia<sup>1</sup>, Xueyang Zhou<sup>1</sup>, Cheng Zhang<sup>13</sup>, Yuanning Feng<sup>1</sup>, Sijia Chen<sup>10</sup>, Junhao Wu<sup>1</sup>, Huan Wang<sup>1</sup>, Koutian Wu<sup>13</sup>, Qingyuan Zheng<sup>1</sup>, Peigan Gao<sup>9</sup>, Wenzhe Li<sup>1</sup>, Jiakun Wu<sup>1</sup>, Jingjing Zhou<sup>1</sup>, Ruixuan Jia<sup>1</sup>, Yuexing Hao<sup>2</sup>, Yan Xu<sup>2</sup>, Hongrui Zhang<sup>7</sup>, Zhengxiang Cheng<sup>1</sup>, and Xianglin Ji<sup>2</sup>.

## B.3 Affiliations

<sup>1</sup> Tsinghua University, Beijing, China

<sup>2</sup> Massachusetts Institute of Technology, Cambridge, MA, USA

<sup>3</sup> Harvard University, Cambridge, MA, USA

4 Carnegie Mellon University, Pittsburgh, PA, USA

<sup>5</sup> University of Michigan, Ann Arbor, MI, USA

<sup>6</sup> University of Illinois Urbana–Champaign, Urbana, IL, USA

<sup>7</sup> Boston University, Boston, MA, USA

<sup>8</sup> The University of Queensland, Brisbane, QLD, Australia

<sup>9</sup> University of Science and Technology of China, Hefei, China

<sup>10</sup> Flatiron Institute, New York, NY, USA

<sup>11</sup> Microsoft Research

<sup>12</sup> AG2 AI

<sup>13</sup> Independent Researcher

## C Contributing to ASI-Bench

Growing ASI-Bench with the Research Community. ASI-Bench is designed as an evolving, community-driven benchmark rather than a fixed collection of scientific problems. The first release contains 60 project-level tasks across 11 scientific domains, but these tasks represent only a small fraction of the scientific challenges on which increasingly capable AI systems should be evaluated. Many of the most meaningful problems are best identified by researchers working directly at the frontier of their respective fields. We therefore invite scientists, engineers, AI researchers, and benchmark developers to contribute new research tasks to future releases of ASI-Bench. We particularly welcome problems that introduce new scientific domains, broaden the methodological diversity of existing domains, or test research capabilities that remain underrepresented in the current benchmark.

A contributed task should represent a genuine scientific problem rather than a conventional question-answering exercise. It should define a meaningful and reproducible scientific objective, require non-trivial reasoning or research execution, and produce artifacts that can be evaluated through explicit and reproducible criteria. Contributors retain the scientific substance of their research problems while expressing them through the standardized ASI-Bench task format.

What Constitutes a Complete Task? A complete contribution contains not only a scientific question, but also the materials required to execute and evaluate it reproducibly. Each submission should include: (i) a clearly stated scientific objective and an explanation of why the problem is non-trivial; (ii) four prompt variants, B1–B4, defining the methodological-information gradient; (iii) explicit input and output specifications for all agent-visible files and required artifacts; (iv) a reproducible reference-generation procedure; (v) an evaluation specification containing validity checks and weighted scoring criteria; (vi) the scientific software dependencies required by the task; and (vii) evidence from local evaluation across all four information conditions.

Constructing the B1–B4 Information Gradient. A central requirement of each contributed task is the controlled construction of four information conditions. B1 provides the scientific background, method, equations, and procedural information required to execute the task. B2 specifies the intended methodological approach and relevant constraints while leaving implementation decisions to the agent. B3 specifies only the scientific objective, available inputs, constraints, and required outputs, requiring the agent to determine an appropriate solution strategy independently. B4 extends B3 with factually correct but non-essential information, allowing robustness to irrelevant context to be evaluated. In particular, B3 and B4 should not reveal the intended algorithm, solver, method, or other information that would compromise the methodological gradient.

How to Contribute a Task. To make task contribution accessible across scientific disciplines, ASI-Bench provides a public authoring scaffold and an online Guided Flow. As shown in Figure 5, contributors can start from the public task template and complete the submission through the online portal at https://asibench.apexin.ai/submit or via the CLI launcher. Before submission, each task must form a self-contained and reproducible package. It should include a well-defined scientific objective, four B1–B4 prompts, task data, a reference-generation script (generate\_gt.py), an evaluation configuration, runtime dependencies, and local-testing evidence. The contribution process consists of three main stages:

![](images/450886829093e9f814e5a220b7d7a8ea5956171bfde0d75cf9587fd8782a1d3a.jpg)  
Figure 5: ASI-Bench task contribution portal. The online submission workspace provides a 15-step Guided Flow for constructing and validating new tasks. Representative interfaces show the major stages of task authoring, including scientific problem definition, B1–B4 prompt construction, evaluation design with gates and scorers, runtime configuration, task-file preparation, and local testing before review.

1. Formulate the Scientific Question. Contributors first define the scientific question to be investigated and explain why it is non-trivial. The task should specify what must be computed or discovered, rather than prescribing how to solve it. It should also declare the scientific dependencies required by the task. The target result must be deterministic and reproducible so that it can support reliable evaluation.

2. Build the Four Prompt Levels. Contributors construct B1–B4 for the same scientific objective. B1 provides the complete method and detailed methodological guidance. B2 retains method-level guidance but leaves implementation decisions to the agent. B3 specifies only the objective, inputs, constraints, and required artifacts, without revealing the intended method. B4 keeps the B3 task unchanged while adding factually correct but non-essential information. B3 and B4 must remain complete task specifications, including all required inputs and outputs. Contributors also provide generate\_gt.py to generate the reference result and define evaluation gates and weighted scorers for the required scientific outputs.

3. Test, Submit, and Revise. Before submission, contributors evaluate B1–B4 using the model and evaluation settings specified by the submission portal at https://asibench.apexin.ai/submit/, and report the resulting scores. Local testing also records execution time, environment information, sandbox provenance, and scorer-replay evidence. These results are used to check task difficulty, information leakage, reproducibility, and evaluation stability. Once the required files and evidence are complete, the task can be submitted for review. Reviewers examine the scientific formulation, B1–B4 information gradient, reference generation, scoring logic, and runtime behavior. Contributors then revise the task in response to feedback until it satisfies the benchmark requirements.

Contribution Recognition. Community contributors are treated as participants in the construction of ASI-Bench rather than merely as external data submitters. For every contributed task that successfully passes scientific and technical review and is incorporated into an official ASI-Bench release, the corresponding task contributor(s) will be included in the contributor list of that benchmark release.

An Open Benchmark for the Path Ahead. The current 60 tasks should be viewed as a starting point rather than a closed test set. The scientific problems that will distinguish increasingly capable AI systems cannot be defined by a single research group alone. We therefore invite researchers from different disciplines to contribute the research problems they believe future AI systems should be able to solve, and invite benchmark and model developers to help refine evaluations, identify failure cases, and validate new tasks. By continuously incorporating new scientific challenges from the research community, we aim for ASI-Bench to evolve together with AI capability, challenge the limits of today’s AI, and ultimately help measure and accelerate progress beyond the frontier of human scientific capability toward artificial superintelligence.

## D Case Study

## Representative Scientific Task

## Task: 2D Anisotropic Stiff Dynamics

The agent is given observations of a two-dimensional nonlinear dynamical system on a periodic spatial domain.

## Input data

• system\_info.json: system parameters and timing metadata;

• field\_evolution.npy: observed spatio-temporal field snapshots;

• initial\_condition.npy: initial field for the prediction task.

## Scientific objective

Given the observations, the agent must:

1. characterize the spatial and temporal dynamics;

2. construct a model that reproduces the observed dynamics;

3. predict the field at a future target time;

4. extract physically meaningful diagnostics.

## Required artifacts

The agent must produce a predicted field, spatial spectrum, physical diagnostics, scientific visualizations, dataanalysis results, and the complete executable simulation code.

The scientific objective, input data, required artifacts, and evaluation criteria are kept fixed across B1–B4. Only the information provided in the prompt changes.

## B1 Prompt — Full Equation and Solver Guide

## Task Overview

You are given observed spatio-temporal evolution data from a 2D nonlinear pattern-forming system governed by a conserved anisotropic Kuramoto–Sivashinsky-type equation on a doubly periodic domain:

u\_t = -alpha \* u   
+ u\_xx + mu \* u\_yy   
- nu \* u\_xxxx   
- 2 \* gamma \* u\_xxyy   
- delta \* u\_yyyy   
+ nabla^2 [   
(lambda\_xx / 2) \* u\_x^2   
+ lambda\_xy \* u\_x \* u\_y   
+ (lambda\_yy / 2) \* u\_y^2   
]

The domain is [0,Lx] x [0,Ly]. The prompt provides the values of Lx, Ly, alpha, nu, mu, gamma, delta, lambda\_xx, lambda\_xy, and lambda\_yy.

Spatial Discretization   
Use a 2D Fourier pseudospectral method. Define   
kx = 2\*pi\*fftfreq(Nx, d=dx)   
ky = 2\*pi\*fftfreq(Ny, d=dy)   
and the Fourier-space linear operator   
L\_hat =

-alpha   
+ kx^2   
+ mu \* ky^2   
- nu \* kx^4   
- 2 \* gamma \* kx^2 \* ky^2   
- delta \* ky^4   
Compute the nonlinear term pseudospectrally in conserved form:   
f(u\_x, u\_y) =   
(lambda\_xx / 2) \* u\_x^2   
+ lambda\_xy \* u\_x \* u\_y   
+ (lambda\_yy / 2) \* u\_y^2   
N\_hat = -(kx^2 + ky^2) \* fft2(f)   
Apply 2/3-rule dealiasing to the nonlinear term.   
Time Integration   
The system is stiff because of the fourth-order terms. Use ETDRK4.   
Precompute   
E = exp(L\_hat \* dt)   
E2 = exp(L\_hat \* dt / 2)   
and construct the ETDRK4 coefficients using the Kassam–Trefethen contour-integral formulas.   
For the current Fourier state:   
a = E2 \* u\_hat + Q \* N\_hat(u\_hat)   
b = E2 \* u\_hat + Q \* N\_hat(a)   
c = E2 \* a   
+ Q \* (2\*N\_hat(b) - N\_hat(u\_hat))   
u\_hat\_next =   
E \* u\_hat   
+ f1 \* N\_hat(u\_hat)   
+ 2\*f2 \* (N\_hat(a) + N\_hat(b))   
+ f3 \* N\_hat(c)   
Your job is to analyze the observed data, reconstruct the numerical solver, predict the field at the target time, and produce the   
required scientific artifacts.

## B2 Prompt — Method Background

## Task

You are given observed data from a 2D nonlinear evolution system that produces directionally uneven, spatially complex patterns. The system is stiff and exhibits irregular pattern dynamics rather than simple steady diffusion. Your goal is to analyze the data, build a numerical model that reproduces the dynamics from the supplied initial condition, predict the future field, and extract several physical diagnostics.

## Physical and Numerical Background

This is a fourth-order nonlinear PDE problem in two spatial dimensions. The main ingredients are:

• weak linear damping;

• destabilizing long-wave growth through second-order terms;

• stabilizing high-order dissipation through fourth-order terms, including cross-derivative coupling;

• nonlinear energy transfer with a conservation-type structure;

• different behavior in the two spatial directions.

Because of the fourth-order terms, the linear stiffness can be severe. For this class of systems, methods based on spectral discretization with stiffness-aware time stepping are needed.   
Reasonable time-stepping approaches include:

• ETD methods such as ETDRK4;

• IMEX schemes;

• semi-implicit spectral integrators.

ETD-style spectral solvers are often competitive, but they are not the only reasonable option.

The system parameters are supplied in data/system\_info.json with neutralized coefficient names.

Inspect the observed data before locking in a solver. The two spatial directions should be analyzed separately where appropriate.   
The task is not only to forecast the field, but also to recover interpretable physical structure from the data.

## B3 Prompt — Research Objective and Data Only

## Task

You are given observed spatio-temporal data from a 2D nonlinear system on a periodic domain. The field develops structured patterns and evolves in a complex, nontrivial way over time.

## Input Data

• data/system\_info.json: system parameters and timing metadata;

• data/field\_evolution.npy: observed 2D field snapshots;

• data/initial\_condition.npy: initial field used for prediction.

## Goal

1. Explore the observed data to characterize its spatial and temporal structure.

2. Identify or construct a mathematical model that can reproduce the observed dynamics from the supplied initial condition.

3. Predict the field at the target time specified in system\_info.json.

4. Extract physically meaningful quantities from both the observed data and the predicted field.

## Constraints

• Outputs must be finite, with no NaN or Inf values.

• Use the provided data files only.

• The governing equation, numerical representation, and effective evolution strategy are not given explicitly. Identifying an appropriate approach is part of the task.

## B4 Prompt — Research Objective with Distracting Information

## Task

You are given observed spatio-temporal data from a 2D nonlinear system on a periodic domain. The field develops structured patterns and evolves in a complex, nontrivial way over time.

## Goal

1. Explore the observed data to characterize its spatial and temporal structure.

2. Identify or construct a mathematical model that can reproduce the observed dynamics.

3. Predict the field at the specified target time.

4. Extract physically meaningful quantities from the observed and predicted fields.

## Additional Context

Several unrelated 2D PDE families can produce structured fields with evolving spectra. Reaction–diffusion systems such as FitzHugh–Nagumo or Gray–Scott models generate Turing patterns through activator–inhibitor interactions. Phase-field models such as Cahn–Hilliard describe spinodal decomposition. Swift–Hohenberg-type equations model convective pattern formation. Thin-film equations describe coating flows and droplet spreading. Complex Ginzburg–Landau variants model oscillatory instabilities.

There are also many possible computational routes. Finite differences with explicit stepping are straightforward. IMEX schemes treat different parts of the operator differently. Exponential integrators handle the linear part through matrix exponentials. Operator splitting divides the problem into substeps. Pseudospectral solvers use FFTs. Neural operators attempt to learn the dynamics directly from data.

For spatial characterization, possible approaches include Fourier analysis, spatial autocorrelation, variograms, structure functions, wavelet decomposition, proper orthogonal decomposition, and template matching. Temporal structure can be analyzed through autocorrelation, recurrence analysis, or temporal spectra.

Some preliminary notes are also attached:

“maybe it is phase-field or Cahn–Hilliard?”

“could be some activator–inhibitor reaction–diffusion thing?”

<table><tr><td>“try a simpler neural surrogate first?&quot;</td></tr><tr><td>“maybe just compare the last observed frame to the target frame?&quot;</td></tr><tr><td>“check if the patterns are just equilibrium Turing patches&quot;</td></tr><tr><td>There is also unrelated background chatter about deadlines, news, semiconductor markets, sports, entertainment, and upcoming</td></tr><tr><td>meetings. Your job remains the same: focus on the supplied scientific data, infer a suitable model, produce the required outputs, and</td></tr></table>