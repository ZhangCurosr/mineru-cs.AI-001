# Hierarchical Federated Transfer Learning in Digital Twin-Based Vehicular Networks

Qasim Zia<sup>a</sup>, Saide Zhu<sup>b</sup>, Haoxin Wang<sup>a</sup>, Zafar Iqbal<sup>a</sup>, Yingshu Li<sup>a</sup>

Hierarchical Federated Transfer Learning in Digital Twin-Based Vehicular Networks

Qasim Zia, Saide Zhu, Haoxin Wang, Zafar Iqbal, Yingshu Li

Georgia State University, Atlanta, GA, 30303   
Penn State Berks, Reading, PA, 19610

![](images/92070dbe84b18f375833dba8dbd395f5b296d5f111c2cc6c451ed01d3c398bf6.jpg)

## Highlights

Hierarchical Federated Transfer Learning in Digital Twin-Based Vehicular Networks

Qasim Zia<sup>a</sup>, Saide Zhu<sup>b</sup>, Haoxin Wang<sup>a</sup>, Zafar Iqbal<sup>a</sup>, Yingshu Li<sup>a</sup>

• Research highlight 1

• Research highlight 2

# Hierarchical Federated Transfer Learning in Digital Twin-Based Vehicular Networks

Qasim Zia<sup>a</sup>, Saide Zhu<sup>b</sup>, Haoxin Wang<sup>a</sup>, Zafar Iqbal<sup>a</sup>, Yingshu Li<sup>a</sup>

Department of Computer Science, Georgia State University, 7th Floor, 25 Park Place Bldg, Atlanta, 30303, GA, USA

College of Information Sciences and Technology, Penn State Berks, 288 N. Burrowes Rd, Reading, 19610, PA, USA

## Abstract

In recent research on the Digital Twin-based Vehicular Ad hoc Network(DT-VANET), Federated Learning (FL) has shown its ability to provide data privacy. However, Federated learning struggles to adequately train a global model when confronted with data heterogeneity and data sparsity among vehicles, which ensure suboptimal accuracy in making precise predictions for diferent vehicle types. To address these challenges, this paper combines Federated Transfer Learning (FTL) to conduct vehicle clustering related to types of vehicles and proposes a novel Hierarchical Federated Transfer Learning (HFTL). We construct a framework for DT-VANET, along with two algorithms designed for cloud server model updates and intra-cluster federated transfer learning, to improve the accuracy of the global model. In addition, we developed a data quality score-based mechanism to prevent the global model from being afected by malicious vehicles. Lastly, detailed experiments on real-world datasets are conducted, considering diferent performance met rics that verify the efectiveness and eficiency of our algorithm.

Keywords: Vehicular ad-hoc network, Hierarchical federated transfer learning, Vehicular Digital Twin, autonomous vehicle, Digital Twin-based Vehicular Networks.

## 1. Introduction

Vehicular networks enable numerous applications for Intelligent Transportation Systems (ITSs), including trafic control, infotainment, and accident reporting Noor-A-Rahim et al. (2020). These applications rely on various performance criteria, such as reliability, latency, and user-defined features like the quality of physical experience. Thus, the efective handling of vehicle heterogeneity, data privacy, and limited computational resources has become an urgent challenge to discuss, while current wireless technologies struggle to assist it thoroughly Zia et al. (2016). Despite advanced onboard sensors and processing capabilities of autonomous vehicles, they still face challenges such as limited sensing range, limited local processing capacity, and communication barriers He et al. (2022). To meet these diverse requirements, the development of vehicular networks must incorporate two key advancements: proactive, intelligent analytics and self-sustaining wireless systems.

Table 1: Nomenclature
<table><tr><td rowspan=1 colspan=1>Symbol</td><td rowspan=1 colspan=2>Description</td></tr><tr><td rowspan=1 colspan=1> $C _ { n }$ </td><td rowspan=1 colspan=2>The Cluster&#x27;s Collection that are not trained</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=2> $\overline { { i ^ { t h } } }$ Cluster</td></tr><tr><td rowspan=1 colspan=1>U</td><td rowspan=1 colspan=2>The vehicle v</td></tr><tr><td rowspan=1 colspan=1> $j$ </td><td rowspan=1 colspan=1>The ${ \overline { { j ^ { t h } } } }$ model update and the total number is equal to  $\overline { { | C _ { n } | } }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1> $n$ </td><td rowspan=1 colspan=2>Total number of clusters</td></tr><tr><td rowspan=1 colspan=1> $\overline { { V _ { h } , V _ { r } , V _ { e } } }$ </td><td rowspan=1 colspan=2>In a cluster, the head vehicle, route vehicle, and edge vehicle DT</td></tr><tr><td rowspan=1 colspan=1> $P _ { O }$ </td><td rowspan=1 colspan=2>All the post vehicles that exist for any vehicle v in a given cluster</td></tr><tr><td rowspan=1 colspan=1> $_ { D t Q }$ </td><td rowspan=1 colspan=2>The data quantity that is used to train any given vehicle</td></tr><tr><td rowspan=1 colspan=1> $w ^ { - }$ </td><td rowspan=1 colspan=2>The model parameters that are used to send for training</td></tr><tr><td rowspan=1 colspan=1>w</td><td rowspan=1 colspan=2>The model parameters that are received after training</td></tr><tr><td rowspan=1 colspan=1> $S _ { r }$ </td><td rowspan=1 colspan=2>The score calculated for the credibility of the vehicle</td></tr><tr><td rowspan=1 colspan=1> $D a t a$ </td><td rowspan=1 colspan=2>The personal data of any vehicle</td></tr><tr><td rowspan=1 colspan=1>W</td><td rowspan=1 colspan=2>The Global model parameters</td></tr><tr><td rowspan=1 colspan=1> $\overline { { V D T } }$ </td><td rowspan=1 colspan=2>Vehicle Digital Twin</td></tr><tr><td rowspan=1 colspan=1> $D T$ </td><td rowspan=1 colspan=2>Digital Twin</td></tr></table>

Proactive Intelligent Analytics is a data analytics technique that analyzes incoming data while anticipating potential challenges, trends, and opportunities in advance. It involves real-time or nearly real-time data analysis using the most advanced machine learning, artificial intelligence, and prediction models. Proactive online learning wireless systems will make it possible to optimize wireless resources to ensure the quality of service (QoS) of diverse ITS applications with varying requirements.

Self-sustaining wireless systems can function without constant external power sources. These systems are helpful for Internet of Things(IoT) devices, remote locations, and scenarios where battery replacement or maintenance is challenging. Self-sustaining wireless networks will allow ITS to run with the least operator or user interaction.

The combination of digital twins and vehicular networks has become a promising paradigm in recent years, transforming our understanding of and approach to managing transportation systems. The development and growth of the digital twin provided insight into how to address the problems associated with autonomous vehicles. Virtual representation of physical systems serves as the foundation for digital twins. A digital twin is an accurate virtual representation of a physical thing on a cloud that shows its lifetime and status. The term “Digital Twin-based Vehicular Ad-Hoc Network” refers to the cloud-based digital twin of a physical vehicle that synchronizes real-time sensing data from actual vehicles through wireless communication. Although digital twins can enhance the understanding of vehicle systems through virtual representations, they still require eficient data processing and analysis technologies to enable decision-making. Distributed or centralized training can serve as a foundation for Machine Learning (ML) because vehicles are reluctant to share their private information, and centralized training causes privacy leaks by moving data from dispersed devices to a centralized location. At the same time, traditional centralized machine learning methods have proven insuficient when dealing with nonindependent and identically distributed (non-IID) data.

Artificial Intelligence (AI) and Machine Learning (ML) have revolutionized biomedical research, enabling breakthroughs in areas such as antibody design Ahmed et al. (2026b), medical imaging Ahmed et al. (2023), and protein-protein interaction prediction Ahmed et al. (2026a). This demonstrates the power of AI in understanding complex molecular interactions Ahmed et al. (2025). However, the data sharing remains a bottleneck for research due to privacy concerns. Federated Learning(FL) is the most recent technique in digital twin-based vehicular ad hoc networks that addresses centralized learning (CL) issuesKhan et al. (2023). FL is well known for its ability to promote collaboration and protect data privacy in various domains, including industrial and medical cyber-physical systems AbdulRahman et al. (2020)Li et al. (2020). FL, which employs dispersed vehicles to learn a global FL model without transporting data from devices to a centralized site for training, is presented as a solution to this centralized ML constraint Khan et al. (2021). Compared to centralized ML, FL maintains better privacy. However, traditional federated learning faces issues such as slow convergence and insuficient accuracy when dealing with the diversity of vehicle types and limited vehicle data. Therefore, we use the Hierarchical Federated Transfer Learning (HFTL) technique in digital twin-based vehicular networks (DT-VANET). HFTL groups vehicles into diferent clusters based on their type, and pre-trained models are used to perform personalized model fine-tuning that fits each vehicle type’s specific needs. It also improves both prediction accuracy and convergence speed in a meaningful way.

![](images/6917d43d53e4f03ea666b18b72b62e8e0ab0323b2d047d9bf074d71787aa9516.jpg)  
Figure 1: High-level architecture of Hierarchical Federated Transfer Learning for digital twin-based vehicular network

We also resolved another issue related to optimizing the route associated with the specific vehicle type and managing the aggregation of communication overhead between various vehicle types that makes precise and accurate predictions for each vehicle type. Our HFTL for DT-VANET has also successfully resolved this issue. To ensure our architecture is trustworthy and reliable, each vehicle will assess its data quality, vehicle health, and safe driving performance score, and these scores will be stored in the blockchain of the cloud, which can be used to give rewards later. We use blockchain’s tamperproof feature to ensure that recorded data cannot be maliciously altered. In that way, they are stopping untrustworthy vehicle nodes from uploading low-quality or fake data. Additionally, the distributed consensus mechanism of blockchain allows data quality scores to be shared across multiple nodes, making the global model updates more fair and reliable. Furthermore, the reward mechanism for vehicles can be automatically carried out through smart contracts. So, our research paper addresses issues and challenges related to privacy, eficiency, accuracy, and scalability of machine learning models in digital twin vehicle networks, enhancing model performance through implementing an HFTL approach.

To the best of our knowledge and understanding, our work is the first to review HFTL and Trustworthiness metrics in a single architecture of DT-VANET. Also, we illustrate a scenario to enable vehicle networks based on digital twins through the diagram. We also discussed the related work, algorithms of HFTL, and performance metrics in twin-based vehicular networks. Therefore, the following is a summary of our contributions:

1. We build a general thorough architecture for vehicular networks based on digital twins. A scenario for a digital twin-based vehicular network is shown, which enables eficient data synchronization between physical and virtual nodes of vehicles, enhancing the model’s scalability and response speed. Additionally, we ofer diagrams for twin-based vehicle networks.

2. We outline the algorithms of federated transfer learning in digital twinbased vehicular networks. It allows for model customization based on diferent vehicle types, which improves prediction accuracy and training eficiency.

3. Lastly, we conducted experiments on real-time datasets to evaluate the performance metrics. By comparing the performance of HFTL, Federated Learning (FL), and Centralized Learning (CL) in terms of model accuracy, training time, communication overhead, and resource consumption, we validated the advantages of HFTL in diferent scenarios.

The rest of the paper is structured as follows: The related works are reviewed in Section 2. We suggest a Weighted Cloud Server Cycling Model Update Algorithm and an Inner Cluster Federated Transfer Learning Algorithm in Section 3. Subsequently, in Section 4, comprehensive experiments are conducted and evaluated to determine the eficiency and efectiveness of our suggested architecture. Finally, Section 6 discusses our conclusions.

## 2. Related Works

This section overviews recent research on Hierarchical Federated Learning, Federated Transfer Learning, and Clustering Techniques for VANET.

## 2.1. Hierarchical Federated Learning in VANET

Numerous research has presented the use of Federated Learning to address data privacy. However, Google suggested the first work on Federated Learning (FL) in 2016 Konečn\`y et al. (2016) for conventional FL. In 2022, Li et al. Li et al. (2022) proposed a FEEL framework using Federated Learning for non-IID data. To predict the high accuracy of network trafic with a high volume of data, Sepasgozar et al. Sepasgozar and Pierre (2022) proposed a new Network Trafic Prediction (Fed-NTP) based on federated learning. Hierarchical Learning in Machine Learning was first introduced by Zhang et al. Zhang and Zhang (2006). The human approach to problem-solving in hierarchies inspires their work. In 2021, Goncalves et al. Gonçalves et al. (2021) discuss the Hierarchical approach for Security Framework in VANET. In 2024, HaghighiFard et al. HaghighiFard and Coleri (2024) discuss the use of Hierarchical Federated Learning in VANET. It uses the cosine similarity of FL model parameters and average relative speed as a metric for making clusters. Although Li et al.Li et al. (2022), the FEEL framework successfully applied federated learning to non-IID data environments. It did not fully address the data communication challenges in highly dynamic VANET environments. In contrast, HaghighiFard et al.HaghighiFard and Coleri (2024) introduced hierarchical federated learning, which reduced communication overhead to some extent. However, their clustering method still relies on fixed vehicle attributes, making it less adaptable to the rapid changes in vehicle environments.

## 2.2. Federated Transfer Learning in VANET

Federated Transfer Learning(FTL) facilitates knowledge transfer without afecting user privacy. The FTL was first discussed by Liu et al. in 2020 Liu et al. (2020), where they enabled the target-domain party to use the source domain’s rich level to create adaptable and eficient models. In 2022, Otoum et al. Otoum et al. (2022) proposed an intrusion detection system in VANET. On that account, they have compared Split Learning, Federated Learning, and Transfer Learning. However, Liu et al. Liu et al. (2020) paper discusses Federated Transfer Learning. Still, the paper does not consider DT-VANET for their findings. Otoum et al. Otoum et al. (2022) consider federated transfer learning only for intrusion detection scenarios, not for digital twinbased communication.

## 2.3. Clustering Techniques for VANET

The primary purpose of clustering is to find discrete groups of vehicles in the environment. It will reduce the communication with the roadside units. Zia (2015) discusses various data-centric protocols related to clustering. The PasCar routing protocol was presented by Wang et al. Wang and Lin (2013) in 2013 to make a reliable cluster structure based on a single hop. It chooses suitable participants from candidate nodes. However, the single-hop mechanism limits the system’s coverage and stability. Because of this, there has been a lot of research on multi-hop clustering techniques in the years that followed. In 2022, Rashid et al. Rashid et al. (2020) discussed eficient multi-hop clustering based on prediction. This paper discusses increasing the cluster coverage area increase, link stability, energy eficiency, and mobility characteristics increase. Temurnikar et al. Temurnikar et al. (2022) in 2022 discussed a new Particle Swarm Optimization (PSO) based multi-hop method because of which you can use the best route. We can also select the Cluster head, which is much more stable. It also identified false messages from malicious vehicles. Zia et al. (2024) in 2024, discussed the possibility of using priorities to achieve faster response time. Compared to the techniques already mentioned, this paper introduces Hierarchical Federated Transfer Learning (HFTL), which ofers more flexible clustering to address vehicle heterogeneity. Additionally, integrating digital twin technology enables real-time data synchronization, improving prediction accuracy and reducing communication overhead.

## 3. THE PROPOSED HIERARCHICAL FEDERATED TRANS-FER LEARNING IN DIGITAL TWIN-BASED VEHICULAR NETWORKS

In this section, we proposed a high-level secure and robust federated transfer learning architecture for vehicular networks, as shown in Figure 1. There is a digital twin layer, and the physical layer forms the two main layers of the proposed architecture. The physical layer consists of objects like base stations (BS), which help end users and distribute autonomous vehicles. The Digital Twin Layer consists of four essential parts: data storage, twin management, virtual model mapping, and blockchain Dai and Zhang (2022). It uses digital twins to efectively map the virtual twin system objects and real-vehicle edge computing. The cloud server implements the digital twin layer, and the idea of digital twin objects is used Khan et al. (2022a). A virtual representation of the physical system is known as a digital twin object, and it is doable to simulate numerous vehicle network operations and applications using this virtual representation type of the physical vehicular network. Mathematical and experimental methodologies can be used to model such network applications/functionsKhan et al. (2022b). However, as we know, mathematical modeling is highly dependent on assumptions; it might not be able to denote the actual network accurately. Meanwhile, mistakes may prevent experimental modeling from being accurate during experimentation. In digital twin-based vehicular networks, a data-driven modeling approach based on Federated Transfer Learning may be considered to overcome these problems. It also provides enhanced network performance and eficiency, improved safety and reliability, and allows advanced applications to run. Hierarchical Federated Transfer Learning(HFTL) in digital twins is used for precise and accurate prediction according to the vehicle type. Blockchain is also used to secure the trustworthiness metrics of the individual vehicle. We will now discuss the features of this architecture one by one.

## 3.1. Hierarchical Federated Transfer Learning (HFTL)

HFTL’s clustering considers vehicle size, purpose, driving patterns, and computational resources. It will help to build a customized model. This model aligns with the learning goals and data properties specific to each kind of vehicle. As shown in Figure 1, vehicles are grouped according to the type of vehicles. The types of vehicles are classified according to vehicle size, purpose, and computational resources in Table 2. Generally, the vehicle with the most advanced features is selected as the cluster head.

Table 2: Categorization of Vehicles
<table><tr><td rowspan=1 colspan=1>Category ID</td><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Types of Vehicles</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Regular Vehicles</td><td rowspan=1 colspan=1>Cars</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Emergency Vehicles</td><td rowspan=1 colspan=1>Ambulances, Fire Trucks,Police Cars</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Public Transportation Vehicles</td><td rowspan=1 colspan=1>Buses</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Delivery Vehicles</td><td rowspan=1 colspan=1>Goods and Packages Trans-portation Vehicles</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Specialized Vehicles</td><td rowspan=1 colspan=1>Construction Vehicles, Util-ity Vehicles, Road Mainte-nance Vehicles</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Two-Wheel Regular Vehicles</td><td rowspan=1 colspan=1>Motorcycles</td></tr></table>

$$
D T = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } T v _ { j } \in C _ { i }\tag{1}
$$

Where $C _ { i }$ is a cluster that contains a type of vehicle j.

Equation 1 discusses that in the Digital Twin, there are many clusters from $C _ { i }$ to $C _ { n }$ where each cluster $C _ { i }$ contains only one type of vehicle $T v _ { j }$

Secondly, HFTL improves the convergence and heterogeneity in Dispersed Federated Learning. In HFTL, a pre-trained model is shared depending on the type of vehicle and target application, such as trafic flow prediction, collision risk prediction, intelligent parking prediction, etc. This pre-trained model will be fine-tuned using the vehicle’s private local data. This enables the model to be customized for the environment and based on the vehicle’s driving patterns. For example, public vehicles that require higher precision will have their model fine-tuning focus more on safety and route optimization. However, private vehicles may prioritize reducing idle times and optimizing the driving experience. After training, share the model update to the cloud server for the global model update. Therefore, model updates are combined from every vehicle’s digital twin to the cloud to make it a more robust and heterogeneous global model. Several digital twins of vehicles use these model updates to make decisions and predictions. The model updates from every vehicle digital twin are given weight according to the trustworthiness metrics. These metrics are recorded and verified through blockchain. Vehicles with higher trustworthiness have greater weight in global model updates, ensuring the security and accuracy of the model updates. These weights will be considered for the Global model update using the Federated Averaging Method.

![](images/bee2080acd1729184c8bfb0de0270354b82bc88c71bc17ef2c5965514139b21f.jpg)  
Figure 2: The Flowchart and Vehicle type data flow for the Inner Cluster of Hierarchical Federated Transfer Learning

## 3.2. Digital Twin Based Vehicular Networks

As we discussed briefly above, the digital twin architecture in VANET allows us to have many benefits, including improved decision-making, increased safety, predictive maintenance, optimized network performance, enhanced scalability, enhanced simulation and testing, real-time monitoring and maintenance, and cost reduction.

## 3.3. Blockchain Integration in Architecture

In this architecture, blockchain can be defined as a distributed ledger integrated into DT-VANET-based HFTL architecture to manage and store weights. The trustworthiness metrics calculate every weight, as discussed in the next section. Low-weight model updates are not considered to update the global model to maintain integrity. It will also help us detect anomalies. We ensure the secure DT-VANET-based HFTL architecture through the weights and tamper-proof blockchain.

## 3.4. Trustworthiness Metrics

$$
\begin{array} { r l } & { D T S _ { z } = w _ { 1 } \cdot D T C _ { z } + w _ { 2 } \cdot S R C _ { z } + w _ { 3 } \cdot E _ { z } } \\ & { \qquad + w _ { 4 } \cdot C S F _ { z } + w _ { 5 } \cdot G R H L _ { z } + w _ { 6 } \cdot S F R _ { z } } \end{array}\tag{2}
$$

where:

$D T S _ { z } =$ The Data Quality Score of vehicle z

$D T C _ { z } =$ The Data Completeness of vehicle z

$S R C _ { z }$ = The sensor collaboration of vehicle z

$E _ { z } =$ The events (trafic incidents, road conditions, etc.) recorded by vehicle z

$C S F _ { z } = { " }$ The constant format (According to Standard Data Formats) for vehicle z

$G R H L _ { z } = \mathrm { T }$ he General Health of vehicle $\mathrm { ~ z ~ } ( \mathrm { e . g . }$ ., looking after maintenance issues)

$S F R _ { z } =$ The reliable and safe driving patterns for vehicle $\mathrm { ^ { z , } }$

$w _ { 1 } , w _ { 2 } , w _ { 3 } . . . w _ { 6 }$ are the weights that are assigned with each factor

Every vehicle will calculate its data quality score depending on data completeness, sensor collaboration, events, constant format, the vehicle’s general health, and safe, reliable driving patterns.

$$
R P S _ { z } ( t m + 1 ) = R P S _ { z } ( t m ) + \alpha \cdot D Q S _ { z } - \beta \cdot M _ { z }\tag{3}
$$

where:

$R P S _ { z } ( \mathrm { t m } ) = \mathrm { \Delta }$ The Reputation Score of vehicle z

$\alpha =$ The Weight for data quality score

$\beta =$ The malicious behaviour Penalty factor

$M _ { z } = { ' }$ The malicious activity indicator for vehicle $\mathrm { ~ z ~ } ( 0 \ \mathrm { o r \ 1 } )$

$R P S _ { z } ( \mathrm { t m } + 1 ) = \mathrm { A t }$ tm $+ ~ 1$ , updated reputation score of vehicle z

$$
W t _ { z } = \frac { R P S _ { z } } { \sum _ { m = 1 } ^ { n } { R P S _ { m } } }\tag{4}
$$

where:

$W T _ { z }$ = Weight of vehicle z’s model update.

$n =$ Total number of vehicles

These scores are input to the reputation system maintained at the cloud end. This reputation system helps DT-VANET to assign weight to the model update as described in the previous section. It will also help to keep the malicious vehicle from interfering.

$$
\Delta R P S _ { z } = R P S _ { z } ( t m + 1 ) - R S _ { z } ( t m )\tag{5}
$$

$$
B C _ { z } = \{ \Delta R P S _ { z } ( t m _ { 1 } ) , \Delta R P S _ { z } ( t m _ { 2 } ) , . . . , \Delta R P S _ { z } ( t m _ { n } ) \}
$$

where:

$B C _ { z }$ is the set of reputation changes over the time $t _ { 1 } , t _ { 2 } , . . . t _ { m }$ for vehicle z

The reputation of a vehicle changes over time, and digital twins on the cloud are in charge of figuring out and recording these value changes on the blockchain.

$$
R W _ { z } = f ( R P S _ { z } ) \quad \mathrm { w h e r e } \quad f ^ { \prime } ( R P S _ { z } ) > 0\tag{6}
$$

Where:

$R W _ { z }$ for vehicle z be a function of reputation score $R P S _ { z }$

$$
R W _ { z } = \gamma \cdot R P S _ { z }
$$

Where:

• $\gamma$ is used for changing reputation score $R P S _ { z }$ into a specific reward with a scaling factor.

Vehicles with good scores and positive contributions may be rewarded with tax or toll reductions or urgent access to services.

$$
R P S _ { z } < \theta \implies \mathrm { V e h i c l e } \ z \ \mathrm { i s \ e x c l u d e d \ f r o m \ p a r t i c i p a t i o n } .\tag{7}
$$

Where:

• Malicious Vehicle reputation score $R P S _ { z }$ for vehicle z is considered dangerous and stopped from participation

A blockchain ensures tamper-proof records of participation, high-quality data, and protocol enforcement.

Algorithm 1 Weighted Cloud Server Cycling Model Update   
1: The central cloud server has existing model parameters $W _ { e x i s t }$ and digital   
twin of the entire system for simulation;   
2: Initialize $C _ { n }$ ;   
3: for each update $j = 0 , 1 , 2 , \cdots , j - 1$ do   
4: $W _ { e x i s t } ^ { - } = W _ { e x i s t } ;$   
5: Select cluster i from $C _ { n } ;$   
6: Send $W _ { e x i s t } ^ { - }$ to Digital Twin of cluster i ;   
7: c =Inner-Cluster $\left( W _ { e x i s t } ^ { - } \right)$ ;   
8: Calculate $F _ { w e i g h t } ( W _ { e x i s t } ^ { + } )$ ;   
9: Eliminate i from $C _ { n } ;$   
10: Update $W _ { e x i s t }$ ;   
11: end for   
12: Acquire the final model parameters $W _ { e x i s t }$ ;

## 3.5.1. Initialization

First, the central cloud server has existing model parameters for the Federated Transfer Learning Process and the digital twins for the entire VANET. It initializes the collection of clusters that have not yet participated in Federated Transfer Learning.

## 3.5.2. Selection of Training Clusters

We have a collection of clusters $C _ { n }$ that participates in FTL Learning. If we want to take updates from all the clusters in the environment, then $C _ { n }$ $= \mathrm { ~ n ~ }$ . There are j model updates in total. Each cluster contains a group of specific types of vehicles. The central cloud server sends the latest model parameters to a base station that forwards it to the particular cluster in $C _ { n }$ that needs the prediction to be done. The cluster may be chosen based on the requirement to make some predictions. Pre-trained model parameters at the start of the jth update are sent to the base station of a selected cluster’s digital twin in $C _ { n }$ . This allows the vehicle in that cluster to carry out innercluster training to derive the model parameters $W _ { \mathrm { e x i s t } } ^ { - }$ . The details of the inner-cluster training will be covered in the following subsection. It is not observable to the central cloud server.

## 3.5.3. Updates to the Cloud Central Model

We denote the model parameters that the central cloud server sends to and receives from the base station for cluster i in the jth update as $W _ { \mathbf { e x } } ^ { - }$ ist and $W _ { \mathrm { e x i s t } } ^ { + }$ . We calculate the weight of the received model parameters using the scores we received for each vehicle in the given cluster.

$$
F _ { \mathrm { w e i g h t } } ( W _ { \mathrm { e x i s t } } ^ { + } ) = \frac { \sum S r } { | v | }\tag{8}
$$

In the above equation 8, $\sum S r$ is the sum of all the scores that each vehicle DT calculated in the given cluster, and |v| is the number representing all the vehicles in each cluster. If there is only one vehicle in the cluster due to a lack of homogeneous vehicle type in the surroundings, we will consider a cluster of only one vehicle. This overall provides us with the average score of vehicles in a cluster. However, If the output weight is below a certain threshold, regarded as the data quality it is trained on is not good enough to update the global model ModelUpdate(.) at the central cloud server. Cluster i will be removed from the $C _ { n }$ after deciding whether to update the global model parameters based on the average score of clusters. After j updates, a complete network model with parameters $W _ { e x i s t }$ is determined. Algorithm 1 indicates that our suggested architecture framework needs j-cluster iterations of the inner-cluster algorithm for the central cloud server updates.

## 3.6. Inter Cluster Hierarchical Federated Transfer Learning

Algorithm 2 Inner Cluster Hierarchical Federated Transfer Learning   
Input: The model parameters of the cluster’s digital twin that need to be   
fine-tuned $W _ { \mathrm { e x i s t } } ^ { - }$   
Output: The updated fine-tuned trained model parameters $W _ { \mathrm { e x i s t } } ^ { + }$   
1: Inner-Cluster():   
2: (I). For base station:   
3: Receive $W _ { e x i s t } ^ { - }$ from the central cloud server;   
4: Send $W _ { e x i s t } ^ { - }$ to $V _ { h }$ ;   
5: Receive $W _ { e x i s t } ^ { + }$ and Sr from $V _ { h } ;$   
6: Send $W _ { e x i s t } ^ { + }$ and Sr to the central cloud server;   
7:   
8: (II). For head vehicle DT $V _ { h } { : }$   
9: Receive $w _ { h } ^ { - } = W _ { e x i s t } ^ { - }$ from the base station;   
10: Note its post arrange vehicles into set $P o _ { h }$   
11: for all vehicles $v \in P o _ { h }$ do   
12: Send $w _ { h } ^ { - }$ to vehicle DT ;   
13: Receive $w _ { v } \mathrm { ~ , ~ } D Q _ { v }$ and Sr from each vehicle DT;   
14: Update $w _ { h } ^ { - } =$ combine $( w _ { h } ^ { - } \ , w _ { v } \ , D t Q _ { v } \ , \ \mathrm { S r } )$   
15: end for   
16: $w _ { h } ^ { - } = \mathbf { F I N E \_ T U N } ( w _ { h } \ , \ D t Q _ { v } )$   
17: Send $w _ { h } ^ { + }$ and Sr to the base station;   
18:   
19: (III). For route vehicle DT $V _ { r } \colon$   
20: Receive $w _ { r } ^ { - }$ <sup>−</sup> from its previous vehicle DT ;   
21: Note its post arrange vehicles into set $P o _ { r }$   
22: for all vehicles $v \in P o _ { r }$ do   
23: Send $w _ { r } ^ { - }$ to vehicle DT ;   
24: Receive $w _ { v } \mathrm { ~ , ~ } D Q _ { v }$ and Sr from each vehicle DT;   
25: Update $w _ { r } ^ { - } = \mathrm { c o m b i n e } ( w _ { r } ^ { - } \ , w _ { v } \ , D t Q _ { v } \ , \mathrm { S r } )$ ;   
26: end for   
27: w<sup>−</sup> = Fine $\mathbf { T U N } ( w _ { r } \ , D t Q _ { v } )$ ;   
28: Update $D t Q _ { r }$ and Sr;   
29: Send $w _ { r } , D t Q _ { r }$ and Sr to its previous vehicle DT;   
30:   
31: (IV). For edge vehicle DT $V _ { e } { \mathrm { : } }$   
32: Receive $w _ { e } ^ { - }$ from its previous vehicle DT;   
33: w<sup>−</sup> = Fine $\mathbf { T U N } ( w _ { e } \ , \ D t Q _ { e } )$ <sub>;</sub>16   
34: $D t Q _ { e } = | D a t a _ { e v } | ;$   
35: Send $w _ { e } , D t Q _ { e }$ and Sr to its previous vehicle DT;

The Algorithm 2 sequence of collaboration and flowchart across vehicles in a cluster is depicted in Figure 2.In our proposed framework, vehicles are further divided into three groups in a cluster named edge vehicle DT $V _ { e } ,$ route vehicle DT $V _ { r }$ , and head vehicle DT $V _ { h }$ if there is more than one vehicle in a cluster which draws inspiration from earlier work HaghighiFard and Coleri (2024) that separates vehicles in each cluster into cluster head and cluster members.

Assumption 1: The Inner Cluster training occurs on any cluster i digital twin that belongs to the Cn cluster’s collection. We assume that our cluster consists of head, route, and edge vehicles. However, if any cluster is not formed of the same vehicle collection, the relevant portion of the cluster will be discarded.

## 3.6.1. Base Station

After the head vehicle, DT $v _ { h }$ , completes its training, it shares the updated model parameters and data quality score with the base station. The Base Station then shares its model parameters and data quality score with the global model that resides on the cloud server. The relevant global model will be updated only after the cluster has a good aggregated data quality score.

## 3.6.2. Head Vehicle

When there is more than one vehicle in the cluster, there is always a Head Vehicle $v _ { h }$ . No vehicle exists before the head vehicle $v _ { h }$ in the cluster. It is directly connected to the base station if there is more than one vehicle. For HFTL to execute, the base station sends global model parameters to $v _ { h }$ vehicle DT, which forwards global parameters to the vehicle DT that exists next to this vehicle DT in the cluster. To represent the collection of all the vehicles that are next to $v _ { h }$ vehicle, we use $P o _ { h }$ . The updated fine-tuned global parameters $w _ { v } .$ , for constructing model parameters Data Quantity $D Q _ { \iota }$ and Sr is the Data Quality scores are received by the $v _ { h }$ vehicle DT for integration when the next vehicle DT is finished with their fine-tuning.

While integrating, we give weightage to the parameters according to the amount of data on which it is fine-tuned and their data quality scores. After this, the $v _ { h }$ vehicle DT forwards the updated integrated model parameters and scores to the next vehicle’s DT, which remains in the cluster for finetuning. The above procedure continues until all the remaining vehicle DTs are completed with the fine-tuning. Ultimately, the model is fine-tuned using the local data of $v _ { h }$ vehicle DT, and the Data quality score is calculated. The finalized fine-tuned model parameters and Data Quality Score are then sent to the base station.

## 3.6.3. Route Vehicle

A vehicle that functions as a routing vehicle DT is represented by the symbol $v _ { r } ,$ which is a type of vehicle that has at least one predecessor vehicle and one successor vehicle. The $v _ { r }$ vehicle DT gets the model parameters $w _ { r } ^ { - }$ from $v _ { h }$ the predecessor vehicle DT and sends them to all the successor vehicles DT one at a time. Like $v _ { h }$ DT, $v _ { r }$ DT integrates $w _ { r } ^ { - }$ with model parameters received after fine-tuning from the successor vehicle and continues with the same process until the successor vehicles in the collection are done with the same. After that, $v _ { r }$ DT, using its local data, fine-tunes the model parameters and calculates the Data Quality Score. After this, $v _ { r }$ sent it to the predecessor vehicle.

## 3.6.4. Edge Vehicle

The term $v _ { e }$ describes a vehicle DT that is either the only vehicle in the cluster or has only predecessor vehicles but no successor vehicles. If it is the only vehicle, it will forward model parameters directly to the base station; otherwise, it will send them to the predecessor vehicles. When $v _ { e }$ DT gets model parameters from its predecessor vehicle’s DTs, it performs fine-tuning and sends back the model parameters and data quality score(Sr) to the predecessor vehicle DT.

By making clusters of similar vehicle types, we also make a low communication overhead with fewer links to the base station. We also make sure that our trained model is customized to each type of vehicle separately.

## 3.6.5. Example Scenario

We present an example of specific Intelligent Transportation System(ITS) applications where Hierarchical Federated Transfer Learning (HFTL) in Digital Twin Vehicular Networks could be particularly beneficial:

## 1. Trafic Flow Management

HFTL in DT-VANET can enhance trafic flow by aggregating data from sensors of multiple vehicles to accurately predict real-time trafic congestion patterns. The digital twin also adjusts the signal timings

![](images/48cdcac49b688067879ee093295a14a04d0c2cf0c6520693caff0af3b8f35b44.jpg)  
Figure 3: Twelve random topologies Network X generated

in real-time according to trafic conditions, helping to maintain trafic flow.

## 2. Predictive Maintenance

By analyzing the patterns of component failures, the system can predict the maintenance needs of the vehicle’s components(e.g., Brake issues, engine performance).

## 3. Real-time Route Optimization

The system can provide route recommendations based on the weather, trafic, and driver patterns. The digital twin can also optimize routes for emergency vehicles.

## 4. Safety Applications

The system can also alert drivers of a potential collision by acquiring data from multiple vehicles. The architecture can also be beneficial for getting weather updates from vehicles and for providing predictions and alerts to upcoming vehicles about the weather.

5. Smart Parking, Charging and Environmental Solutions Digital Twins helps vehicles find parking and charging spots in crowded areas. It also helps identify the polluted regions of the city.

## 4. PERFORMANCE EVALUATION

This section includes a thorough evaluation of our suggested framework using Hierarchical Federated Transfer Learning (HFTL) with Clustered Federated Learning (CFL), Federated Learning (FL), and Centralized Learning (CL) by diferent performance evaluations. We chose Centralized Learning (CL), Federated Learning (FL), and Clustered Federated Learning (CFL) as our baseline comparisons because CL represents the simplest centralized training method, while FL represents distributed learning. Conversely, CFL extends federated learning with clusters that resonate with our algorithm. The above techniques conflict with HFTL regarding data heterogeneity and resource utilization, efectively highlighting the advantages of HFTL in heterogeneous vehicle environments. The generation of the cluster topology in our experiment is based on the Network X network structure, using a random number of vehicles distributed according to diferent vehicle types. There are twelve clusters, each having more than a random ten number of vehicles. It helps us to analyze results in high-density trafic scenarios, as shown in Figure 3. It also helps to evaluate the system’s ability to handle high-trafic data, which is very helpful for performance metrics results. Although the number of vehicles in each cluster is random, the clustering is based on vehicle type. To ensure the realism of the experiment in each generated cluster, only one head vehicle connects with the base station, and the rest of the vehicles are either routing vehicles or edge vehicles, simulating the actual distribution of diferent vehicle types. In Figure 3, node A represents the head vehicle, and the rest of the node represents vehicles that are either route or edge vehicles. Below are the specific experimental settings.

## 4.1. Experiment Settings

## 4.1.1. Experimental Setup

Network X<sup>1</sup>creates the random Inner Cluster topologies while Torch<sup>2</sup> implements our suggested federated transfer learning framework. The Google ColabGoogle Colab (2022) is used for implementation.

## 4.1.2. Setting for Cluster

We form twelve random clusters, each with a variable number of vehicles but the same vehicle type as described in table 2. As mentioned in Figure 2, if each cluster has more than one vehicle, there would be a single cluster head; otherwise, in the case of the single vehicle, it would act as an edge vehicle. In our case for the evaluation of performance metrics, we keep the number of vehicles to more than a random ten number of vehicles because it will help us to evaluate the system’s ability to handle high trafic data as shown in Figure 3

## 4.1.3. Datasets

For our performance metrics, we have conducted our experiments on the real-world dataset named vehicle mobility trace <sup>3</sup>. The data sets comprise real-time values, such as vehicle position, angle, vehicle ID, coordinates, speed, etc. As part of preprocessing, we first perform data cleaning, which handles outliers by replacing them with extreme values and using interpolation techniques to fill in missing data. After that, we use a standard scalar to normalize the data. It makes sure that all features are on the same scale. We split the data set in such a way that 80 % is used as training while 20% as testing. The main reason behind this data split is to make sure there is enough data for training to retain some portion for evaluation. However, we also used cross-validation to verify the model performance across several data subsets to guarantee the reliability and stability of the results. We assigned vehicles a numerical ID according to their category, as mentioned in Table 2. This will help us implement Hierarchical Federated Learning by randomly clustering vehicles according to their similar category.

## 4.1.4. Baseline Studies

We will compare centralized learning, federated learning, clustered federated learning, and federated transfer learning. For comparison, we have used centralized, federated learning, and clustered federated learning as they are most closely related to federated transfer learning. Furthermore, while implementing the search for the most eficient hyperparameters, we utilized a checkpoint that led us to set the number of epochs to 200, the learning rate (α) to 0.1, and the batch size to 128 for both the proposed Federated transfer learning and the baseline algorithms.

## 4.1.5. Performance Metrics

We computed average Model Accuracy, Training Time, Resource Consumption (Computational), Communication Overhead, Latency, Convergence Time, and Throughput. Model accuracy tells us our model’s accuracy in correctly predicting output. Training time refers to the duration required to complete the training process. Resource Consumption refers to the utilization of computational resources during the training process. Communication Overhead describes the amount of data exchanged during transmission. The amount of time required with an input and producing output (prediction) by the model is known as Latency. Convergence Time is the time needed for the training procedure to attain the required accuracy or stable condition (in our case, 90% accuracy). The number of predictions a model can make in an amount of time given is known as Throughput.

![](images/91f911585d00c54347524f62a5098c022de5b0a5881eeeb6b1674812b036de1c.jpg)  
Figure 4: Performance metrics graph results of our proposed Federated Transfer Learning algorithm with Clustered Federated Learning, Federated Learning, and Centralized Learning

## 4.1.6. Experimental Results and Performance Analysis

We have compared existing centralized learning, federated learning, and clustered federated learning with federated transfer learning. We used a realtime vehicular mobility trace CSV file dataset to make our findings realistic. Figure 4 shows the graph for the performance metrics with 100 simulations to eliminate any bias scenario. Table 3 shows the average performance of our HFTL algorithm over CL and FL.

Table 3: Experimental results regarding the average performance metrics across learning algorithms with Federated Transfer Learning Algorithm
<table><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>ModelAccur</td><td rowspan=1 colspan=1>TrainTime</td><td rowspan=1 colspan=1>ResourConsu</td><td rowspan=1 colspan=1>CommOhead</td><td rowspan=1 colspan=1>Latncy</td><td rowspan=1 colspan=1>ConverTTime</td><td rowspan=1 colspan=1>hrput</td></tr><tr><td rowspan=1 colspan=1>Centralized Learning</td><td rowspan=1 colspan=1>0.708</td><td rowspan=1 colspan=1>2.03s</td><td rowspan=1 colspan=1>20.19%</td><td rowspan=1 colspan=1>6.19</td><td rowspan=1 colspan=1>2.323s 0.23</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>6.20</td></tr><tr><td rowspan=1 colspan=1>Federated Learning</td><td rowspan=1 colspan=1>0.721  1.53</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>15.65%</td><td rowspan=1 colspan=1>6.25</td><td rowspan=1 colspan=1>2.314s 0.12</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>6.23</td></tr><tr><td rowspan=1 colspan=1>Clustered Federated Learning0</td><td rowspan=1 colspan=1>.752  1.46</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>14.96%</td><td rowspan=1 colspan=1>6.21</td><td rowspan=1 colspan=1>2.304s 0.08</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>6.24</td></tr><tr><td rowspan=1 colspan=1>Federated Transfer Learning</td><td rowspan=1 colspan=1>0.822  1.02</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>8.30%</td><td rowspan=1 colspan=1>6.28</td><td rowspan=1 colspan=1>2.240s 0.04</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>6.35</td></tr></table>

The Federated transfer learning-based algorithm has the highest model accuracy average because it uses a pre-trained model according to the specific vehicle type. So, its prediction accuracy is better on average than the other two models.

For the performance metric Training Time, on average, of a hundred different simulations, our algorithm takes less time because fine-tuning a model requires less time in seconds compared to training a model from scratch.

The federated transfer learning model requires fewer computational resources for resource consumption than the other models. That is because fine-tuning requires, on average, fewer computational resources than training from scratch, especially when we have customized models for each vehicle type.

The federated transfer learning algorithm has slightly more communication overhead when compared to Centralized learning(CL), Federated learning(FL), and Clustered Federated Learning (CFL). First, a pre-trained model must be fine-tuned and optimized based on each vehicle’s personalized requirements, which will take up more communication overhead. Centralized learning(CL), Federated learning (FL), and Clustered Federated Learning (CFL) don’t require pre-trained models for training, so their communication overhead is lower than our algorithm’s. Having a slight diference in communication overhead does not afect our algorithm eficiency. However, the increased communication overhead is ofset by significant improvements in accuracy and latency, giving HFTL a clear advantage in real-time decisionmaking and prediction precision, particularly in scenarios with high vehicle heterogeneity.

The Federated Transfer Learning algorithm has less latency than Centralized learning(CL), Federated learning(FL), and Clustered Federated Learning (CFL). This is due to its use of pre-trained models that require less time to make predictions. Because of pre-trained model fine-tuning, this algorithm also has more throughput, faster convergence, a hierarchical structure, and low resource usage compared to Federated and Centralized learning.

![](images/1549b8330478c48872c2f2d981ac7116e3532afb0463fbc62073c676adce76d2.jpg)

![](images/b05b709247358405b766cd7c453bfb667d8b2380c0f78e5bc39a35b0ebba18d9.jpg)  
Figure 5: Graphical Comparison of Diferent Algorithm’s Model Accuracy, Resource Consumption, Convergence Time, Training Time, Communication Overhead, Latency, Throughput

Figure 5 shows the bar graph comparison of HFTL over CFL, FL, and CL for Model Accuracy, Training Time, and Resource Consumption. HFTL outperforms the other algorithms. However, Figure 5 shows the bar graph comparison of HFTL over CFL, FL, and CL for Communication Overhead, where the other algorithm’s communication overhead is slightly lower than HFTL.

## 4.1.7. Scalability Experimental Results and Performance Analysis

The above results are tailored to a few vehicles and specific network conditions. We want to see how our algorithm performs if we gradually increase the number of vehicles in our network to check whether it remains efective. Figure 6 shows the performance of learning algorithms for scalability metrics. In the Dynamic Environment, as compared to Clustered Federated, Federated, and Centralized Learning, our proposed algorithm performs better except for some underperformance in communication overhead. Table 4 shows the average scalability metrics values for the algorithms.

Figure 7 shows these algorithms’ diferences in scalability performance

![](images/9fb9387d237f55dbe9ce89b64a99d999cff0450270e901866d6dca12bc0fe004.jpg)

![](images/409b35d9a4da8b4dd2a5eb579f7ebd23506d5ecfe875df48004a29b2bdf93a77.jpg)

![](images/ae1419cfbc0bf5303ac4c71daeba1a715166adad5625f0431a854bc058698e98.jpg)

![](images/d9618922423d529bf7413e8e0c93dd7d7606da7729dc56d04b219e13ad891524.jpg)

![](images/fb4a719d1c9ab7e82e5c9a33204c561a341e716ef27d84b8200874b18e2ba878.jpg)

![](images/0adca5f5c6f098ac5a9aefb7c6c81faa9205a366ac2ab4a4d7f7601b88703e09.jpg)

![](images/4eb0bc8376f1ef2a1d9b7b2f876771394aa3cb38676c40ec193ba4c373fa205e.jpg)  
Figure 6: Scalability of our Federated Transfer Algorithm with Clustered Federated, Federated, and Centralized Learning

Table 4: Experimental results regarding the average scalability metrics across the learning algorithms with Federated Transfer Learning Algorithm
<table><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>ModelAccur</td><td rowspan=1 colspan=1>TrainTime</td><td rowspan=1 colspan=1>ResourConsu</td><td rowspan=1 colspan=1>CommOhead</td><td rowspan=1 colspan=1>Latncy</td><td rowspan=1 colspan=1>ConverTTime</td><td rowspan=1 colspan=1>hrput</td></tr><tr><td rowspan=1 colspan=1>Centralized Learning</td><td rowspan=1 colspan=1>0.682  6.01</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>54.06%</td><td rowspan=1 colspan=1>6.77</td><td rowspan=1 colspan=1>2.47s</td><td rowspan=1 colspan=1>45.82s</td><td rowspan=1 colspan=1>6.79</td></tr><tr><td rowspan=1 colspan=1>Federated Learning</td><td rowspan=1 colspan=1>0.689  6.07</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>54.73%</td><td rowspan=1 colspan=1>6.82</td><td rowspan=1 colspan=1>2.46s</td><td rowspan=1 colspan=1>45.72s</td><td rowspan=1 colspan=1>6.81</td></tr><tr><td rowspan=1 colspan=1>Clustered Federated Learning</td><td rowspan=1 colspan=1>0.68   5.96</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>53.61%</td><td rowspan=1 colspan=1>6.81   2.45</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>45.46s</td><td rowspan=1 colspan=1>6.84</td></tr><tr><td rowspan=1 colspan=1>Federated Transfer Learning</td><td rowspan=1 colspan=1>0.823  2.78</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>25.87%</td><td rowspan=1 colspan=1>6.90   2.42</td><td rowspan=1 colspan=1>s  29.29</td><td rowspan=1 colspan=1>s</td><td rowspan=1 colspan=1>6.91</td></tr></table>

metrics output as a bar graph. <sup>5</sup>

Scalability testing in large-scale VANETs shows that HFTL maintains high model accuracy and fast convergence as the number of nodes increases. However, as the network grows continuously, communication overhead and resource consumption may become limiting factors according to the results. Future work could explore communication compression techniques or asynchronous update mechanisms to further optimize HFTL’s performance in large-scale networks. As we increase the number of vehicles, the communication overhead increases based on the following:

1. The model updates that need to be communicated also increase.

![](images/95e8278c1d9e52164dc1d9d9c90688eb3b58fa5a79a4cd45fcad2652f186937a.jpg)

![](images/2d7725d8803415e9513b781101a555891c6b4f84aadf33e0f23f4e26f2ef873e.jpg)

![](images/2498c55a288549118f4f0b5f8a375207067c439fd62e8e498ae1ee99247f6a82.jpg)  
Figure 7: Scalability of our Federated Transfer Algorithm with Federated and Centralized Learning

2. Aggregation complexity also increases communication overhead as vehicle updates are integrated with the head vehicle DT and sent to the central server.

3. More Communication Rounds are needed to achieve convergence because the number of vehicles grows, increasing communication overhead.

As the number of vehicles increases, the communication overhead increases because of more data trafic and aggregation complexity. A slightly higher communication overhead than the other algorithms will not afect eficiency, as other performance metrics have greatly improved.

## 5. Challenges and Limitations

Although the above results show the significance of our proposed research, some challenges and limitations are associated with it.

1. Data heterogeneity: Significant diferences in the data distribution between digital twins could result in less accuracy, more training time, and ineficient model convergence.

2. Network limitations: Real-time synchronization and data transfer between vehicles and their respective digital twins can be afected by environments with erratic, unstable, or low-bandwidth communication links.

## 3. Model Bias:

The global model may inherit bias if the local training data is biased

4. Dynamic Contexts:

The hierarchical model may take time to adjust in real-time where the vehicle patterns change quickly

5. Heterogeneous VANET Types:

Our HFTL works well in urban and highways. It may struggle in sparely populated regions where data is less available.

## 6. Conclusion

This paper proposes the Hierarchical Federated Transfer Learning architecture and algorithm for the Digital Twin-based Vehicular Ad Hoc Network that provides faster and more accurate predictions. We have also formed clusters based on the specific vehicle type to make precise decision-making. We first designed the high-level architecture of Hierarchical Federated Learning. Then, we describe algorithms for our proposed architecture that are time-eficient and more accurate. Our experimental results demonstrate that HFTL outperforms Clustered Federated Learning (CFL), Federated Learning (FL), and Centralized Learning (CL) across several key metrics, including model convergence speed, throughput, and resource consumption. We also scale our network to see how our technique performs in the dynamic environment. In the future, we plan to conduct further research in the following areas: First, we aim to explore eficient communication compression techniques to reduce communication overhead. Second, we investigate asynchronous model update strategies to improve system real-time performance. Lastly, we intend to optimize lightweight blockchain implementations to ensure data privacy and security in large-scale vehicular networks.

## References

M. Noor-A-Rahim, Z. Liu, H. Lee, G. M. N. Ali, D. Pesch, and P. Xiao, “A survey on resource allocation in vehicular networks,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 2, pp. 701–721, 2020.

Q. Zia, M. S. Farooq, and A. Abid, “Improving response time of vehicular ad hoc networks (VANET),” 2016.

C. He, T. H. Luan, R. Lu, Z. Su, and M. Dong, “Security and privacy in vehicular digital twin networks: Challenges and solutions,” IEEE Wireless Communications, 2022.

L. U. Khan, E. Mustafa, J. Shuja, F. Rehman, K. Bilal, Z. Han, and C. S. Hong, “Federated learning for digital twin-based vehicular networks: Architecture and challenges,” IEEE Wireless Communications, 2023.

S. AbdulRahman, H. Tout, H. Ould-Slimane, A. Mourad, C. Talhi, and M. Guizani, “A survey on federated learning: The journey from centralized to distributed on-site learning and beyond,” IEEE Internet of Things Journal, vol. 8, no. 7, pp. 5476–5497, 2020.

B. Li, Y. Wu, J. Song, R. Lu, T. Li, and L. Zhao, “DeepFed: Federated deep learning for intrusion detection in industrial cyber–physical systems, IEEE Transactions on Industrial Informatics, vol. 17, no. 8, pp. 5615–5624, 2020.

L. U. Khan, W. Saad, Z. Han, and C. S. Hong, “Dispersed federated learning: Vision, taxonomy, and future directions,” IEEE Wireless Communications, vol. 28, no. 5, pp. 192–198, 2021.

J. Konečn\`y, H. B. McMahan, F. X. Yu, P. Richtárik, A. T. Suresh, and D. Bacon, “Federated learning: Strategies for improving communication eficiency,” arXiv preprint arXiv:1610.05492, 2016.

B. Li, Y. Jiang, Q. Pei, T. Li, L. Liu, and R. Lu, “FEEL: Federated end to-end learning with non-IID data for vehicular ad hoc networks,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 9, pp. 16 728–16 740, 2022.

S. S. Sepasgozar and S. Pierre, “Fed-NTP: A federated learning algorithm for network trafic prediction in VANET,” IEEE Access, vol. 10, pp. 119 607– 119 616, 2022.

L. Zhang and B. Zhang, “Hierarchical machine learning—a learning methodology inspired by human intelligence,” in International Conference on Rough Sets and Knowledge Technology. Springer, 2006, pp. 28–30.

F. Gonçalves, J. Macedo, and A. Santos, “An intelligent hierarchical security framework for VANETs,” Information, vol. 12, no. 11, p. 455, 2021.

M. Ahmed, U. Sardar, S. Ali, S. Alam, M. Patterson, and I. U. Khan, “Robust brain age estimation via regression models and MRI-derived features,” International Conference on Computational Collective Intelligence, pp. 661– 674, 2023.

M. S. HaghighiFard and S. Coleri, “Hierarchical federated learning in multihop cluster-based VANETs,” arXiv preprint arXiv:2401.10361, 2024.

Y. Liu, Y. Kang, C. Xing, T. Chen, and Q. Yang, “A secure federated transfer learning framework,” IEEE Intelligent Systems, vol. 35, no. 4, pp. 70–82, 2020.

S. Otoum, N. Guizani, and H. Mouftah, “On the feasibility of split learning, transfer learning and federated learning for preserving security in ITS systems,” IEEE Transactions on Intelligent Transportation Systems, 2022.

M. Ahmed, H. Chai, H. Wang, H. Venkateswara, and M. Patterson, “Epi-Former: Learning antigen–antibody interactions for epitope prediction via geometric deep learning,” arXiv preprint arXiv:2606.04154, 2026.

Q. Zia, “A survey of data-centric protocols for wireless sensor networks,” Computer Science Systems Biology, OMICS Publishing Group, vol. 8, no. 3, pp. 127–131, 2015.

S.-S. Wang and Y.-S. Lin, “PassCAR: A passive clustering aided routing protocol for vehicular ad hoc networks,” Computer Communications, vol. 36, no. 2, pp. 170–179, 2013.

S. A. Rashid, L. Audah, M. M. Hamdi, and S. Alani, “Prediction based eficient multi-hop clustering approach with adaptive relay node selection for VANET,” J. Commun., vol. 15, no. 4, pp. 332–344, 2020.

M. Ahmed, N. Taj, I. U. Khan, H. Venkateswara, and M. Patterson, “ChiMERa-Bench: A benchmark dataset for epitope-specific antibody design,” ICLR 2026 Workshop on Generative and Experimental Perspectives for Biomolecular Design, 2026.

A. Temurnikar, P. Verma, and G. Dhiman, “A PSO enable multi-hop clustering algorithm for VANET,” International Journal of Swarm Intelligence Research (IJSIR), vol. 13, no. 2, pp. 1–14, 2022.

Q. Zia, C. Wang, S. Zhu, and Y. Li, “Priority based inter-twin communication in vehicular digital twin networks,” International Journal of Parallel, Emergent and Distributed Systems, pp. 1–16, 2024.

Y. Dai and Y. Zhang, “Adaptive digital twin for vehicular edge computing and networks,” Journal of Communications and Information Networks, vol. 7, no. 1, pp. 48–59, 2022.

M. Ahmed, S. Ali, A. Jan, I. U. Khan, and M. Patterson, “Improved graphbased antibody-aware epitope prediction with protein language modelbased embeddings,” International Conference on Computational Advances in Bio and Medical Sciences, pp. 290–302, 2025.

L. U. Khan, W. Saad, D. Niyato, Z. Han, and C. S. Hong, “Digital-twinenabled 6G: Vision, architectural trends, and future directions,” IEEE Communications Magazine, vol. 60, no. 1, pp. 74–80, 2022.

L. U. Khan, Z. Han, W. Saad, E. Hossain, M. Guizani, and C. S. Hong, “Digital twin of wireless systems: Overview, taxonomy, challenges, and opportunities,” IEEE Communications Surveys & Tutorials, vol. 24, no. 4, pp. 2230–2254, 2022.

Google Colab, Jun 2022. [Online]. Available: https://colab.research. google.com/