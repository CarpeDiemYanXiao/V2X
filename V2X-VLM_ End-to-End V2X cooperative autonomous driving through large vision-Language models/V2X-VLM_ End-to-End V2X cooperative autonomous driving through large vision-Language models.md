![](_page_0_Picture_1.jpeg)

Contents lists available at ScienceDirect

# Transportation Research Part C

journal homepage: www.elsevier.com/locate/trc

![](_page_0_Picture_5.jpeg)

# V2X-VLM: End-to-End V2X cooperative autonomous driving through large vision-Language models

Junwei You <sup>b</sup> , Zhuoyu Jiang <sup>c</sup> , Zilin Huang <sup>b</sup> , Haotian Shi a,b,<sup>∗</sup> , Rui Gan <sup>b</sup> , Keshu Wu <sup>d</sup> , Xi Cheng <sup>e</sup> , Xiaopeng Li <sup>b</sup> , Bin Ran <sup>b</sup>

- <sup>a</sup> *College of Transportation, Tongji University, Shanghai, 201804, China*
- <sup>b</sup> *Department of Civil and Environmental Engineering, University of Wisconsin-Madison, Madison, WI, 53706, USA*
- <sup>c</sup> *Kuaishou Technology, Beijing, 100085, China*
- <sup>d</sup> *Zachry Department of Civil and Environmental Engineering, Texas A&M University, College Station, TX, 77840, USA*
- <sup>e</sup> *Systems Engineering Program, Cornell University, Ithaca, NY, 14853, USA*

## a r t i c l e i n f o

#### *Keywords:* End-to-end autonomous driving V2X Cooperation Vision-language model Knowledge distillation Contrastive learning Trajectory planning

## a b s t r a c t

Vehicle-to-everything (V2X) cooperation has emerged as a promising paradigm to overcome the perception limitations of classical autonomous driving by leveraging information from both egovehicle and infrastructure sensors. However, effectively fusing heterogeneous visual and semantic information while ensuring robust trajectory planning remains a significant challenge. This paper introduces V2X-VLM, a novel end-to-end (E2E) cooperative autonomous driving framework based on vision-language models (VLMs). V2X-VLM integrates multiperspective camera views from vehicles and infrastructure with text-based scene descriptions to enable a more comprehensive understanding of driving environments. Specifically, we propose a contrastive learning-based mechanism to reinforce the alignment of heterogeneous visual and textual characteristics, which enhances the semantic understanding of complex driving scenarios, and employ a knowledge distillation strategy to stabilize training. Experiments on a large real-world dataset demonstrate that V2X-VLM achieves state-of-the-art trajectory planning accuracy, significantly reducing L2 error and collision rate compared to existing cooperative autonomous driving baselines. Ablation studies validate the contributions of each component. Moreover, the evaluation of robustness and efficiency highlights the practicality of V2X-VLM for real-world deployment to enhance overall autonomous driving safety and decision-making.

## **1. Introduction**

End-to-end (E2E) autonomous driving has emerged as a compelling paradigm by directly mapping raw sensor inputs to vehicle control commands, offering a simplified alternative to labor-intensive modular pipelines (Hu et al. (2023b), Jiang et al. (2023), Li et al. (2024d), Zheng et al. (2024a)). While these classical E2E methods reduce hand-engineered complexity, they often struggle to interpret complex traffic scenarios without higher-level semantic reasoning.

Emerging developments in foundation models, especially large language models (LLMs) and vision-language models (VLMs), introduce richer multimodal understanding, which enables E2E pipelines to better interpret visual scenes and textual cues (Xu et al. (2024b), Sima et al. (2024), Shao et al. (2024), Tian et al. (2024), Fu et al. (2024), Ma et al. (2024), Huang et al. (2024b), Hwang

*E-mail address:* shihaotian95@tongji.edu.cn (H. Shi).

https://doi.org/10.1016/j.trc.2025.105457

<sup>∗</sup> Corresponding author.

![](_page_1_Figure_2.jpeg)

**Fig. 1.** Overview of end-to-end autonomous driving pipelines. (a) A cooperative driving scenario where infrastructure sensors supplement the ego vehicle's limited field of view; (b.1) the classical end-to-end pipeline that relies solely on-board sensor data; (b.2) a VLM-based end-to-end system that integrates multimodal reasoning within a single vehicle; (b.3) UniV2X-the pioneering end-to-end cooperative autonomous driving pipeline that fuses vehicle and infrastructure data; and (b.4) our proposed V2X-VLM framework, which leverages large VLM to unify multimodal data for robust end-to-end trajectory planning.

et al. (2024)). However, since both classical and VLM-enhanced E2E systems rely solely on a single vehicle's sensor data, they remain limited in challenging conditions where supplemental context is needed, such as occlusions and blind spots.

Cooperative autonomous driving extends beyond the single-vehicle view by leveraging Vehicle-to-Everything (V2X) communication to integrate data from both vehicles and infrastructure. As illustrated in Fig. 1(a), infrastructure sensors contribute crucial contextual information that complements the ego vehicle's field of view. In this case, the early cooperative methods focused mainly on vehicle-to-vehicle (V2V) data fusion (Wang et al. (2020), Cui et al. (2022), Xu et al. (2022b), Hu et al. (2023a), Xu et al. (2024a)) to enhance perception under multi-agent settings. Nevertheless, they typically addressed only partial tasks such as detection or occupancy mapping, which fell short of offering fully integrated planning and control, limiting their ability to provide E2E solutions for real-world autonomous driving scenarios.

To address this problem, more recent studies culminate in UniV2X (Yu et al. (2024)), the first E2E cooperative autonomous driving pipeline that fuses data from both vehicles and infrastructure to produce an integrated framework for comprehensive perception and planning. Fig. 1(b.1-b.3) illustrates the evolution of existing E2E autonomous driving approaches from the classical pipeline to UniV2X-represented cooperative pipeline. However, UniV2X still relies on traditional deep learning architectures that struggle to unify heterogeneous sensor data for complete semantic understanding, especially given the increased complexity introduced by infrastructure-side inputs. These data streams not only capture a broader field of view, but also reflect more intricate road geometries, additional traffic agents, and a wealth of environmental context not observed by onboard vehicle sensors. Thus, there is a pressing need for more advanced approaches capable of bridging these disparate perspectives and extracting cohesive high-level representations to drive a more effective cooperative E2E decision-making.

Inspired by the success of VLMs in single-vehicle E2E setups - where advanced scene understanding and reasoning are achieved through the multimodal fusion of visual data and textual cues (Xu et al. (2024b), Sima et al. (2024), Shao et al. (2024), Tian et al. (2024), Fu et al. (2024), Ma et al. (2024), Huang et al. (2024b), Hwang et al. (2024), Jiao and Fang (2024), Long et al. (2024), Liu et al. (2025), Zhang and Nie (2024))-we posit that integrating VLMs in a cooperative autonomous driving framework could enhance joint perception, situation awareness, and planning accuracy. Motivated by this, we propose V2X-VLM, a novel VLM-based E2E cooperative autonomous driving framework, as shown in Fig. 1(b.4). Unlike single-modality E2E architectures (Hu et al. (2023b), Jiang et al. (2023), Xu et al. (2024a)), V2X-VLM unifies multiperspective sensor data from vehicles and infrastructure, augmenting them with text prompts for advanced spatial semantic reasoning.

Specifically, to further reinforce the cross-modal alignment of contextual and visual information in traffic conditions, we introduce a contrastive learning-based feature alignment mechanism that achieves more discriminative situation awareness for effective trajectory planning. Furthermore, to improve the efficiency of large VLM training, we incorporate a teacher-student distillation strategy during fine-tuning, which produces a smoother learning process. The main contributions are as follows:

- We introduce a novel VLM-based end-to-end cooperative autonomous driving framework that unifies sensor data from both vehicles and infrastructure with textual scene descriptions, thus enhancing trajectory planning through advanced multimodal understanding.
- We propose a contrastive learning-based alignment mechanism that explicitly synchronizes visual inputs with their corresponding textual cues, resulting in a more discriminative understanding of complex driving scenarios.
- We integrate a knowledge distillation strategy during fine-tuning to stabilize the learning process and efficiently transfer rich multimodal representations.
- We evaluated the proposed framework on the DAIR-V2X dataset (Yu et al. (2022)), demonstrating significant improvements over the state-of-the-art methods. The robustness and efficiency evaluation validates the practicality of V2X-VLM for real-world deployment.

## **2. Related work**

#### *2.1. End-to-End autonomous driving*

E2E autonomous driving directly maps raw sensor data to vehicle control commands without relying on a fully disassembled perception-prediction-planning pipeline. Classical E2E approaches typically employ convolutional or transformer-based architectures to infer ego-vehicle movements from onboard camera views (Hu et al. (2023b), Jiang et al. (2023), Hu et al. (2022a), Shao et al. (2023), Ye et al. (2023), Chen et al. (2024), Sun et al. (2024), Li et al. (2024c), Guo et al. (2024), Yuan et al. (2024)). These methods reduce hand-engineered components and simplify the pipeline, but they often struggle under occlusions, dense multi-agent interactions, and visually ambiguous urban scenes, which limits their reliability in complex traffic.

To address these issues, several technical directions have emerged. Generative E2E planners leverage diffusion and related models to capture epistemic and aleatoric uncertainty in future states (Zheng et al. (2024a), Liao et al. (2024)). Occupancy- and BEV-based formulations reason over spatial occupancy or vectorized maps to strengthen geometric grounding (Mahjourian et al. (2022), Li et al. (2024b)). Gaussian-based representations introduce continuous and compact 3D parameterizations for scene structure and motion (Zheng et al. (2024b)). World-model-based methods learn latent dynamics for closed-loop rollouts and planning in imagination (Li et al. (2024a), Gao et al. (2024), Wang et al. (2024a,b)). While these paradigms advance scene understanding and motion forecasting, they still struggle to capture the semantic abstractions and high-level reasoning that complex driving scenarios demand.

Motivated by foundation models, recent work has explored integrating LLMs and VLMs into end-to-end pipelines to enhance semantic grounding and reasoning. For example, DriveGPT-4 (Xu et al. (2024b)) and DriveLM (Sima et al. (2024)) leverage pretrained large language models to inject natural-language reasoning capabilities into planning. LMDrive (Shao et al. (2024)) and DriveVLM (Tian et al. (2024)) focus on aligning visual observations with language-based instructions or BEV features, while Drive-CoT (Fu et al. (2024)) explicitly incorporates chain-of-thought prompting for more interpretable decision making. Other approaches, such as Dolphins (Ma et al. (2024)), VLM-Drive (Huang et al. (2024b)), EMMA (Hwang et al. (2024)), LAVIDA (Jiao and Fang (2024)), LongVLM (Long et al. (2024)), CoDriveVLM (Liu et al. (2025)), and InternDrive (Zhang and Nie (2024)), demonstrate the potential of multimodal vision-language alignment for trajectory planning and multi-agent interaction reasoning. By aligning visual features with textual cues such as driving goals, traffic context, and implicit traffic rules, these models improve situation awareness and intent grounding. However, these models are still limited by their single-vehicle perspective. Problems such as inaccurate prompts, weak coordinate grounding, and inconsistent cross-view reasoning make them fragile in occluded or highly interactive traffic scenes, where cooperative inputs are needed.

In parallel, the field also distinguishes between open-loop approaches, which generate future trajectories or control commands without feedback, and closed-loop approaches that continuously update actions in real time. Most of the aforementioned paradigms remain open-loop, although some recent methods have attempted closed-loop integration, typically under simulated or controlled conditions (Huang et al. (2024b), Wang et al. (2024b), Jia et al. (2025), Shao et al. (2024), Huang et al. (2024a)).

Compared to existing paradigms, our proposed V2X-VLM capitalizes on multi-perspective inputs from vehicles and roadside infrastructure to expand the ego view and strengthen spatio-temporal coverage in occluded or visually ambiguous situations. Unlike single-modality or single-vehicle pipelines, V2X-VLM unifies heterogeneous visual streams with textual prompts in a large VLM backbone to enable collaborative semantic reasoning for diverse road scenarios. This design, coupled with explicit cross-modal alignment and stabilized training, delivers more reliable motion planning and offers a principled foundation for cooperative, efficiency-aware E2E driving.

#### *2.2. Cooperative autonomous driving*

Cooperative autonomous driving leverages V2V and vehicle-to-infrastructure (V2I) communication to integrate distributed sensing and decision making. Although early work primarily focused on the fusion of data between multiple vehicles to improve perception quality (Xu et al. (2022a), Yu et al. (2023), Lu et al. (2023), Chen et al. (2023), Hu et al. (2022b), Cui et al. (2022)), recent efforts emphasize infrastructure-based sensing that broadens the field of view for enhanced situational awareness. This shift reflects the limitation of V2V fusion, which is constrained by shared occlusions and low viewpoints. Infrastructure-side sensing provides a top-down perspective of complex road networks and captures broader traffic dynamics involving diverse participants (Yi et al. (2024), Mo et al. (2024), Khan et al. (2025)). For example, at occluded intersections or curved on-ramps, infrastructure cameras can observe vehicles and pedestrians outside the line of sight of onboard sensors. Similarly, in dense urban traffic with multi-lane merges, infrastructure input enables a more complete understanding of agent interactions beyond what nearby vehicles can perceive alone. UniV2X (Yu et al. (2024)) represents a critical step, which integrates vehicle and infrastructure data into an E2E pipeline for comprehensive perception and planning.

Yet, UniV2X still relies on traditional deep learning models that often struggle with the semantic richness and heterogeneity of multi-perspective data streams. In contrast, V2X-VLM addresses these limitations of UniV2X by unifying heterogeneous vehicle and infrastructure data within a large VLM backbone, augmented by textual prompts that inject high-level contextual cues. This design enables stronger semantic grounding, improved robustness to occlusions, and better generalization across diverse traffic scenarios, resulting in more accurate and reliable E2E planning.

#### 3. Problem formulation

The objective of the proposed V2X-VLM framework is to generate an action-level plan for the ego vehicle in the form of a future trajectory. In line with the E2E paradigm, this trajectory encapsulates the sequence of control decisions that the ego vehicle would take, while the influence of surrounding agents is implicitly modeled through the fused multimodal inputs. Specifically, the vehicle-side and infrastructure-side images provide spatiotemporal observations of both the ego vehicle and neighboring agents, and the VLM-generated textual prompt *E* encodes high-level contextual information such as scene descriptions, interaction cues, and the ego's current position (see Appendix E). In this way, although the output is expressed as an ego trajectory, the decision-making process inherently accounts for surrounding-agent behaviors and traffic dynamics.

Formally, let  $I_v \in \mathbb{R}^{H_v \times W_v \times 3}$  represent the camera input of the ego vehicle of height  $H_v$  and width  $W_v$ ,  $I_i \in \mathbb{R}^{H_i \times W_i \times 3}$  denote the image data from the infrastructure cameras of height  $H_i$  and width  $W_i$ , and E represent the textual prompt. The model outputs a discrete sequence of 2D positions for the ego vehicle over a time horizon T, corresponding to the planned future action sequence:

$$\tau = \{(x_t, y_t) \mid t = 1, 2, \dots, T\},\tag{1}$$

where  $(x_t, y_t)$  denotes the planned position of the ego vehicle in the global ground-plane coordinate system at time t. Our end-to-end model  $F(\cdot)$  learns to generate  $\tau$  directly from  $(I_v, I_i, E)$ , trained by minimizing the discrepancy between the planned trajectory  $\tau$  and the ground-truth trajectory  $\tau^*$ :

$$\min \mathcal{L}(\tau, \tau^*) = \min \mathcal{L}(F(I_v, I_i, E), \tau^*), \tag{2}$$

where  $\mathcal{L}(\cdot)$  is a suitable loss function. By formulating the problem in terms of ego trajectory generation, we follow prior E2E planning-oriented frameworks (Hu et al. (2023b), Jiang et al. (2023), Yu et al. (2024)) in treating the trajectory as the surrogate for the ego vehicle's control actions. At the same time, surrounding-agent behaviors and road-level interactions are incorporated through the integrated vehicle-infrastructure visual observations and semantically rich textual prompts, which enable the model to reason about multi-agent dynamics in a holistic manner. While our formulation adopts a supervised learning style by mimicking logged trajectories, the safety dimension is not ignored: we explicitly evaluate collision rate and robustness under perturbations as safety proxies, and the inclusion of cooperative V2X views allows the model to implicitly account for multi-agent decision interactions.

## 4. Methodology

## 4.1. V2X-VLM Framework

The overall framework of V2X-VLM is demonstrated in Fig. 2.

As addressed previously, V2X-VLM generates ego-vehicle trajectories by fusing heterogeneous visual inputs from vehicles and infrastructure, together with textual prompts that provide high-level semantic context.

Specifically, the vehicle camera image  $I_v$  captures the ego vehicle's local scene, including nearby lanes and dynamic agents. The infrastructure image  $I_i$ , collected from cameras placed at strategic points like intersections, provides a wider view of broader traffic patterns and pedestrian activities that might not be visible from the vehicle's perspective. In addition to visual data, the framework incorporates a text prompt E, which includes semantic textual information relevant to the driving context. It encompasses three key elements: scene description resulted from the ability of VLM to understand and interpret the complex driving environment and crafted by human; the current position of the ego vehicle serving as the planning basis; as well as the explicit planning task description. The ego vehicle's position in the text prompt is represented in the Virtual World Coordinate System provided by the dataset, where the x-y plane is parallel to the ground plane and the z-axis is positive upwards. The real-time position and orientation obtained from GPS/IMU are further transformed into this Virtual World Coordinate System to ensure data security and facilitate research, while maintaining spatial consistency across all sensor modalities. The textual prompts are designed for scene description rather than decision provision. They describe road geometry, traffic participants, and traffic signals but do not contain explicit control or action guidance, as the latter is produced by the model itself in the planned trajectory. While not prescribing actions, these scene descriptions indirectly inform trajectory planning by providing safety-critical context (e.g., crosswalk locations, surrounding infrastructure). Since the prompts are mainly general scene descriptions without precise decision-level information, the majority of their content can be directly derived from a separate VLM with only minor human refinement. Since they describe the current scene at each time step, they are temporally synchronized with the corresponding vehicle-side and infrastructure-side camera frames, and thus inherently match the corresponding visual frames. Human verification further ensures correctness and coverage. These inputs are fed into a large VLM backbone containing a visual encoder for  $I_v$  and  $I_i$  and a text encoder for E. The outputs of these encoders are merged into a shared

![](_page_4_Figure_2.jpeg)

Fig. 2. Overview of the Proposed V2X-VLM Framework. Camera images from the vehicle and infrastructure sides merged with semantic text prompt are fed in a VLM backbone for multiperspective and multimodal data fusion. Through comprehensive scene understanding and reasoning, V2X-VLM delivers accurate and reliable E2E trajectory planning. Contrastive learning-based feature alignment is applied during fine tuning to ensure the effective fusion of visual and semantic features for enhanced scene understanding. Knowledge distillation stabilizes the learning process to fulfill the complex E2E autonomous driving task.

latent space. This process allows for a more synthetic analysis of the environment, where visual cues and textual information are correlated to provide a holistic understanding of the situation. The primary output of V2X-VLM is a planned trajectory  $\tau$  for the ego vehicle.

To further enhance the correct correlation between image inputs and text descriptions, a contrastive learning-based feature alignment technique is employed during fine tuning. Furthermore, a knowledge distillation strategy is leveraged to ensure efficient and stabilized and training process with knowledge transferring. Both methods are detailed in the following sections. The trajectory refinement procedure is also applied to avoid the planning result being skewed by misleading or atypical data points.

#### 4.2. Multimodal feature alignment

Contrastive learning-based technique is conducted to align visual features from  $(I_v, I_i)$  with text features from E. This alignment ensures that the model accurately correlates each visual scene with its corresponding semantic description, thereby strengthening robust scene understanding (Zeng et al. (2023), Liu et al. (2023)).

Feature Extraction. Given a vehicle image  $I_v$  and an infrastructure image  $I_i$ , we first concatenate them along the width for an image tensor  $[I_v, I_i] \in \mathbb{R}^{H \times (W_v + W_i) \times 3}$ . This composite image is then processed by the image encoder within VLM, and its output is aggregated via a pooling function to produce a fixed length visual embedding:

$$z = \text{pooling}\left(\text{image\_encoder}\left([I_v, I_i]\right)\right) \in \mathbb{R}^{d_z}.$$
 (3)

Simultaneously, the text encoder processes the textual input E, and a pooling operation yields the textual embedding:

$$h = \text{pooling}\left(\text{text\_encoder}(E)\right) \in \mathbb{R}^{d_h}.$$
 (4)

For simplicity, we align the dimensions so that  $d_z = d_h = d'$ , ensuring both embeddings are compatible for subsequent multimodal alignment.

Contrastive Alignment. We apply  $\ell_2$  normalization to both z and h:

$$\hat{z} = \frac{z}{\|z\|_2}, \quad \hat{h} = \frac{h}{\|h\|_2}.$$
 (5)

Given a training batch of size K, we compute pairwise similarities  $S_{ij}$  for  $i, j \in \{1, ..., K\}$  as:

$$S_{ij} = \frac{\hat{z}_i^{\mathsf{T}} \hat{h}_j}{r},\tag{6}$$

where  $\kappa$  is a temperature hyperparameter controlling the sharpness of the similarity distribution. Positive and correct image-text pairs (i=i) are encouraged to have larger similarity scores  $S_{ii}$ , while negative and incorrect pairs  $(i\neq j)$  are penalized, as shown in Fig. 2. By doing so, each visual embedding  $\hat{z}_i$  is brought close to its matching text embedding  $\hat{h}_i$ , and pushed away from all unrelated text embeddings. This approach improves the understanding of the heterogeneous scene of the V2X-VLM framework by ensuring that the combined multiperspective image aligns correctly with its corresponding prompt. Matching the image with the correct prompt adds an additional layer of validation, which further refines the model's understanding of traffic scenes beyond the processing capabilities of the VLM alone.

#### 4.3. Knowledge distillation

Training such a large VLM with diverse cooperative data from multiple cameras and textual prompts for outstanding performance can be challenging. To efficiently transfer multimodal knowledge while stabilizing the training dynamics, we employ a teacher-student distillation strategy (Hinton et al. (2015), Wang et al. (2022), Zhang et al. (2024)) with temperature scaling. Since planning datasets typically contain only a single logged trajectory per frame, while the underlying decision space is inherently multimodal, direct supervision on one hard target may cause mode collapse and over-confident predictions. We therefore distill from a teacher's temperature-softened trajectory distribution, which encodes the relative likelihoods of multiple feasible futures and yields smoother, better-calibrated gradients for a compact student. As shown in Fig. 2, we maintain a frozen pretrained teacher model  $F_T$  and a trainable student model  $F_S$  initialized with pretrained weights. Both models process identical input batches ( $I_v$ ,  $I_i$ , E), producing trajectory logits  $\tau_T = F_T(I_v, I_i, E)$  and  $\tau_S = F_S(I_v, I_i, E)$ , respectively.

Softened Distribution Matching. We calculate the KL divergence between the student's predictions and the teacher's temperature-scaled distribution. First, we soften both logits with a temperature parameter  $\mathcal{T}$ :

$$\tau_T' = \frac{\tau_T}{\mathcal{T}}, \quad \tau_S' = \frac{\tau_S}{\mathcal{T}}. \tag{7}$$

The teacher's target probabilities are then obtained via softmax normalization:

$$p_T = \operatorname{softmax}(\tau_T'). \tag{8}$$

Similarly, the student's target probabilities are obtained as following:

$$p_S = \operatorname{softmax}(\tau_c').$$
 (9)

Distillation Loss Formulation. The final KL divergence loss encourages distributional alignment between student and teacher:

$$\mathcal{L}_{KD} = \mathcal{T}^2 \cdot KL(p_S \parallel p_T). \tag{10}$$

The  $\mathcal{T}^2$  multiplier compensates for gradient scaling induced by temperature, ensuring stable optimization. This softened target distribution provides richer supervision than hard labels, particularly during early training, when the student's random initialization leads to unstable gradients.

## 4.4. Training objective

The complete training objective of V2X-VLM combines three key components:

*Trajectory Prediction Loss.* The primary trajectory prediction loss in the context of vision-language prediction is represented as the loss for next-token prediction:

$$\mathcal{L}_{\text{traj}} = -\sum_{i=1}^{N} \sum_{i=1}^{C} y_{i,n} \log(\hat{y}_{i,n}), \tag{11}$$

where N is the total number of tokens in the generated sequence, C is the number of possible classes in the model's vocabulary,  $y_{i,n}$  is a binary indicator indicating whether the i-th token is the correct one at the n-th position in the true sequence,  $\hat{y}_{i,n}$  represents the predicted probability of the i-th token at the n-th position in the predicted sequence.

Contrastive Alignment Loss. The multimodal feature alignment is controlled by the image-text contrastive loss computed over similarity scores:

$$\mathcal{L}_{\text{align}} = -\frac{1}{K} \sum_{i=1}^{K} \log \frac{\exp(S_{ii})}{\sum_{i=1}^{K} \exp(S_{ii})}.$$
(12)

Knowledge Distillation Loss. The KL divergence loss measures the discrepancy between the student's predictions and the teacher's softened distribution. Expanding the KL divergence term, we have:

$$\mathcal{L}_{\text{KD}} = \mathcal{T}^2 \cdot \sum_{t=1}^{T} p_T^{(t)} \Big( \log p_T^{(t)} - \log p_S^{(t)} \Big), \tag{13}$$

where  $p_T^{(t)}$  and  $p_S^{(t)}$  denote the probabilities of the teacher and the student at the point of the trajectory t, respectively.

Table 1
Latency breakdown of textual scene description generation using Florence-2-large, averaged over 100 images..

| Process                                                   | Description                                                                                                          | Latency (ms)                                                                 | Proportion (%)                |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|-------------------------------|
| Preprocessing<br>Generation<br>Decoding<br>Postprocessing | Image tokenization and formatting<br>Forward pass through Florence-2<br>Convert outputs to text<br>Tokenizer cleanup | $28.01 \pm 1.37$<br>$431.39 \pm 83.44$<br>$0.10 \pm 0.30$<br>$0.01 \pm 0.10$ | 6.10<br>93.88<br>0.02<br>0.00 |
| Total                                                     | -                                                                                                                    | $459.50 \pm 83.53$                                                           | 100.00                        |
| FPS                                                       | -                                                                                                                    | 2.18                                                                         | -                             |

Aggregated Objective. The final training loss combines these components with the weighting factors  $\lambda_1$ ,  $\lambda_2$ :

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{traj}} + \lambda_1 \mathcal{L}_{\text{align}} + \lambda_2 \mathcal{L}_{\text{KD}}. \tag{14}$$

Although the training loss itself does not impose explicit safety or multi-agent constraints, our evaluation protocol includes collision rate and robustness analysis, ensuring that the learned policies reflect deployability considerations. The full minibatch update routine and the corresponding onboard inference workflow are provided in Appendix B as Algorithm 2 and Algorithm 1, respectively. The gradients of the alignment and KD losses, together with the full layer-wise complexity analysis, are provided in Appendix C.1-C.4.

## 5. Experiments

#### 5.1. Dataset

The proposed V2X-VLM framework is evaluated on the DAIR-V2X dataset (Yu et al. (2022)), an extensive and well-annotated resource designed for research on cooperative autonomous driving V2X. It includes 22,325 frames of data from vehicle-mounted sensors and 10,084 frames from infrastructure sensors, capturing RGB images and LiDAR data at up to 25 Hz. This comprehensive dataset is crucial for tasks such as trajectory prediction and multi-sensor data fusion, which facilitates the development of V2X systems that improve traffic safety, navigation accuracy, and cooperative driving strategies.

#### 5.2. Implementation details

We implement the proposed V2X-VLM framework using PyTorch and train it on a single NVIDIA RTX 4090 GPU. We use Florence-2 (Xiao et al. (2024)) as the VLM backbone. Florence-2 is one of state-of-the-art VLMs that delivers high-quality multimodal representations and fine-grained visual understanding. Specifically, the Florence-2-large model trained serves as the teacher model, while the Florence-2-base model serves as the student. We also adopt the pre-trained Florence-2-large model as the backbone for generating scene descriptions in the textual prompt. Specifically, the descriptions are obtained using the built-in <DETAILED\_CAPTION> API option, rather than being manually designed. This choice reflects a practical deployment consideration: in real-world autonomous driving systems, handcrafted prompts would introduce unnecessary latency and computation overhead. Using a standardized API call ensures low-latency and semantically rich textual descriptions that can be consistently integrated into real-time cooperative planning. Table 1 reports the latency breakdown of textual scene description generation with Florence-2-large, averaged over 100 randomly sampled images. During fine-tuning, the vision encoder parameters in the student model are kept frozen to ensure efficient learning. Training converges in 10 epochs using the AdamW optimizer with a batch size of 4, a learning rate of  $1 \times 10^{-6}$  and a linear learning rate scheduler. The loss of contrastive alignment and the loss of knowledge distillation are weighted by hyperparameter  $\lambda_1 = 0.1$  and  $\lambda_2 = 0.5$ , respectively. The distillation employs KL divergence with a temperature scaling factor of  $\mathcal{T} = 2.0$ . The derivation that justifies this constant  $\mathcal{T}$  choice is shown in Appendix C.3-C.4. The planning results of V2X-VLM are assessed using the metrics of L2 error, collision rate, and the transmission cost.

#### 5.3. Results evaluation

## 5.3.1. L2 Error and collision rate

Table 2 compares the performance of the cooperative autonomous driving methods of baseline in terms of L2 error, collision rate defined by oriented bounding-box overlap between the predicted ego trajectory and surrounding obstacles, and transmission cost. UniV2X (Yu et al. (2024)) develops a state-of-the-art E2E pipeline that fuses data from both vehicles and infrastructure to improve perception, online mapping and planning. However, while UniV2X pioneers an E2E vehicle-infrastructure cooperative autonomous driving (VICAD) framework, the L2 errors of trajectory planning and collision rates remain higher than those of our proposed approach. CooperNaut (Hu et al. (2023a)) uses a decentralized fusion strategy for V2V communication to improve perception, but it still fails in trajectory planning and safety compared to our method. In contrast, V2X-VLM achieves the lowest L2 error across all time horizons and maintains an average low collision rate of 0.03 %. These improvements can be attributed to the advanced multimodal fusion of vehicle and infrastructure imagery with textual scene descriptions, complemented by contrastive learning and knowledge distillation to further refine feature alignment.

**Table 2**Comparison of L2 error, collision rate, and transmission cost across different methods. Lower L2 error and collision rate indicate better planning accuracy and safety, while transmission cost reflects the required bandwidth in BPS.

| Method                         | L2 Err | L2 Error (m) ↓ |      |      |      | on Rate ( | %)↓  | Transmission Cost (BPS) ↓ |                      |
|--------------------------------|--------|----------------|------|------|------|-----------|------|---------------------------|----------------------|
|                                | 2.5s   | 3.5s           | 4.5s | Avg  | 2.5s | 3.5s      | 4.5s | Avg                       | , , ,                |
| UniV2X - No Fusion             | 2.58   | 3.37           | 4.36 | 3.44 | 0.15 | 1.04      | 1.48 | 1.08                      | 0                    |
| UniV2X - Vanilla               | 2.33   | 3.69           | 5.12 | 3.71 | 0.59 | 2.07      | 3.70 | 2.07                      | $8.19 \times 10^{7}$ |
| UniV2X - BEV Feature Fusion    | 2.31   | 3.29           | 4.31 | 3.30 | 0.00 | 1.04      | 1.48 | 0.93                      | $8.19 \times 10^{7}$ |
| UniV2X (Yu et al. (2024))      | 2.59   | 3.35           | 4.49 | 3.48 | 0.00 | 0.44      | 0.59 | 0.34                      | $8.09 \times 10^{5}$ |
| CooperNaut (Hu et al. (2023a)) | 3.84   | 5.33           | 6.87 | 5.35 | 0.44 | 1.33      | 1.93 | 0.54                      | $8.19 \times 10^{5}$ |
| V2X-VLM (Ours)                 | 1.09   | 1.12           | 1.42 | 1.21 | 0.02 | 0.03      | 0.03 | 0.03                      | $1.24 \times 10^7$   |

**Table 3**Performance comparison for different downsampling scaling factors in V2X communication.

| Canling Engton | Resolution after | Information Quality                                                                                  | Transmission           | L2 Err | or (m) ↓ |      | Total | FPS          |       |
|----------------|------------------|------------------------------------------------------------------------------------------------------|------------------------|--------|----------|------|-------|--------------|-------|
|                | DownSampling     |                                                                                                      | Cost (BPS) ↓           | 2.5s   | 3.5s     | 4.5s | Avg.  | Latency (ms) |       |
| UniV2X         | -                | -                                                                                                    | $8.09 \times 10^{5}$   | 2.59   | 3.35     | 4.49 | 3.48  | -            | -     |
| 1 (No Change)  | 1080 × 1920      | Full resolution; all details preserved; highest fidelity.                                            | $1.24 \times 10^{7}$   | 1.09   | 1.12     | 1.42 | 1.21  | 353.36       | 11.32 |
| 0.5            | 540 × 960        | Moderate downsampling; accept-<br>able detail with minor loss in fine<br>features; moderate fidelity | 3.11 × 10 <sup>6</sup> | 1.23   | 1.27     | 1.45 | 1.32  | 352.90       | 11.33 |
| 0.2            | 316 × 384        | High downsampling; significant reduction in detail; degraded fidelity                                | 4.98 × 10 <sup>5</sup> | 1.34   | 1.38     | 1.59 | 1.44  | 264.79       | 15.11 |
| 0.1            | 108 × 192        | High downsampling; severe loss of details; low fidelity                                              | 1.24×10 <sup>5</sup>   | 1.42   | 1.47     | 1.71 | 1.53  | 263.97       | 15.15 |

#### 5.3.2. Transmission cost

In V2X-VLM, cooperative perception is achieved by integrating both vehicle-side and infrastructure-side images into the VLM backbone. Since infrastructure-side images are not locally available on the ego vehicle, they must be transmitted over a communication network before being processed. This introduces a trade-off between transmission cost and planning accuracy. The method of calculating transmission cost is provided in Appendix A. When transmitting full-resolution images at  $1080 \times 1920$ , the required bandwidth reaches  $1.24 \times 10^7$  BPS, significantly higher than the  $8.09 \times 10^5$  BPS used in UniV2X, which relies on feature-level transmission rather than raw image sharing. This substantial transmission cost arises from the direct transmission of high-resolution images, as the bandwidth requirement scales with pixel density.

To mitigate the high communication overhead, a practical approach is to downsample the infrastructure-side images before transmission and upsample them upon reception. This reduces the amount of data sent over the network while still allowing the VLM to process the visual information. As shown in Table 3, a scaling factor of 0.5 decreases the transmission cost to  $3.11 \times 10^6$  BPS, while extreme downsampling with 0.1 scaling factor reduces it to just  $1.24 \times 10^5$  BPS, achieving a two-order-of-magnitude reduction. However, a lower resolution leads to inevitable degradation in visual detail, which impacts performance. The L2 error increases from 1.21 m at full resolution to 1.53 m at the lowest resolution, highlighting the trade-off between bandwidth efficiency and accuracy.

Beyond transmission cost and planning accuracy, there are additional trade-offs to consider. As image resolution decreases, computational efficiency improves due to lower input complexity. This effect is evident in the metrics of FPS and per-batch inference latency: the highest-resolution setup runs at 11.32 FPS with a latency of 353.36 ms, whereas the lowest-resolution configuration achieves 15.15 FPS with a reduced latency of 263.97 ms. These results suggest that aggressive compression benefits real-time inference while introducing a slight accuracy penalty. Notably, despite using an extremely low-resolution infrastructure image, V2X-VLM still consistently outperforms UniV2X in trajectory planning across all time horizons. This demonstrates that the proposed multimodal feature alignment and VLM-based reasoning can effectively compensate for degraded visual information, ensuring robust trajectory prediction even under constrained bandwidth conditions.

V2X-VLM currently adopts raw infrastructure image transmission to maximize accuracy and preserve pixel-level alignment between visual and textual modalities. While Table 3 demonstrates that downsampling effectively reduces communication cost with limited accuracy degradation, the framework still performs full feature extraction on the received images. A promising direction to further improve efficiency is to transmit pre-encoded visual features instead of raw images, which enables compact representations to be shared without disrupting multimodal alignment.

**Table 4** Robustness evaluation of V2X-VLM under perturbations.

| Condition                               |      | L2 Error (m) ↓ |      | Collision Rate (%) ↓ |      |      |  |
|-----------------------------------------|------|----------------|------|----------------------|------|------|--|
|                                         | 2.5s | 3.5s           | 4.5s | 2.5s                 | 3.5s | 4.5s |  |
| Image Noise (std = 5)                   | 1.31 | 1.34           | 1.50 | 0.03                 | 0.03 | 0.04 |  |
| Image Noise (std = 10)                  | 1.17 | 1.21           | 1.49 | 0.03                 | 0.03 | 0.04 |  |
| Text Perturbation (p = 0.1)             | 1.33 | 1.37           | 1.46 | 0.02                 | 0.03 | 0.04 |  |
| Combined (Image Noise 10, Text p = 0.1) | 1.34 | 1.36           | 1.76 | 0.02                 | 0.03 | 0.03 |  |
| No Perturbation                         | 1.09 | 1.12           | 1.42 | 0.02                 | 0.03 | 0.03 |  |

**Table 5** Latency breakdown and FPS analysis for inference efficiency. Total latency represents the time taken for a batch of inputs. FPS indicates the number of frames processed per second.

| Process                                                  | Description                                                                                                                                 | Latency (ms)            | Proportion (%)      |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|---------------------|
| Preprocessing<br>Inference<br>Postprocessing<br>Residual | Tokenization and image processing<br>Forward pass through the model<br>Decoding the model outputs<br>Minor operations such as data loading, | 269.01<br>72.72<br>9.02 | 76.1<br>20.6<br>2.6 |
| Overhead                                                 | synchronization, and loop overhead                                                                                                          | 2.61                    | 0.7                 |
| Total                                                    | -                                                                                                                                           | 353.36                  | 100.0               |
| FPS                                                      | -                                                                                                                                           | 11.32                   | -                   |

#### *5.4. Robustness and efficiency analysis*

To assess the robustness of V2X-VLM, we introduce controlled perturbations to both the visual and textual inputs and evaluate their effects on trajectory accuracy and safety. Table 4 shows that the model maintains strong performance even under perturbed conditions. Adding Gaussian noise to infrastructure-side images (*Image Noise*) slightly increases the L2 error but does not significantly degrade planning accuracy, as seen in the cases of standard deviation 5 and 10. Text perturbation (*Text Perturbation*) simulates potential errors in language descriptions by randomly modifying portions of textual inputs. The impact on L2 error remains minor, highlighting the robustness of the model in handling imperfect textual descriptions. When both image and text perturbations are applied simultaneously (*Combined Perturbation*), the average L2 error increases to 1.49, still outperforming existing baselines shown in Table 2. The collision rate remains nearly constant across all perturbation settings, further demonstrating V2X-VLM's stability and resilience to noisy inputs. Furthermore, a complementary robustness evaluation under communication degradation is provided in Appendix E, which further assesses V2X-VLM's performance when infrastructure-side data become partially unavailable.

Beyond robustness, we analyze the inference efficiency of V2X-VLM by breaking down the total latency per batch in Table 5. The overall latency for a batch of samples is 353.36 ms, corresponding to a real-time processing rate of 11.32 FPS. The majority of the latency stems from preprocessing (76.1 %), which includes tokenization and image feature extraction. This step is necessary for the multimodal input fusion but could be further optimized. The model's forward pass (*Inference*) accounts for 20.6 % of the total latency, reflecting the computational cost of large-scale vision language processing. *Postprocessing*, which involves decoding the model output into trajectories, is relatively lightweight (2.6 %), and the residual overhead from data loading and synchronization is negligible (0.7 %). Despite computational complexity, V2X-VLM achieves real-time inference capabilities, demonstrating its feasibility for deployment in practical cooperative autonomous driving systems.

#### *5.5. Ablation study*

We conduct an ablation study to evaluate the contributions of key components in V2X-VLM. Results are presented in Table 6. Each ablation setting is described as follows:

- **No Fusion:** Only ego-vehicle images are used in training and testing, omitting infrastructure-side input.
- **w/o Knowledge Distillation:** The student model is trained without knowledge distillation.
- **w/o Scene Prompting:** The semantic textual scene descriptions are removed from the input.
- **w/o Feature Alignment:** The contrastive learning-based feature alignment between image and text is disabled.

As shown in Table 6, removing infrastructure input results in the highest L2 error, demonstrating the necessity of multiperspective fusion. This finding reinforces the fundamental advantage of cooperative autonomous driving over a single-vehicle-based solution. The significant performance drop in the single vehicle setting underscores the limitations of independent perception and planning, further supporting the case for cooperative autonomous driving. Among the model components, feature alignment provides the largest accuracy gain. Scene prompting and knowledge distillation further enhance multimodal understanding and trajectory planning. The results validate the effectiveness of each component and demonstrate their combined impact in achieving state-of-the-art performance.

**Table 6** Ablation study result. Removing each component of V2X-VLM degrades planning accuracy, demonstrating their importance.

| Method                                  | L2 Error (m) ↓ |              |              |              |              | Collision Rate (%) ↓ |              | Transmission Cost (BPS) ↓ |                          |
|-----------------------------------------|----------------|--------------|--------------|--------------|--------------|----------------------|--------------|---------------------------|--------------------------|
|                                         | 2.5s           | 3.5s         | 4.5s         | Avg.         | 2.5s         | 3.5s                 | 4.5s         | Avg.                      |                          |
| No Fusion                               | 1.45           | 1.50         | 1.53         | 1.49         | 0.03         | 0.03                 | 0.04         | 0.03                      | 0                        |
| w/o Distillation                        | 1.33           | 1.33         | 1.59         | 1.42         | 0.03         | 0.03                 | 0.03         | 0.03                      | 1.24 × 107               |
| w/o Scene Prompting                     | 1.34           | 1.37         | 1.58         | 1.43         | 0.03         | 0.03                 | 0.03         | 0.03                      | 1.24 × 107               |
| w/o Feature Alignment<br>V2X-VLM (Ours) | 1.40<br>1.09   | 1.44<br>1.12 | 1.69<br>1.42 | 1.51<br>1.21 | 0.03<br>0.02 | 0.03<br>0.03         | 0.04<br>0.03 | 0.03<br>0.03              | 1.24 × 107<br>1.24 × 107 |

![](_page_9_Figure_4.jpeg)

**Fig. 3.** Visualization of V2X-VLM trajectory planning on three common driving scenarios. Continuous frames are visualized at a frequency of 1 Hz.

# *5.6. Visualization*

To further illustrate the effectiveness of V2X-VLM, we provide qualitative results showcasing the performance of V2X-VLM in both normal scenarios and some corner cases.

Fig. 3 showcases the trajectories planned by V2X-VLM in three common driving maneuvers: left turn, going straight, and right turn. It illustrates the consistent ability of V2X-VLM to produce high-quality trajectory output. Further visualization and discussion regarding more challenging corner cases, such as rainy conditions with vehicle camera lens blur, as well as driving through complex intersections, are presented in Appendix D .

To further examine how cooperative perception enhances planning safety under partial occlusion, we conduct two representative case studies focusing on vehicle-infrastructure complementary visibility. In both cases, the ego vehicle's front camera suffers from occlusions that conceal nearby vehicles, while the roadside infrastructure camera maintains a clear view of the intersection and reveals

![](_page_10_Figure_2.jpeg)

![](_page_10_Figure_3.jpeg)

![](_page_10_Figure_5.jpeg)

![](_page_10_Figure_6.jpeg)

**Fig. 4.** Case study of occlusion handling through vehicle-infrastructure cooperation.

the otherwise hidden traffic participants. This complementary perception enables V2X-VLM to reason jointly across viewpoints and generate safe, collision-free trajectories even when critical agents are invisible from the ego perspective.

Fig. 4 presents two examples. In the first case (Fig. 4(a)), at a signalized intersection, the ego vehicle performs a left turn while oncoming vehicles proceed straight. In the ego's front-camera view (middle column), an oncoming vehicle is almost completely concealed behind its leading vehicle, which makes it difficult for the ego to perceive the second vehicle or anticipate the potential conflict. In contrast, the infrastructure view (right column) clearly reveals the hidden vehicle, as marked in the red box, which enables V2X-VLM to perceive or anticipate the potential conflict. The resulting trajectory planning and ground truth are illustrated in the front-camera view (middle cloumn), which shows that V2X-VLM generates a conservative turning path that closely aligns with the ground truth, reflecting the realistic and safe maneuver taken to avoid the occluded vehicle. The leftmost plot visualizes the corresponding speed profile, where V2X-VLM maintains relatively low speeds during the early phase of the maneuver and gradually accelerates through the intersection, reflecting a cautious yet smooth planning behavior that ensures safety under occlusion.

In the second case (Fig. 4(b)), the ego vehicle proceeds straight through an intersection while two vehicles are making left turns from the opposite direction. In the ego's front-camera view (middle column), only the first oncoming vehicle is visible, whereas the second one is almost entirely hidden behind it, which makes it difficult for the ego vehicle to anticipate the additional crossing threat. However, in the infrastructure view (right column), the occluded vehicle is clearly visible, as marked in the red box, which allows V2X-VLM to accurately perceive the complete traffic dynamics. As shown in the front-camera view (middle column), the planned trajectory remains consistent with the ground truth, indicating that V2X-VLM successfully maintains a safe and steady straight path without abrupt maneuvers or collisions. The leftmost plot illustrates the corresponding speed profile, where V2X-VLM maintains a steady speed throughout the intersection, indicating confident and smooth motion planning once both left-turning vehicles are correctly perceived through the cooperative infrastructure view.

Beyond trajectory outputs, we further visualize the cross-modal attention behavior of V2X-VLM to provide interpretable insights into how the model allocates attention between vehicle- and infrastructure-side inputs under different driving conditions to get reliable collaborative trajectory planning. To achieve this, we adopt a Gradient-times-Input saliency attribution technique. For each vehicle-infrastructure image pair, the gradient of the model's output loss with respect to the input pixels is multiplied by the input itself to estimate the contribution of each pixel to the final decision. The resulting saliency maps highlight visually influential regions across both modalities. These maps are then separated into the vehicle side and the infrastructure side, averaged along the width dimension, and visualized as attention energy curves with shaded regions representing the relative activation strength. This process enables a clear examination of how V2X-VLM dynamically balances perception between near-field onboard sensing and far-field cooperative inputs.

Fig. 5 illustrates five representative examples that reveal interpretable and context-consistent attention patterns. In Fig. 5(a), on the vehicle side, the saliency concentrates sharply along the far-left region, corresponding to the ego vehicle's intended turning path. This area contains the adjacent lane and near-field curb geometry that guide the left-turn maneuver. On the infrastructure side, attention expands broadly across about the right two-thirds of the image, the region directly ahead of the ego vehicle where multiple oncoming cars are present on the opposite approach. The prominent peak within this region aligns with those oncoming vehicles, indicating that the model leverages infrastructure perception to anticipate cross-traffic interactions and evaluate safe gaps before executing the turn. In Fig. 5(b), vehicle-side attention concentrates slightly right of the ego's forward axis, where a few vehicles are present and

![](_page_11_Picture_2.jpeg)

**Fig. 5.** Cross-modal saliency visualization of V2X-VLM under five representative scenarios. Each pair overlays vehicle-side (orange) and infrastructure-side (blue) attention distributions.

immediate yield decisions are formed. Infrastructure-side saliency shifts to the image's left, aligning with the ego vehicle's position and intended left-turn path, to assess cross-traffic and left-turn gaps from a global vantage. In Fig. 5(c), with an open roadway ahead, vehicle-side attention is relatively uniform, indicating no single near-field hazard dominates. Infrastructure-side attention biases left, again aligned with the ego's left-turn trajectory, while also accounting for opposing vehicles that define the primary conflict zone. In Fig. 5(d), under rain-induced blur, the vehicle-side attention becomes fragmented and weak, as the ego camera fails to capture clear contours of nearby vehicles. Conversely, the infrastructure-side saliency forms a distinct peak directly in front of the ego vehicle, precisely at the location where a few vehicles are approaching from the left-turn conflict direction. This focused pattern shows that when the onboard view is degraded, the model actively shifts reliance to the infrastructure view to track approaching vehicles and maintain awareness of potential collision threats in the ego's turning path. In Fig. 5(e), despite vehicle-side blur, the model infers a likely right turn and correspondingly biases saliency to the right-hand sector to resolve near-field lane geometry and merging gaps. The infrastructure view places the ego on the right and similarly increases attention on the right-side conflict area, corroborating the right-turn hypothesis from a global perspective. Collectively, these results demonstrate that V2X-VLM achieves interpretable vehicleinfrastructure collaboration: the model prioritizes vehicle-side cues for near-field maneuvering and curvature following, while relying on infrastructure-side cues for far-field traffic reasoning, cross-lane interaction, and robustness under degraded visual conditions.

#### **6. Conclusion**

This paper introduces V2X-VLM, an end-to-end cooperative autonomous driving framework that integrates multimodal scene understanding using VLMs for enhanced trajectory planning. V2X-VLM fuses multiperspective vehicle and infrastructure images with semantic text to enable cooperative understanding of driving scenes. By further applying contrastive feature alignment and knowledge distillation, the model achieves state-of-the-art planning accuracy over baseline methods. The ablation study validates the necessity of each component. Robustness evaluations further validate the model's resilience against input perturbations, while efficiency analysis highlights its feasibility for real-time deployment.

Future work will focus on two key areas of improvement. First, we aim to enhance the model's generalization by addressing more long-tail scenarios. This will involve generating various long-tail scenarios for model training and evaluation. Second, efforts will be made to reduce transmission costs by exploring a vehicle-road-cloud distributed training and deployment paradigm dedicated to optimizing the balance between data processing and communication for scalable and cost-effective driving applications.

#### **CRediT authorship contribution statement**

**Junwei You:** Writing – original draft, Visualization, Validation, Software, Resources, Methodology, Investigation, Formal analysis, Data curation; **Zhuoyu Jiang:** Visualization, Validation, Software, Methodology, Investigation, Data curation; **Zilin Huang:** Visualization, Software, Resources, Methodology, Investigation, Formal analysis; **Haotian Shi:** Writing – review & editing, Validation, Supervision, Resources, Project administration, Methodology, Investigation, Conceptualization; **Rui Gan:** Visualization, Validation, Software, Investigation; **Keshu Wu:** Validation, Software, Methodology, Formal analysis; **Xi Cheng:** Validation, Methodology, Investigation, Formal analysis; **Xiaopeng Li:** Validation, Supervision, Conceptualization; **Bin Ran:** Validation, Supervision, Project administration, Conceptualization.

## **Data availability**

Data will be made available on request.

#### **Appendix A. Communication Cost Calculation**

In V2X cooperative perception, infrastructure-side images must be transmitted over a network before processing, which introduces a significant communication overhead. The transmission cost, measured in bytes per second (BPS), depends on image resolution, color channels, transmission frequency, and potential downsampling.

For an image of width , height , and color channels, transmitted at frequency , the required bandwidth is:

$$BPS = s^2 W H C f, \tag{A.1}$$

where is the downsampling factor (0 *<*  ≤ 1), reducing the image resolution before transmission.

Following this, for a full-resolution infrastructure-side image of 1080 × 1920 pixels with three color channels, transmitted at 2 Hz without downsampling ( = 1), the required bandwidth is calculated as 1*.*24 × 10<sup>7</sup> BPS.

## **Appendix B. Core Training and Inference Implementation**

This appendix consolidates the two key procedural components of our V2X-VLM framework: the *online cooperative inference pipeline*, which runs on the ego vehicle in real time, and the *student-teacher training routine*, used offline to optimise the lightweight student model. The pseudocode is intentionally concise to highlight where the critical processes, such as V2X communication, multimodal reasoning, and trajectory planning, occur during inference, and how alignment and knowledge-distillation (KD) losses are combined during training.

## Algorithm 1 Real-Time Cooperative Inference Flow.

**Require:** Ego image  $I_v$ , roadside image  $I_i$ , pre-trained foundation model  $\Phi$ 

**Ensure:** Smoothed future trajectory  $\tau$ 

1: Spawn Task-A (vehicle): Generate scene text  $E = \Phi_{\text{describe}}(I_v)$ 

- ▶ e.g. GPT-4o, Florence-2-large
- 2: **Spawn Task-B (roadside)**: Compress **(Optional)**, and send  $I_i$  as payload  $P_i$
- 3: Synchronize tasks (vehicle): Receive roadside image  $I_i$  (decode if compressed)
  - d)  $\triangleright$  ensure E is ready
- 4: Run V2X-VLM: Compute multimodal feature  $f = V2XVLM(I_{mv} = [I_v, I_i], E)$
- 5: **Decode trajectory tokens**: Obtain  $\hat{\tau} = \text{Decoder}(f)$
- 6: Apply refinement: Produce τ = Refine(τ̂)
  7: return τ

## Algorithm 2 Offline Training with Alignment & Knowledge Distillation.

**Require:** Pre-trained teacher foundation model  $F_T$ , student V2X-VLM model, training batch B

**Ensure:** Updated student parameters  $\theta_S$ 

- 1: for all  $(I_v, I_i, \tau^*) \in \mathcal{B}$  do
- 2: Generate preliminary scene description text:  $E_0 = F_T(I_n)$

▷ e.g. Florence-2-large▷ crowd-worker OA

- 3: Refine text manually: Obtain final E
- 4: **Forward teacher model**: Predict  $\tau_T = F_T(I_v, I_i, E)$
- 5: **Forward student model:** Predict  $\tau_S$ , z,  $h = V2X-VLM(I_v, I_i, E)$
- 6: Calculate alignment loss: Compute  $\mathcal{L}_{align}$  via InfoNCE

⊳ Eq. 10

7: Calculate KD loss: Compute L<sub>KD</sub> with temperature T
 8: Calculate trajectory loss: Compute L<sub>traj</sub> for next-token prediction

⊳ Eq. 8 ⊳ Eq. 9

- 9: end for
- 10: **Aggregate losses:** Obtain  $\mathcal{L}_{total} = \mathcal{L}_{traj} + \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{KD}$

⊳ Eq. 12

11: **Update parameters**: Apply AdamW to  $\theta_S$ 

Algorithms 1 and 2 are referenced in Sections 4.1-4.4 of the main text for clarity. They require no additional hyperparameters beyond those described in Section 5.2.

## Appendix C. Theoretical Analysis

This appendix expands the concise derivations given in the main paper. The first part details the floating-point operations (FLOPs) for a multimodal transformer layer. The second part derives the gradients of the image-text alignment loss and the knowledge-distillation (KD) loss. The third part analyses how the temperature  $\mathcal{T}$  influences optimization.

#### C.1. Exact FLOP count per layer

The FLOPs for a single multimodal transformer layer are derived from the standard  $QK^TV$  formulation. Let the hidden size be d, the number of heads be h such that the head dimension satisfies  $d_h = d/h$ , the sequence length for the vision branch be  $N_v$ , and the sequence length for the text branch be  $N_t$ .

The vision branch self-attention requires

$$FLOP_{vis} = 2N_n d^2 + N_n^2 d,$$
 (C.1)

where the first term is for the linear projections and the second term covers  $QK^{T}$  and the subsequent  $softmax(QK^{T})V$ .

Likewise, the text branch self-attention requires

$$FLOPs_{text} = 2N_{r}d^{2} + N_{r}^{2}d$$
. (C.2)

The cross-modal attention, taking the vision tokens as query and the text tokens as key-value, requires

$$FLOPs_{cross} = 2N_v d^2 + N_v N_t d. \tag{C.3}$$

Since d is fixed after architecture selection, the projection terms  $2N_vd^2$  and  $2N_td^2$  grow only linearly in the token numbers and are therefore asymptotically dominated by the attention terms. Discarding these lower-order components yields

$$\mathcal{O}(N_v d) + \mathcal{O}(N_t d) + \boxed{\mathcal{O}(N_v N_t)},$$
 (C.4)

which shows that cross-modal attention is the only  $\Theta(N^2)$  bottleneck.

Although the main experiments in Section 5 keep the full-rank cross-modal projections, for deployment on resource-constrained hardware, each  $d \times d$  projection can be replaced by a low-rank product  $AB^{T}$  with  $A, B \in \mathbb{R}^{d \times r}$ . This change reduces the two projection terms in FLOPs<sub>cross</sub> from  $2d^2$  to 2dr and leaves the rest of the layer untouched. Detailed derivations appear later in this subsection, and an example with r = 4 yields a four-to-one reduction in these projections. The adapter is therefore a compatible but optional engineering choice rather than part of the core model.

**Table C.7** Sensitivity analysis of the distillation temperature  $\mathcal{T}$ . A moderate temperature ( $\mathcal{T}$  = 2) achieves the best trade-off between planning accuracy and stability..

| Temperature $\mathcal{T}$ | L2 Erro | or (m) ↓ |      |      | Collisio | Collision Rate (%) ↓ |      |      |  |  |
|---------------------------|---------|----------|------|------|----------|----------------------|------|------|--|--|
|                           | 2.5s    | 3.5s     | 4.5s | Avg. | 2.5s     | 3.5s                 | 4.5s | Avg. |  |  |
| 1                         | 1.57    | 1.67     | 1.86 | 1.70 | 0.04     | 0.04                 | 0.05 | 0.04 |  |  |
| 3                         | 1.36    | 1.36     | 1.64 | 1.45 | 0.03     | 0.04                 | 0.05 | 0.04 |  |  |
| 4                         | 1.30    | 1.30     | 1.38 | 1.30 | 0.03     | 0.04                 | 0.05 | 0.04 |  |  |
| 2                         | 1.09    | 1.12     | 1.42 | 1.21 | 0.02     | 0.03                 | 0.03 | 0.03 |  |  |

## C.2. Gradient of the alignment loss

The InfoNCE alignment loss for a batch of size K is defined as:

$$\mathcal{L}_{\text{align}} = -\frac{1}{K} \sum_{i=1}^{K} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{B} \exp(S_{ij})}, \quad S_{ij} = \langle z_i, h_j \rangle / \kappa.$$
 (C.5)

Let  $\sigma_{ij} = \exp(S_{ij}) / \sum_k \exp(S_{ik})$ , and then

$$\frac{\partial \mathcal{L}_{\text{align}}}{\partial S_{ii}} = -\frac{1}{K} (1 - \sigma_{ii}) \tag{C.6}$$

$$\frac{\partial \mathcal{L}_{\text{align}}}{\partial S_{ii}} = \frac{1}{K} \sigma_{ij} \quad j \neq i. \tag{C.7}$$

Eq. (C.6) and (C.7) show that the gradient becomes small once the model assigns high confidence to the true pair, thereby concentrating learning on hard positives.

#### C.3. Gradient of the distillation loss

The temperature-scaled KD loss is

$$\mathcal{L}_{\text{KD}} = \frac{\mathcal{T}^2}{N} \sum_{k} p_T^{(k)} \log \frac{p_T^{(k)}}{p_S^{(k)}}, \quad p_S = \operatorname{softmax}(\tau_S/\mathcal{T}), \ p_T = \operatorname{softmax}(\tau_T/\mathcal{T}). \tag{C.8}$$

Since  $\partial \log p_S^{(k)}/\partial \tau_S^{(\ell)} = (\delta_{k\ell} - p_S^{(\ell)})/\mathcal{T}$ , we have

$$\frac{\partial \mathcal{L}_{\text{KD}}}{\partial \tau_S^{(\ell)}} = \frac{\mathcal{T}}{N} \left( p_S^{(\ell)} - p_T^{(\ell)} \right),$$
 (C.9)

which shows that the gradient scale grows with  $\mathcal{T}$ .

## C.4. Impact of temperature on optimization

Temperature Scheduling. Eq. (C.9) shows that a sharp teacher distribution reduces gradient magnitude at the start of training. A constant temperature of  $\mathcal{T} = 2$  amplifies the signal by a factor of four throughout all epochs, which we found sufficient for stable convergence and did not observe further benefit from annealing.

Combined Effect with Alignment Loss. The InfoNCE gradient concentrates on the hardest image-text pairs, while the higher temperature in knowledge distillation restores the signal that would otherwise vanish early. Together these two mechanisms smooth the loss landscape and shorten convergence time.

Summary of Findings. The boxed Eq. (C.4), (C.6), (C.7) and (C.9) justify the hyperparameter choice  $\mathcal{T} = 2$ , which gives the best balance between computational efficiency and prediction accuracy across all experiments.

To further validate the theoretical analysis above, we conduct an empirical sensitivity study by varying the distillation temperature  $\mathcal{T}$ . As summarized in Table C.7, increasing  $\mathcal{T}$  initially enhances optimization stability and alignment strength, but overly high values yield diminishing returns. The results confirm that  $\mathcal{T}=2$  achieves the most favorable balance between convergence smoothness and performance. The sensitivity analysis in Table C.8 shows that moving either weight away from moderate settings slightly degrades both accuracy and safety, and the choice  $\lambda_1=0.1$  and  $\lambda_2=0.5$  provides the most stable trade-off.

**Table C.8** Sensitivity analysis of loss weights <sup>1</sup> and <sup>2</sup> on planning accuracy and safety.

| Setting                                      |              | L2 Error (m) ↓ |              |              | Collision Rate (%) ↓ |              |              |              |  |
|----------------------------------------------|--------------|----------------|--------------|--------------|----------------------|--------------|--------------|--------------|--|
|                                              | 2.5s         | 3.5s           | 4.5s         | Avg.         | 2.5s                 | 3.5s         | 4.5s         | Avg.         |  |
| 𝜆1 = 0.20, 𝜆2 = 0.50<br>𝜆1 = 0.30, 𝜆2 = 0.50 | 1.67<br>1.67 | 1.72<br>1.70   | 2.11<br>2.09 | 1.83<br>1.82 | 0.03<br>0.04         | 0.04<br>0.04 | 0.05<br>0.06 | 0.04<br>0.05 |  |
| 𝜆1 = 0.10, 𝜆2 = 0.25<br>𝜆1 = 0.10, 𝜆2 = 0.75 | 1.25<br>1.88 | 1.27<br>2.06   | 1.54<br>2.29 | 1.35<br>2.08 | 0.03<br>0.03         | 0.04<br>0.04 | 0.05<br>0.05 | 0.04<br>0.04 |  |
| 𝜆1 = 0.10, 𝜆2 = 0.50 (Original)              | 1.09         | 1.12           | 1.42         | 1.21         | 0.02                 | 0.03         | 0.03         | 0.03         |  |

## **Appendix D. Corner Case Visualization**

To further assess the robustness of V2X-VLM, we provide additional qualitative visualizations focusing on corner cases, including adverse weather conditions where vehicle camera lens is blur and complex intersection scenarios. Again, we examine three most common maneuvers, going straight, left turn, and right turn, under these challenging conditions.

In adverse weather scenarios, blurred vehicle cameras reduce visibility, making it harder to extract reliable visual cues. This tests V2X-VLM's ability to leverage infrastructure-side views and textual descriptions for robust trajectory planning. Meanwhile, complex intersections introduce ambiguous right-of-way situations and occlusions that require effective multiperspective and multimodal fusion to ensure accurate and safe navigation. The qualitative results, presented in Fig. D.6, Fig. D.7, and Fig. D.8, demonstrate that V2X-VLM consistently generates stable and contextually appropriate trajectories despite these challenges, reinforcing its capability to handle real-world uncertainties in cooperative autonomous driving.

![](_page_16_Figure_2.jpeg)

**Fig. D.6.** Visualization of V2X-VLM's trajectory planning for going-straight scenarios in challenging corner cases. Continuous frames are displayed at a frequency of 1 Hz.

![](_page_17_Figure_2.jpeg)

**Fig. D.7.** Visualization of V2X-VLM's trajectory planning for right-trun scenarios in challenging corner cases. Continuous frames are displayed at a frequency of 1 Hz.

![](_page_18_Figure_2.jpeg)

**Fig. D.8.** Visualization of V2X-VLM's trajectory planning for left-turn scenarios in challenging corner cases. Continuous frames are displayed at a frequency of 1 Hz.

#### Vehicle View

![](_page_19_Picture_3.jpeg)

# Infrastructure View

![](_page_19_Picture_5.jpeg)

#### **Brief Scene Description:**

1. Weather: Sunny 2. Time: Daytime

3. Road Environment: Urban

4. Ego Lane Position: Within the Intersection

#### **Brief Scene Description:**

- 1. Ego Vehicle Location: Eastbound
- 2. Ego vehicle movement: Left Turn
- 3. Intersection Condition: Busy

## **Detailed Scene Description:**

The vehicle is positioned within the intersection, preparing to make a left turn. There are a few cars visible in the distance, some also turning or waiting. Traffic lights and signs are present, along with greenery and buildings in the background, indicating an urban environment. The clear and sunny weather provides good visibility conditions.

## **Detailed Scene Description:**

A busy intersection with multiple vehicles, including a small truck and various cars, maneuvering through. Several lanes are visible, with some vehicles moving and others stopped. The intersection has clearly marked pedestrian crosswalks and is surrounded by buildings and greenery, indicating a well-developed urban area. The scene shows active traffic, with vehicles, including the ego vehicle in the eastbound direction, making left turns.

(a)

## Vehicle View

![](_page_19_Picture_20.jpeg)

## Infrastructure View

![](_page_19_Picture_22.jpeg)

## **Brief Scene Description:**

1. Weather: Overcast 2. Time: Daytime

3. Road Environment: Urban

4. Ego Lane Position: Second Lane from the Left

## **Brief Scene Description:**

- 1. Ego Vehicle Location: Southbound
- 2. Ego vehicle movement: Proceeding Straight
- 3. Intersection Condition: Steady

## **Detailed Scene Description:**

The vehicle is positioned in the second lane from the left, facing the intersection, with overcast weather conditions. There are a few cars visible ahead, some turning or waiting. The environment is urban, characterized by traffic lights, signs, and nearby buildings. The road is wide, and the intersection appears to be in a developed area. The overcast weather provides consistent lighting, aiding visibility.

#### **Detailed Scene Description:**

The intersection is steady, with multiple vehicles, including a small truck and various cars, maneuvering through. The ego vehicle, traveling southbound, is proceeding straight through the intersection. Several lanes are present, some with vehicles moving and others stopped. The intersection includes clearly marked pedestrian crosswalks and is surrounded by greenery and buildings, reflecting a well-developed urban setting. The traffic flow is steady, indicating regular activity at the intersection.

(b)

Fig. D.9. Examples of VLM vehicle-side and infrastructure-side scene understanding.

## **Appendix E. Impact of Network Variability on Cooperative Inference**

To evaluate the robustness of V2X-VLM under realistic network conditions, we design an experiment that simulates communication degradation between the infrastructure and vehicle agents. In practical V2X deployments, the transmission of infrastructure-side images may be affected by latency, packet loss, or temporary disconnection due to limited bandwidth or channel contention. When such degradation occurs, the vehicle would have to rely primarily on its own camera view to make planning decisions.

To approximate this scenario, we randomly drop the infrastructure-side images for a proportion of the test samples, while keeping the rest of the pipeline unchanged. The dropped samples therefore represent cases where cooperative perception is partially lost, and only the vehicle-side image is available for inference. We evaluate three levels of degradation: = 0*.*1, = 0*.*3, and = 0*.*5, corresponding respectively to mild, moderate, and severe communication impairment. The remaining 1 − fraction of samples are processed with full cooperation, allowing quantitative comparison under mixed-availability conditions.

**Table E.9** Impact of simulated V2X communication degradation on planning accuracy and safety. Higher packet loss probabilities correspond to reduced availability of infrastructure-side information.

| Packet Loss 𝑝                                |              | L2 Error (m) ↓ |              |              | Collision Rate (%) ↓ |              |              |              |  |
|----------------------------------------------|--------------|----------------|--------------|--------------|----------------------|--------------|--------------|--------------|--|
|                                              | 2.5s         | 3.5s           | 4.5s         | Avg.         | 2.5s                 | 3.5s         | 4.5s         | Avg.         |  |
| 𝑝 = .1                                       | 1.82         | 2.07           | 2.28         | 2.06         | 0.04                 | 0.05         | 0.06         | 0.05         |  |
| 𝑝 = .3                                       | 2.51         | 2.63           | 2.93         | 2.69         | 0.04                 | 0.05         | 0.07         | 0.05         |  |
| 𝑝 = .5                                       | 2.88         | 3.22           | 3.73         | 3.28         | 0.04                 | 0.07         | 0.07         | 0.06         |  |
| UniV2X (Yu et al. (2024))<br>V2X-VLM (𝑝 = 0) | 2.59<br>1.09 | 3.35<br>1.12   | 4.49<br>1.42 | 3.48<br>1.21 | 0.00<br>0.02         | 0.44<br>0.03 | 0.59<br>0.03 | 0.34<br>0.03 |  |

As summarized in Table E.9, V2X-VLM exhibits graceful degradation as the loss probability increases. The L2 error rises from 2.06 m at = 0*.*1 to 3.28 m at = 0*.*5, while the average collision rate remains below 0.06 %. Even in the most severe condition ( = 0*.*5), the model still outperforms the UniV2X baseline (Yu et al. (2024)) by a large margin in most horizons. This demonstrates that V2X-VLM can maintain stable and safe planning performance even when the cooperative information from infrastructure cameras becomes partially unavailable.

Although the model is trained using fully cooperative vehicle-infrastructure pairs, its performance under partial loss shows strong generalization and resilience. This robustness stems from the VLM backbone and multimodal fusion design, which allow the model to infer semantic scene structures and motion intent directly from the ego-vehicle view. In deployment, this property ensures that transient communication failures, packet drops, or latency fluctuations will not lead to catastrophic degradation in planning quality, supporting the model's practicality for real-world V2X systems.

#### **Appendix F. Scene Understanding in VLM for Cooperative Autonomous Driving**

This section showcases the capability of VLMs in understanding driving scenes within the cooperative autonomous driving setup. The structured textual descriptions presented in Fig. D.9 are derived from VLM-based scene interpretation and are crafted as semantic prompts for training V2X-VLM.

The structured textual inputs include the following components:

- **Brief Scene Description:** Summarizes the fundamental scene attributes, including weather conditions, time of day, type of road environment, and the current position of the ego-vehicle.
- **Detailed Scene Description:** Provides a more comprehensive semantic interpretation, describing surrounding vehicles, traffic density, infrastructure elements, and interaction dynamics within the scene.

These VLM-generated descriptions offer a high-level semantic representation of the driving environment, which is integrated into V2X-VLM's input alongside vehicle and infrastructure camera images. By incorporating multimodal inputs, the model gains a deeper contextual understanding of driving scenarios, improving cooperative end-to-end trajectory planning. This highlights the effectiveness of VLMs in extracting structured knowledge from visual data to facilitate enhanced decision-making in cooperative autonomous driving.

# **References**

Chen, S., Jiang, B., Gao, H., Liao, B., Xu, Q., Zhang, Q., Huang, C., Liu, W., Wang, X., 2024. Vadv2: end-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243

Chen, Z., Shi, Y., Jia, J., 2023. Transiff: an instance-level feature fusion framework for vehicle-infrastructure cooperative 3d detection with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18205–18214.

Cui, J., Qiu, H., Chen, D., Stone, P., Zhu, Y., 2022. Coopernaut: end-to-end driving with cooperative perception for networked vehicles. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17252–17262.

- Fu, D., Li, X., Wen, L., Dou, M., Cai, P., Shi, B., Qiao, Y., 2024. Drive like a human: rethinking autonomous driving with large language models. In: 2024 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW). IEEE, pp. 910–919.
- Gao, Z., Mu, Y., Chen, C., Duan, J., Luo, P., Lu, Y., Li, S.E., 2024. Enhance sample efficiency and robustness of end-to-end urban autonomous driving via semantic masked world model. IEEE Trans. Intell. Transp. Syst.
- Guo, M., Zhang, Z., He, Y., Wang, K., Jing, L., 2024. End-to-end autonomous driving without costly modularization and 3d manual annotation. arXiv preprint arXiv:2406.17680
- Hinton, G., Vinyals, O., Dean, J., 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531
- Hu, S., Chen, L., Wu, P., Li, H., Yan, J., Tao, D., 2022a. St-p3: end-to-end vision-based autonomous driving via spatial-temporal feature learning. In: European Conference on Computer Vision. Springer, pp. 533–549.
- Hu, Y., Fang, S., Lei, Z., Zhong, Y., Chen, S., 2022b. Where2comm: communication-efficient collaborative perception via spatial confidence maps. Adv. Neural Inf. Process. Syst. 35, 4874–4886.
- Hu, Y., Lu, Y., Xu, R., Xie, W., Chen, S., Wang, Y., 2023a. Collaboration helps camera overtake lidar in 3d detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9243–9252.
- Hu, Y., Yang, J., Chen, L., Li, K., Sima, C., Zhu, X., Chai, S., Du, S., Lin, T., Wang, W., et al., 2023b. Planning-oriented autonomous driving. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17853–17862.
- Huang, Z., Sheng, Z., Ma, C., Chen, S., 2024a. Human as AI mentor: enhanced human-in-the-loop reinforcement learning for safe and efficient autonomous driving. Commun. Transp. Res. 4, 100127.
- Huang, Z., Sheng, Z., Qu, Y., You, J., Chen, S., 2024b. Vlm-rl: a unified vision language models and reinforcement learning framework for safe autonomous driving. arXiv preprint arXiv:2412.15544
- Hwang, J.-J., Xu, R., Lin, H., Hung, W.-C., Ji, J., Choi, K., Huang, D., He, T., Covington, P., Sapp, B., et al., 2024. Emma: end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262
- Jia, X., Yang, Z., Li, Q., Zhang, Z., Yan, J., 2025. Bench2drive: towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. Adv. Neural Inf. Process. Syst. 37, 819–844.
- Jiang, B., Chen, S., Xu, Q., Liao, B., Chen, J., Zhou, H., Zhang, Q., Liu, W., Huang, C., Wang, X., 2023. Vad: vectorized scene representation for efficient autonomous driving. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8340–8350.
- Jiao, S., Fang, Y., 2024. Lavida drive: vision-text interaction VLM for autonomous driving with token selection, recovery and enhancement. arXiv preprint arXiv:2411.12980
- Khan, D., Aslam, S., Chang, K., 2025. Vehicle-to-infrastructure multi-sensor fusion (v2i-MSF) with reinforcement learning framework for enhancing autonomous vehicle perception. IEEE Access .
- Li, Y., Fan, L., He, J., Wang, Y., Chen, Y., Zhang, Z., Tan, T., 2024a. Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481
- Li, Y., Yuan, D., Zhang, H., Yang, Y., Luo, X., 2024b. End to end autonomous driving via occupancy and motion flow. In: 2024IEEE International Conference on Real-time Computing and Robotics (RCAR). IEEE, pp. 360–365.
- Li, Z., Li, K., Wang, S., Lan, S., Yu, Z., Ji, Y., Li, Z., Zhu, Z., Kautz, J., Wu, Z., et al., 2024c. Hydra-MDP: end-to-end multimodal planning with multi-target hydradistillation. arxiv. arXiv preprint arXiv:2406.06978
- Li, Z., Yu, Z., Lan, S., Li, J., Kautz, J., Lu, T., Alvarez, J.M., 2024d. Is ego status all you need for open-loop end-to-end autonomous driving? In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14864–14873.
- Liao, B., Chen, S., Yin, H., Jiang, B., Wang, C., Yan, S., Zhang, X., Li, X., Zhang, Y., Zhang, Q., et al., 2024. Diffusiondrive: truncated diffusion model for end-to-end autonomous driving. arXiv preprint arXiv:2411.15139
- Liu, H., Yao, R., Liu, W., Huang, Z., Shen, S., Ma, J., 2025. CodriveVLM: VLM-enhanced urban cooperative dispatching and motion planning for future autonomous mobility on demand systems. arXiv preprint arXiv:2501.06132
- Liu, L., Sun, X., Xiang, T., Zhuang, Z., Yin, L., Tan, M., 2023. Contrastive vision-language alignment makes efficient instruction learner. arXiv preprint arXiv:2311.17945 Long, K., Shi, H., Liu, J., Li, X., 2024. Vlm-mpc: vision language foundation model (vlm)-guided model predictive controller (mpc) for autonomous driving. arXiv preprint arXiv:2408.04821
- Lu, Y., Li, Q., Liu, B., Dianati, M., Feng, C., Chen, S., Wang, Y., 2023. Robust collaborative 3d object detection in presence of pose errors. In: 2023IEEE International Conference on Robotics and Automation (ICRA). IEEE, pp. 4812–4818.
- Ma, Y., Cao, Y., Sun, J., Pavone, M., Xiao, C., 2024. Dolphins: multimodal language model for driving. In: European Conference on Computer Vision. Springer, pp. 403–420.
- Mahjourian, R., Kim, J., Chai, Y., Tan, M., Sapp, B., Anguelov, D., 2022. Occupancy flow fields for motion forecasting in autonomous driving. IEEE Rob. Autom. Lett. 7 (2), 5639–5646.
- Mo, Y., Vijay, R., Rufus, R., Boer, N.d., Kim, J., Yu, M., 2024. Enhanced perception for autonomous vehicles at obstructed intersections: an implementation of vehicle to infrastructure (v2i) collaboration. Sensors 24 (3), 936.
- Shao, H., Hu, Y., Wang, L., Song, G., Waslander, S.L., Liu, Y., Li, H., 2024. Lmdrive: closed-loop end-to-end driving with large language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15120–15130.
- Shao, H., Wang, L., Chen, R., Waslander, S.L., Li, H., Liu, Y., 2023. Reasonnet: end-to-end driving with temporal and global reasoning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13723–13733.
- Sima, C., Renz, K., Chitta, K., Chen, L., Zhang, H., Xie, C., Beißwenger, J., Luo, P., Geiger, A., Li, H., 2024. Drivelm: driving with graph visual question answering. In: European Conference on Computer Vision. Springer, pp. 256–274.
- Sun, W., Lin, X., Shi, Y., Zhang, C., Wu, H., Zheng, S., 2024. Sparsedrive: end-to-end autonomous driving via sparse scene representation. arXiv preprint arXiv:2405.19620
- Tian, X., Gu, J., Li, B., Liu, Y., Wang, Y., Zhao, Z., Zhan, K., Jia, P., Lang, X., Zhao, H., 2024. Drivevlm: the convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289
- Wang, T., Zhou, W., Zeng, Y., Zhang, X., 2022. Efficientvlm: fast and accurate vision-language models via knowledge distillation and modal-adaptive pruning. arXiv preprint arXiv:2210.07795
- Wang, T.-H., Manivasagam, S., Liang, M., Yang, B., Zeng, W., Urtasun, R., 2020. V2vnet: Vehicle-to-vehicle communication for joint perception and prediction. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. Springer, pp. 605–621.
- Wang, X., Zhu, Z., Huang, G., Chen, X., Zhu, J., Lu, J., 2024a. Drivedreamer: towards real-world-drive world models for autonomous driving. In: European Conference on Computer Vision. Springer, pp. 55–72.
- Wang, Y., He, J., Fan, L., Li, H., Chen, Y., Zhang, Z., 2024b. Driving into the future: multiview visual forecasting and planning with world model for autonomous driving. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14749–14759.
- Xiao, B., Wu, H., Xu, W., Dai, X., Hu, H., Lu, Y., Zeng, M., Liu, C., Yuan, L., 2024. Florence-2: advancing a unified representation for a variety of vision tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4818–4829.
- Xu, R., Chen, C.-J., Tu, Z., Yang, M.-H., 2024a. V2x-Vitv2: improved vision transformers for vehicle-to-everything cooperative perception. IEEE Trans. Pattern Anal. Mach. Intell.
- Xu, R., Tu, Z., Xiang, H., Shao, W., Zhou, B., Ma, J., 2022a. CoBEVT: cooperative bird's eye view semantic segmentation with sparse transformers. arXiv preprint arXiv:2207.02202
- Xu, R., Xiang, H., Tu, Z., Xia, X., Yang, M.-H., Ma, J., 2022b. V2x-Vit: vehicle-to-everything cooperative perception with vision transformer. In: European Conference on Computer Vision. Springer, pp. 107–124.
- Xu, Z., Zhang, Y., Xie, E., Zhao, Z., Guo, Y., Wong, K.-Y.K., Li, Z., Zhao, H., 2024b. Drivegpt4: interpretable end-to-end autonomous driving via large language model. IEEE Rob. Autom. Lett.

- Ye, T., Jing, W., Hu, C., Huang, S., Gao, L., Li, F., Wang, J., Guo, K., Xiao, W., Mao, W., et al., 2023. Fusionad: multi-modality fusion for prediction and planning tasks of autonomous driving. arXiv preprint arXiv:2308.01006
- Yi, S., Zhang, H., Liu, K., 2024. V2IVIewer: towards efficient collaborative perception via point cloud data fusion and vehicle-to-infrastructure communications. IEEE Trans. Netw. Sci. Eng.
- Yu, H., Luo, Y., Shu, M., Huo, Y., Yang, Z., Shi, Y., Guo, Z., Li, H., Hu, X., Yuan, J., et al., 2022. Dair-v2x: a large-scale dataset for vehicle-infrastructure cooperative 3d object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21361–21370.
- Yu, H., Yang, W., Ruan, H., Yang, Z., Tang, Y., Gao, X., Hao, X., Shi, Y., Pan, Y., Sun, N., et al., 2023. V2x-Seq: a large-scale sequential dataset for vehicle-infrastructure cooperative perception and forecasting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5486–5495.
- Yu, H., Yang, W., Zhong, J., Yang, Z., Fan, S., Luo, P., Nie, Z., 2024. End-to-end autonomous driving through v2x cooperation. arXiv preprint arXiv:2404.00717
- Yuan, C., Zhang, Z., Sun, J., Sun, S., Huang, Z., Lee, C. D.W., Li, D., Han, Y., Wong, A., Tee, K.P., et al., 2024. Drama: an efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601
- Zeng, Y., Jiang, C., Mao, J., Han, J., Ye, C., Huang, Q., Yeung, D.-Y., Yang, Z., Liang, X., Xu, H., 2023. Clip2: contrastive language-image-point pretraining from real-world point cloud data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15244–15253.
- Zhang, Y., Nie, Y., 2024. Interndrive: a multimodal large language model for autonomous driving scenario understanding. In: Proceedings of the 2024 4Th International Conference on Artificial Intelligence, Automation and High Performance Computing, pp. 294–305.
- Zhang, Z., Meyer, G.P., Lu, Z., Shrivastava, A., Ravichandran, A., Wolff, E.M., 2024. Vlm-kd: knowledge distillation from vlm for long-tail visual recognition. arXiv preprint arXiv:2408.16930
- Zheng, W., Song, R., Guo, X., Zhang, C., Chen, L., 2024a. Genad: generative end-to-end autonomous driving. In: European Conference on Computer Vision. Springer, pp. 87–104.
- Zheng, W., Wu, J., Zheng, Y., Zuo, S., Xie, Z., Yang, L., Pan, Y., Hao, Z., Jia, P., Lang, X., et al., 2024b. GaussianAD: gaussian-centric end-to-end autonomous driving. arXiv preprint arXiv:241.10371