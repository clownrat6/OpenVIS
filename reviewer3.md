Dear Reviewer,

Thank you for your thorough review and constructive feedback. We appreciate the opportunity to address your concerns.
****
**Comment**: "In the proposed pipeline, most modules are based on other existing works, such as for the VLP, segmentor among others. It will improve the paper, if the author could explain the reason to choose them, for example Mask2Former for the segmentor."

**Response**: There are two reasons about why we incorporate modules from existing works such as Mask2Former and CLIP into our pipeline:
1. There modules are commonly used in previous studies (e.g., OpenVIS/InstCLIP), allowing us to make fair comparisons to previous works.
2. Our main contribution lies in introducing Brownian bridge modeling to capture motion dynamics and reduce inference overhead. 
Integrating these modules to construct a stronger pipeline strengthens the credibility of our main contribution by showcasing the improvements of our method on this enhanced baseline. Nevertheless, the construction of the pipeline is auxiliary and not our main contribution.

Incorporating these modules to build a stronger pipeline can enhance the the credibility of our main contribution by demonstrating our method's improvements on this stronger baseline. 


and demonstrating improvements of our method on this stronger baseline aim at enhancing the credibility of our experiments. However, The construction of the pipeline is not our main contribution.

Because our method is architecture-agnostic, we build a simple and effective baseline for verify our method. However, 

while incorporating these modules to build a 

我们的主要贡献在于提出了布朗桥建模来进行运动建模并降低推理开销。我们的方法是架构无关的，所以我们构建了一个简单且有效的 pipeline 来更好地验证方法。但构建 pipeline 并不是我们的主要贡献。
The construction of the pipeline is not our main contribution, while our approach is architecture-agnostic

reduce the costs of architecture design and hyperparameter tuning because they are effective, simple, and of high open-source quality.
2. Intergrating these modules as a pipeline and tuning this pipeline to achieve better performance are constructive because the open-sourced code of this pipeline will provide a stronger baseline for segmentation community. Moreover, since our approach is architecture-agnostic, achieving performance improvements on a stronger baseline enhances the credibility of our experiments.

****
**Comment**: "The idea ... could be better validated. For efficiency, there is no comparison result between the proposed work and other SOTAs."

**Response**: 
1. Since inference overhead is determined by multiple components (e.g., segmentor, VLP) of the whole framework, comparing inference overhead across different frameworks may not effectively demonstrate the efficiency of our approach. Therefore, to control experimental variables, we conduct ablation studies on self-constructed baseline to validate the effectiveness of Brownian bridge modeling in reducing inference overhead.
2. Since our method is architecture-agnoistic, the Brownian bridge modeling can be adopted to different baselines for reducing inference overhead. For example, our method can accelerate OpenVIS from 3.5s/iter to 2.8s/iter.
****
**Comment**: "This paper is missing some related work to discuss or compare with. For example, Open-vocabulary SAM .... SAM was not mentioned or discussed in this paper."

**Response**:
Thanks for your valuable suggestions. We will consider incorporating the concepts from Open-vocabulary SAM and SAM into the design of our next version model and include relevant discussions in our revised version.