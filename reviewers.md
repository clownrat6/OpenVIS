# Reviewer 1/#fX4s (score: 6, confidence: 4)

Dear Reviewer,

Thanks for your valuable feedback. We are delighted to see that you find our idea is **novel and reasonable**, our method **shows significant improvement**, and our manuscript is **well-organized**.
****
**Comment**: "Comparison with prior methods on additional benchmarks like YTVIS19, YTVIS21, and OVIS would be beneficial."

**Response**: We add additional benchmark results and provide corresponding analysis.
|Method|Open|Train|ytvis2019|ytvis2021|ovis|
|:-|:-:|:-:|:-:|:-:|:-:|
|Mask2Former|×|in-domain|46.4|40.6|17.3|
|DVIS|×|in-domain + coco|51.2|**46.4**|**33.8**| 
|OV2Seg|√|ytvis2019|46.2|37.7|14.3|
|BriVIS|√|ytvis2019|49.7|42.2|13.3|
|BriVIS|√|ytvis2019 + coco|**51.8**|43.1|19.6|

Analysis:
1. Our method is a highly effective VIS architecture. For example,  we improve SOTA closed-vocabulary model (DVIS) on the YTVIS 2019 val by 0.6 mAP, with the same training set (YTVIS 2019 + COCO).
2. Our method has robust open-vocabulary capabilities for cross-dataset transfer. For example, we outperform Mask2Former and achieve comparable results to DVIS on the YTVIS2021 val and OVIS val. Moreover, we exhibits better transfer performance compared to previous open-vocabulary methods. 
<!-- 3. Current methods trained on short video training datasets (YTVIS) still struggle to perform well on long video datasets (OVIS), because there is a significant domain gap between short and long videos. Addressing this gap remains a challenge that we aim to tackle in future work. -->
****
**Comment**: "There is no qualitative comparison with existing methods."

**Response**: According to the visualization of segmentation results, we find that we perform better on fast-moving objects compared to previous methods. For instance, with rapidly moving surfboards, our approach consistently achieves precise segmentation and tracking. In contrast, previous methods generate error segmentation when the surfboard was obscured by waves.

The relevant results will be included in the revised version.

****
**Comment**: "What are the advantages over recent work OVFormer?"

**Response**: There are two advantages:
1. Our BriVIS has better performance than OVFormer with same settings, e.g., our method achieve 27.7 mAP and improve upon OVFormer by 5.8 mAP on LV-VIS val, when training on LV-VIS train set.
2. Our Brownian bridge modeling can extra reduce inference cost, except enhancing temporal consistency like OVFormer.

We will attach more discussions about comparison with OVFormer on our revised version.

# Reviewer 2/#WPAc

Dear Reviwer,

Thanks for your valuable feedback. We are happy to see that you find our method **is effective for fast-moving instances**, **can work for uncommon instances**, **outperforms the other SOTA models**, and **is good enough to be accepted.**
****
**Comment**: 
"Although the relative gain compared with previous method is high, the overall performance for open-vocabulary video instance segmentation remains low."

**Response**: 
We discuss several challenges in open-vocabulary video instance segmentation that hinder current methods from achieving promising performance. We also propose feasible strategies to overcome these challenges, which will guide our future efforts.
* Lack of video segmentation data with diverse and rich vocabulary. Given the abundance and diversity of image segmentation data, we believe that joint training of open-vocabulary image and video segmentation can help mitigate this issue.
* Lack of adopting of effective video foundation model. With recent advancements in foundational video segmentation models like SAM2, combining their powerful temporal segmentation capabilities with CLIP's zero-shot classification ability could significantly advance the field of open-vocabulary video instance segmentation.

# Reviewer 3/#dF8u

Dear Reviewer,

Thank you for your thorough review and constructive feedback. We appreciate the opportunity to address your concerns.
****
**Comment**: "In the proposed pipeline, most modules are based on other existing works, such as for the VLP, segmentor among others. It will improve the paper, if the author could explain the reason to choose them, for example Mask2Former for the segmentor."

**Response**: There are three reasons about why we incorporate modules from existing works such as Mask2Former and CLIP into our pipeline:
1. The construction of the pipeline is auxiliary and primarily serves to bolster the credibility of our main contribution: "*introducing Brownian bridge modeling to capture motion dynamics and reduce inference overhead*". 
Integrating these modules can construct a pipeline which is stronger than previous methods. Showcasing the improvements of our method on this baseline can strengthen the credibility of our main contribution.
2. Mask2Former is a unified model for instance, semantic, and panoptic segmentation so building our method based on it allow us to extend to other tasks like open-vocabulary video panoptic segmentation. Moreover, choosing CLIP is mainly due to its excellent scalability, allowing our method to be more easily extended to larger datasets and bigger models for better performance.
3. There modules are commonly used in previous studies (e.g., OpenVIS/InstCLIP), allowing us to make fair comparisons to previous works.
****
**Comment**: "The idea ... could be better validated. For efficiency, there is no comparison result between the proposed work and other SOTAs."

**Response**: We compare efficiency with other methods and find that our approach maintains competitive performance with low inference overhead.
|Method|Train|burst|lvvis|Speed (s/iter)|
|:-|:-:|:-:|:-:|:-:|
|OV2Seg|lvvis|3.7|14.2|1.85|
|OpenVIS|ytvis2019 + coco|3.5|12.0|3.52|
|BriVIS|ytvis2019 + coco|**5.7**|**20.9**|1.88|
****
**Comment**: "This paper is missing some related work to discuss or compare with. For example, Open-vocabulary SAM .... SAM was not mentioned or discussed in this paper."

**Response**:
Thanks for your valuable suggestions. We will consider incorporating the concepts from Open-vocabulary SAM and SAM into the design of our next version model and include relevant discussions in our revised version.

# Reviewer 4/#a3Vg

Dear Reviewer,
    
Thanks for your valuable feedback. We are delighted to see that you find our Brownian bridge modeling is **novel and interesting**, our method **achieves sota performance**, and our paper is **well written**.
****
**Comment**: "The proposed model architecture is very similar to DVIS."

**Response**: We acknowledge that our scheme draws from the DVIS online version, but significant effort is still needed to build a strong open-vocabulary VIS pipeline. For instance, we adopt an attention bias-based projector to transform instance queries into instance-aware CLIP embeddings for open-vocabulary classification.
****
**Comment**: "How do attention biases, instance queries, and category embeddings interact."

**Response**: There are four steps:
1. Multiply the $N$ instance queries with pixel embeddings across $T$ frames to obtain the segmentation masks $M\in\mathbb{R}^{T\times N \times H\times W}$. Flatten these masks into attention biases $B\in\mathbb{R}^{T\times N \times L}$, where $L=H\times W$.
2. Input the $T$ frames into CLIP to obtain patch embeddings $P\in\mathbb{R}^{T\times L \times c}$. Replicate the class token of CLIP $T\times N$ times to generate initial instance embeddings $E\in\mathbb{R}^{T\times N \times c}$. Concatenate these to form the CLIP tokens $C\in\mathbb{R}^{T\times (N + L) \times c}$.
3. Add the attention biases to the attention weights $W\in\mathbb{R}^{T\times (N+L) \times (N+L)}$ in the last three layers (i.e., $W[:, :N, N:N+L] += B$), when using the CLIP image encoder for inference on $C$. 
4. Extract the first $N$ CLIP tokens as the projected instance embeddings $E$, and compute their dot product with category embeddings for open-vocabulary classification.
****
**Comment**: How did the author allocate base and novel categories?

**Response**: We only adopt YTVIS 2019 and COCO as our training set so we set 101 categories of these two datasets as base categories. In base-novel evaluation protocol, we designate the categories that overlap with these 101 classes in LV-VIS and BURST as base categories, while the remaining categories are considered novel.
****
**Comment**: It is unclear whether the performance improvement comes from the novel or base categories.

**Response**: Our method significantly improves performance in categories that rely heavily on strong temporal information, such as fast-moving objects. This improvement is evident in base categories like cars and in novel categories such as yo-yo and blimp.