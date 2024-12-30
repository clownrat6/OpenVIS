Dear Reviewer,
    
Thanks for your valuable feedback. We are delighted to see that you find our Brownian bridge modeling is **novel and interesting**, our method **achieves sota performance**, and our paper is **well written**.
****
**Comment**: "The proposed model architecture is very similar to DVIS."

**Response**: We acknowledge that our scheme draws from the DVIS online version, but applying it to build a strong open-vocabulary VIS pipeline still requires significant effort. For instance, we adopt an attention bias-based projector to transform instance queries into instance-aware CLIP embeddings, allowing us to calculate alignment scores with text for open-vocabulary classification.
****
**Comment**: How do attention biases, instance queries, and category embeddings interact.

**Response**: There are three steps:
<!-- 1. 获取 instance queries $Q$。假设视频有 $T$ 帧，每帧都能提取 $N$ 个 instance queries ，进行追踪匹配之后，可以得到整个视频的 instance queries $Q\in\mathbb{R}^{T\times N \times c}$。
2. 获取 instance masks $M$。我们将 instance queries $Q$ 与 pixel embedding 相乘得到 $N$ 个 mask $M\in\mathbb{R}^{T\times N \times H\times W}$ 。
3. 获取 Instance Embeddings $E$。 -->

1. Multiply the $N$ instance queries with pixel embeddings across $T$ frames to obtain the segmentation masks $M\in\mathbb{R}^{T\times N \times H\times W}$, These masks are then flattened into attention biases $B\in\mathbb{R}^{T\times N \times L}$, where $L=H\times W$.
2. Input the $T$ frames into CLIP to perform patchification, resulting in patch embeddings $P\in\mathbb{R}^{T\times L \times c}$. The class token from CLIP is replicated $T\times N$ times to generate the initial instance embeddings $E\in\mathbb{R}^{T\times N \times c}$. These are then concatenated to form the CLIP tokens $C\in\mathbb{R}^{T\times (N + L) \times c}$.
3. Add the attention bias to the attention weights $W\in\mathbb{R}^{T\times (N+L) \times (N+L)}$ in the last three attention layers (i.e., $W[:, :N, N:N+L] += B$), when using the CLIP image encoder to perform inference on $C$. 
4. Extract the first $N$ CLIP tokens as the projected instance embeddings $E$. Finally, compute alignment scores between these embeddings and the category embeddings for open-vocabulary classification.


1. 将 $T$ 帧的 $N$ 个 Instance Queries 与 Pixel Embeddings 相乘，得到分割 masks $M\in\mathbb{R}^{T\times N \times H\times W}$，将其 flatten 成 attention bias $B\in\mathbb{R}^{T\times N \times L}$，其中 $L=H\times W$。
1. 将 $T$ 帧图像输入 CLIP 进行 patchify，得到 Patch Embeddings $P\in\mathbb{R}^{T\times L \times c}$。将 CLIP 的 class token 复制 $T\times N$ 份，得到初始的 Instance Embeddings $E\in\mathbb{R}^{T\times N \times c}$。将二者进行拼接得到 CLIP tokens $C\in\mathbb{R}^{T\times (N + L) \times c}$。
3. 使用 CLIP Image Encoder 对 $C$ 进行推理，将 Attention Bias 加到最后三层 Attention Layer 的 Attention Weights $A\in\mathbb{R}^{T\times (N+L) \times (N+L)}$ 上 (i.e., $A[:, :N, N:N+L] += B$)。
4. 将前 $N$ 个 CLIP token 取出来作为经过投影的 Instance Embeddings $Q$，最终与 Category Embeddings 计算对齐分数来进行开放词汇分类。

将 $T$ 帧图像输入 CLIP Image Encoder 进行编码时中间特征为 $F\in\mathbb{R}^{T\times HW \times c}，将 $M$ 作为 Attention Bias 输入最后三层 Attention Layer 中的 Self-Attention Block。
4. 计算匹配分数。将 Instance Embeddings 与 Category Embeddings 进行点积，计算匹配分数。

将各帧 100 个 instance queries 与 pixel embedding 相乘得到 100 个 mask，将 100 个 mask 复制 num_heads 份得到 100 个 Attention Bias。
2. 在对各帧图像使用 CLIP Image Encoder 进行编码时，将各帧的 100 个 Attention Bias 输入当前帧编码的最后三层 Self-Attention 最终为每帧获得 100 个 Instance-aware class tokens
3. 将 class token 与 category embeddings 进行点积
****
**Comment**: How did the author allocate base and novel categories?

**Response**: 
****
**Comment**: It is unclear whether the performance improvement comes from the novel or base categories.

**Response**: 
