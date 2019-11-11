# Ideas
- Paper: [https://arxiv.org/pdf/1909.08072.pdf](https://arxiv.org/pdf/1909.08072.pdf)
- Use VAE to denoise.
- Use robust manifold defenses
- Loss function: CE + a * KLD + class_error(real) - sum(class_error(other)) + proximity_loss - distance_loss
    - Proximity loss: [https://arxiv.org/pdf/1904.00887.pdf](https://arxiv.org/pdf/1904.00887.pdf), K-Means
    - Topological loss: [https://arxiv.org/pdf/1909.03334.pdf](https://arxiv.org/pdf/1909.03334.pdf)

Test Case:
* Use pytorch pretrained CNN as classifier, and use ImageNet as dataset
* [https://cloud.google.com/vision](https://cloud.google.com/vision)
* White-box attack (models involve training)
* Black-box attack (models outside of training)
* Compare with [Defense-GAN](https://arxiv.org/pdf/1805.06605.pdf) and [Defense-VAE](https://arxiv.org/pdf/1812.06570.pdf)