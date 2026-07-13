# MSPath: A Multi-Scale Vision-Language Model with Clinical Information Prompting for Pathology Report Generation]🔬

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-red)](https://pytorch.org/)

## Abstract

As the gold standard for disease diagnosis, the interpretation of histopathological images has long been constrained by inter-observer variability in manual reading and the substantial clinical workload, thereby highlighting the urgent need for AI-assisted generation of standardized pathology reports. However, existing methods are often limited by the computational bottlenecks associated with whole-slide images, where coarse-grained sampling may lead to the loss of diagnostically critical details. Moreover, purely vision-based models generally lack clinical priors, resulting in superficial image descriptions that are insufficient for precise diagnostic decision-making.To address these limitations, a deep hashing-based salient feature selection mechanism is proposed, by which the computational burden is substantially reduced while microscopic regions with the highest diagnostic value are preserved. To mitigate semantic insufficiency, patient clinical information is incorporated as a global prior into the model, thereby effectively alleviating the representation alignment bias between visual features and diagnostic conclusions. Finally, the reasoning capability of large language models is leveraged to translate multimodal features into standardized pathology reports that conform to medical reporting conventions.Experiments conducted on two public datasets and one private dataset demonstrate that MSPath achieves state-of-the-art performance in both diagnostic accuracy and clinical relevance.




## Main Contributions
	1. MSPath, a vision-language model framework that integrates clinical priors with multi-scale visual features, is proposed. To overcome the limited contextual awareness of existing purely vision-based models in complex diagnostic scenarios, patient clinical information is incorporated into the model as a global prior. In this manner, the semantic mapping bias between microscopic pathological morphological features and macroscopic clinical diagnostic conclusions is effectively mitigated, thereby ensuring the clinical accuracy and logical consistency of the generated reports.
	
	2. A deep hashing-based salient region selection mechanism is designed to substantially reduce the computational cost of gigapixel-level whole-slide images. In contrast to conventional coarse-grained downsampling strategies, which may lead to the loss of critical pathological details, redundant information is efficiently filtered through hash encoding. Consequently, the computational burden is significantly reduced, while diagnostically informative microscopic regions are preserved and computational efficiency is improved.
	
	3. A high-fidelity pathology report generation paradigm driven by large language models is constructed. Multimodal features are transformed into standardized pathology reports that strictly conform to medical writing conventions, while the need for complex vocabulary construction and task-specific decoder design is eliminated. Extensive experiments on two public datasets and one private dataset demonstrate that MSPath achieves state-of-the-art performance in terms of diagnostic accuracy and text quality, indicating its strong generalization capability and promising potential for clinical application.

