<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/serve-kosmos-2/releases/download/v0.0.1/kosmos_2.png"/>  

# Serve Kosmos 2

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/serve-kosmos-2)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-kosmos-2)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/serve-kosmos-2.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/serve-kosmos-2.png)](https://supervisely.com)

</div>

# Overview

KOSMOS-2 is a multimodal large language model that has new capabilities of multimodal grounding and referring. KOSMOS-2 can understand multimodal input, follow instructions, perceive object descriptions (bounding boxes) and ground language to the visual world.

![kosmos 2](https://github.com/supervisely-ecosystem/serve-kosmos-2/releases/download/v0.0.1/kosmos_2_capabilities.png)

Based on KOSMOS-1, KOSMOS-2 enhances multimodal large language models by incorporating grounding and referring capabilities. KOSMOS-2 also uses a Transformer-based causal language model as the backbone and is trained with the next-token prediction task. KOSMOS-2 shows new capabilities of grounding and referring. The referring capability enables to point out image regions with bounding boxes. KOSMOS-2 can understand the image regions users refer to by the coordinates of bounding boxes. The referring capability provides a new interaction
method. Different from previous MLLMs, which can only provide text output, KOSMOS-2 can provide visual answers (bounding boxes) and ground text output to the image. The grounding capability enables the model to provide more accurate, informative, and comprehensive responses.

# How To Run

**Step 1.** Select pretrained model architecture and press the **Serve** button

![pretrained_models](https://github.com/supervisely-ecosystem/serve-kosmos-2/releases/download/v0.0.1/kosmos_2_deploy.png)

**Step 2.** Wait for the model to deploy

![deployed](https://github.com/supervisely-ecosystem/serve-kosmos-2/releases/download/v0.0.1/kosmos_2_deploy_2.png)

# Acknowledgment

This app is based on the great work [Kosmos 2](https://github.com/microsoft/unilm/tree/master/kosmos-2). ![GitHub Org's stars](https://img.shields.io/github/stars/microsoft/unilm?style=social)
