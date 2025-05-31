---
#
# By default, content added below the "---" mark will appear in the home page
# between the top bar and the list of recent posts.
# To change the home page layout, edit the _layouts/home.html file.
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
#
layout: home
---
<img src="/images/banner2025_v2.png" style="pointer-events: none; user-select: none;">

 
<!-- 
<div style="
  border: 1px solid #ccc; 
  border-radius: 15px; 
  padding: 20px; 
  width: 100%; 
  background-color: #f9f9f9;">
  <h2 style="margin-top: -10px;">📰 <b>Latest Updates</b></h2>
    <ul>
    <span class="news-item-icon">📢</span> <b>author notifications have been released on Oct. 9th. 
	</b>
	<p>
    We have added a special fast track for papers reviewed at NeurIPS 2024 that were not accepted. Authors can submit their papers with a link to their (anonymous) OpenReview page, giving our program committee access to the reviews of their paper.
  	</p>
	<span class="news-item-icon">📢</span> Our Submission Portal will remain open until September 18th, AOE, for editing existing submitted papers. 
     Add more news items as needed 
  </ul>
</div> 
-->

<p>
The Efficient Natural Language and Speech Processing (ENLSP) Workshop aims to advance the efficiency of large language and speech models across three key dimensions: <b>Model Design</b>, <b>Training</b>, and <b>Inference</b>—with a particular emphasis this year on <b>reasoning capabilities</b> and <b>test-time compute</b>.
While enabling models to ``think longer'' has shown promise, they must also <b><i>think smarter</i></b> that requires a stronger focus on efficiency in how models reason and adapt across diverse deployment settings.
ENLSP brings together academic researchers and industry experts in a dynamic program featuring <i>invited talks</i>, <i>panel discussions</i>, peer-reviewed <i>paper</i> and <i>poster presentations</i>, a <i>demo track</i>, and <i>mentorship sessions</i> for new researchers.
The workshop provides a unique opportunity to explore emerging challenges in efficient AI, exchange ideas, and foster collaborations to bring together researchers from the machine learning, systems, optimization, theory and applied AI communities.
</p>


<h2 class="blackpar_title" id="overview">Overview</h2>
<p>
Recent advances in large language models (LLMs) (such as <a href="https://arxiv.org/abs/2412.16720">OpenAI o1</a>, 
<a href="https://arxiv.org/abs/2501.12948">DeepSeek</a>, 
<a href="https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf">Gemini 2.5</a>, 
<a href="https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf">GPT-4.5</a>, 
<a href="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct">Llama 3.3</a>, 
and <a href="https://arxiv.org/abs/2505.09388">Qwen3</a>), 
alongside traditional speech models (e.g., 
<a href="https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf">Nvidia NeMo Parakeet-TDT</a>, 
<a href="https://arxiv.org/abs/2410.15608">Moonshine</a>, 
<a href="https://arxiv.org/abs/2501.02832">Samba-ASR</a>, 
<a href="https://arxiv.org/abs/2402.08093">Amazon BASE TTS</a>, 
<a href="https://arxiv.org/abs/2503.01710">Spark-TTS</a>, 
<a href="https://arxiv.org/abs/2306.15687">Meta Voicebox</a>, 
and <a href="https://proceedings.mlr.press/v202/radford23a/radford23a.pdf">Whisper</a>) 
and modern speech language models (e.g., 
<a href="https://moshi-ai.com/">Moshi</a>, 
<a href="https://arxiv.org/abs/2503.20215">Qwen2.5-Omni</a>, 
and <a href="https://arxiv.org/abs/2310.13289">SALMONN</a>) 
have significantly transformed the field of AI, powering breakthroughs in code generation, scientific discovery, conversational agents, real-time interactions, multimodal integration and beyond. 
Generative models have showcased remarkable capabilities in understanding, reasoning, and generating human-like text and speech. 
More recently, a new paradigm has emerged centered on enhancing reasoning abilities through large-scale reinforcement learning 
(e.g., <a href="https://arxiv.org/abs/2412.16720">OpenAI O1</a> and <a href="https://arxiv.org/abs/2405.04434">DeepSeek-V2</a>) 
and by scaling test-time compute 
<a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">[Link1]</a>, 
<a href="https://arxiv.org/abs/2407.21787">[Link2]</a>. 
Allocating additional compute during inference effectively allows models to “<b>think longer</b>,” thereby improving their reasoning in a manner akin to human deliberation.
</p>
<p>
These advancements have been fueled by increasingly large model sizes, vast pretraining datasets, and access to powerful GPUs. 
However, this advancement comes at a steep cost: rising computational demands raise serious concerns around scalability, accessibility, and environmental impact. 
A notable example is <a href="https://arxiv.org/abs/2501.12948">DeepSeek R1</a>, a 671B-parameter reasoning model, with an estimated training cost of <b>$5.6 million</b> <a href="https://arxiv.org/abs/2412.19437">[Link]</a>, far beyond the means of most academic labs. 
Moreover, the growing reliance on increased test-time compute to boost reasoning further worsens the issue, especially as it often requires handling much longer sequences. 
This is particularly challenging for Transformer models, whose compute and memory requirements scale quadratically with sequence length 
<a href="https://arxiv.org/abs/2310.12109">[Link1]</a>, <a href="https://arxiv.org/abs/2307.14995">[Link2]</a>.
Overcoming these barriers calls for efficiency-focused innovation in model architecture, training strategies, compression, and inference-time optimization, especially for reasoning tasks.
</p>
<p>
Building on the momentum of previous editions, this year’s workshop continues to advance the efficiency of language and speech models, while introducing several timely and impactful directions to address emerging challenges and broaden community engagement:
</p>

<p><strong>(1) Efficient Architectures:</strong> 
We spotlight architectures that balance performance with computational and memory efficiency. While Transformers dominate, their quadratic attention and KV cache overhead limit scalability 
<a href="xxx">Zhang et al. 2023</a>. 
New directions include sub-quadratic models (e.g., 
<a href="xxx">Mamba</a>, 
<a href="xxx">Hedgehog</a>, 
<a href="xxx">xLSTM</a>, 
and <a href="xxx">RVKW</a>), hybrid designs (e.g., 
<a href="xxx">Jamba</a>, 
<a href="xxx">Nemotron-H</a>), and diffusion-based language models 
<a href="xxx">Nie et al. 2025</a>, 
<a href="xxx">Li et al. 2025</a>.
</p>

<p><strong>(2) Custom Model Composition:</strong> 
As applications diversify, we need modular, efficient frameworks that adapt pre-trained models without full retraining. This enables customization across hardware and user needs, beyond what compression or NAS alone can offer 
<a href="xxx">Muralidharan et al. 2024</a>, 
<a href="xxx">Bercovich et al. 2025a</a>, 
<a href="xxx">Bercovich et al. 2025b</a>.
</p>

<p><strong>(3) Efficient Training and Knowledge Transfer:</strong> 
Efficient models often rely on expensive training (e.g., 
<a href="xxx">Samba</a> and 
<a href="xxx">Hymba</a>, which involve pre-training) or suffer from poor transfer (e.g., 
<a href="xxx">MambaInLLaMA</a>). We invite solutions for effective post-training adaptation, and KV compression that preserve model quality at lower cost.
</p>

<p><strong>(4) Efficient Reasoning Models:</strong> 
Models like OpenAI’s O1 and DeepSeek show promise in reasoning, but their pipelines (training regimes, data curation, and design decisions) remain opaque. We encourage work that improves transparency and efficiency of reasoning models across training, data, inference, and deployment.
</p>

<p><strong>(5) Adaptive Test-Time Compute:</strong> 
Reasoning often benefits from increased inference compute. We seek methods like speculative decoding (e.g., multi-branch CoT) and adaptive compute allocation to extend “thinking time” based on task complexity without incurring quadratic costs.
</p>

<p><strong>(6) Evaluation and Benchmarking:</strong> 
Robust, compute-aware evaluation is essential. We seek evaluation methodologies that incorporate compute-aware metrics—such as latency-quality trade-offs, and energy efficiency—to assess models under practical deployment conditions, including edge and low-power scenarios.
</p>

<p>
Building upon the framework of our previous three editions, this workshop remains dedicated to investigating solutions for enhancing the efficiency of pre-trained language and foundation models but with introducing some fresh and important new topics to the community and encouraging their contributions.
Just to highlight a few: <b>(1)</b> Despite the ubiquitous usage of Transformers, they suffer from quadratic computational complexity which limits their efficiency especially for longer sequence lengths. Should we improve the efficiency of Transformers (e.g. in <a href="https://openreview.net/forum?id=4g02l2N2Nx">Hedgehog</a>, <a href="http://arxiv.org/abs/2312.06635">Gated Linear Attention</a>) or look for other architectures (e.g. <a href="https://arxiv.org/abs/2312.00752">Mamba</a>, <a href="http://arxiv.org/abs/2403.19887">Jamba</a>, <a href="http://arxiv.org/abs/2305.13048">RVKW</a>, <a href="http://arxiv.org/abs/2405.04517">xLSTM</a>, and <a href="http://arxiv.org/abs/2111.00396">SSMs</a>)? <b>(2)</b> For accelerating training, we have seen the significant impact of designing hardware efficient implementations such as in <a href = "http://arxiv.org/abs/2205.14135">Flash Attention</a>. Should we focus more on these hardware-aware solutions or more on new/improved architectures?
<b>(3)</b> For efficient inference, there are solutions such as: Speculative Decoding <a href="http://arxiv.org/abs/2302.01318">[Link1]</a> <a href="http://arxiv.org/abs/2211.17192">[Link2]</a> where the performance is strongly model and task-dependent and the draft and target models should have the same vocabulary (tokenizer); improved KV-caching (e.g. <a href="http://arxiv.org/abs/2306.14048">[Link]</a>) which has a limited speed-up; and many-in-one models such as <a href="http://arxiv.org/abs/2309.00255">SortedNet</a>, <a href="http://arxiv.org/abs/2310.07707">MatFormer</a>, and <a href="http://arxiv.org/abs/2404.16710">LayerSkip</a> but the performance of sub-models drops compared to their corresponding individual models.
<b>(4)</b> While there are many so-called efficient solutions in the literature, there is no fair, comprehensive and practical evaluation of these models and their comparison to each other. For example, we do not know the hallucination extent of the new architectures vs. the transformer model (e.g. in <a href="http://arxiv.org/abs/2402.01032">[Link]</a>). 
</p>

<!-- Call for Papers -->
<h2 class="blackpar_title" id="call_for_papers">Call for Papers</h2>
<p>
Investing in the future of language and foundation models requires a concrete effort to enhance their efficiency across multiple dimensions (including architecture, training, and inference) and having a comprehensive evaluation framework. 
To encourage engagement from the NeurIPS community, we present several active research topics in this field that invite participation and contributions. The scope of this workshop includes, but not limited to, the following topics:
<br><br>
<b>Efficient Architectures</b> Proposing alternative architectures that are more efficient than Transformers (in terms of computational complexity, memory footprint, handling longer sequence lengths ) or modifying Transformer architectures to make them more efficient  
<ul>
	<li>Linear and sub-quadratic Transformers , sparse attention Transformers</li>
	<li>New architures for LLMs and foundation models and their scalability</li>
	<li>Evaluation and benchmarking of new architectures (fair comparison of different models)</li>
	<li>Long sequence modeling</li>
	<li>Dense vs. sparse architectures (MoEs)</li>
</ul>
<b>Efficient Training</b> How can we reduce the cost of pre-training or fine-tuning new models?
<ul>	
	<li>More efficient pre-training solutions, from better initialization and hyper-parameter tuning to better optimization which lowers the cost of pre-training</li>
	<li>Parameter efficient fine-tuning  (PEFT) solutions for large pre-trained models</li>
	<li>Efficient instruction tuning,  prompt engineering and in-context learning</li>
	<li>Hardware-aware solutions (e.g. better CUDA kernels), memory read/write aware solutions </li>
	<li>Data-efficient training, reducing the requirement for labeled data, data compression and distillation</li>
</ul>
<b>Efficient Inference</b> How can we reduce the cost of inference for LLMs and foundation models?
<ul>
	<li>Improved speculative sampling for LLMs, self-speculative sampling, selecting among multiple drafts, one draft model for different heterogeneous target models</li>
	<li>Neural model compression techniques such as quantization, pruning, and knowledge distillation</li>
	<li>Improved KV-caching solutions for Transformers</li>
	<li>Distributed inference of large pre-trained models</li>
	<li>Serving many target devices with one model, many-in-one models, early exiting, elastic networks</li>
</ul>
<b>Evaluation and Benchmarking of Efficient Models</b> Introducing new efficient solutions underscores the need for comprehensive benchmarks to accurately evaluate their efficacy and performance. 
<ul>
	<li>Datasets, benchmarks, leaderboards for evaluating efficient models</li>
	<li>Benchmarking the performance of efficient models from different perspectives such as reasoning, hallucination,  understanding, and generation quality </li>
	<li>Benchmarking efficiency of models in terms of their memory footprint, training time, inference time, different target hardware devices and inference platforms (e.g. GPU vs. CPU) </li>
</ul>
<b>Efficient Solutions in other Modalities and Applications </b> 
<ul>
	<li> Efficiency of foundational or pre-trained models in multi-modal set-up and other modalities (beyond NLP and Speech) such as biology, chemistry, computer vision, and time series </li>
	<li>Efficient representations (e.g. Matryoshka representation) and models in dense retrieval and search</li>
	<li>Efficient Federated learning, lower communication costs, tackling heterogeneous data and models</li>
	<li>Efficient graph and LLM joint learning</li>
</ul>

</p>

<h2 class="blackpar_title">Submission Instructions</h2>
<p>
You are invited to submit your papers in our CMT submission portal <a href="https://cmt3.research.microsoft.com/ENLSP2024">(Link)</a>. All the submitted papers have to be anonymous for double-blind review. We expect each paper will be reviewed by at least three reviewers. The content of the paper (excluding the references and supplementary materials) should not be more than <b>8 pages for Long Papers</b> and <b>4 pages for Short Papers</b>, strictly following the NeurIPS template style <a href= "https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh">(Link)</a>. Please be advised that the NeurIPS submission checklist is not needed for our workshop submissions. 
<br />
Authors can submit up to 100 MB of supplementary materials separately. Authors are highly encouraged to submit their codes for reproducibility purposes. According to the guideline of the NeurIPS workshops, already published papers are not encouraged for submission, but you are allowed to submit your ArXiv papers or the ones which are under submission (for example <b> any NeurIPS submissions can be submitted concurrently to workshops </b>). Moreover, a work that is presented at the main NeurIPS conference should not appear in a workshop. Please make sure to indicate the complete list of conflict of interests for all the authors of your paper. To encourage higher quality submissions, our sponsors are offering the <b>Best Paper</b> and the <b>Best Poster</b> Awards to qualified outstanding original oral and poster presentations (upon nomination of the reviewers). Bear in mind that our workshop is not archival, but the accepted papers will be hosted on the workshop website. Moreover, we are currently negotiating with a publisher to host opt-in accepted papers in a special issue proceeding for our workshop.
</p>

<h2 class="blackpar_title">Important Dates:</h2>
<p>
<ul>
	<li>Submission Deadline: <b>August 22, 2025, Anywhere on Earth (AOE)</b></li>
	<li>Acceptance Notification: <b>September 22, 2025 AOE</b></li>
	<li>Camera-Ready Submission: <b>October 5, 2025 AOE</b> </li>
	<li>Workshop Date: <b>December 5th or 6th, 2025 </b></li>
</ul>
</p>


<!--Confirmed Speakers-->
<h2 class="blackpar_title" id="speakers">Keynote Speakers</h2>
<p>
{% include speakers.html %}
</p>

<h2 class="blackpar_title" id="speakers">Panelists</h2>
<p>
{% include panelists.html %}
</p>

<!-- Schedule -->
<h2 class="blackpar_title" id="schedule">Schedule</h2>
<p>
{% include schedule.html %}
</p>

<!-- Organizers -->
<h2 class="blackpar_title" id="organizers">Organizers</h2>
<p>
{% include organizers.html %}
</p>

<h2 class="blackpar_title" id="Organizers">Volunteers</h2>
<p>
{% include volunteers.html %}
</p>
<!-- <div class="row_perso">
	<div class="card_perso column_perso justify-content-center" id="volunteer_card">
	  <img src="/images/khalil_bibi.png" alt="Khalil Bibi" class="img_card_perso">
	  <div class="container_perso" >
		<center>
		<h6>
			<b>Khalil Bibi</b>
			<br>
			Huawei Noah's Ark Lab
			<br>
			<a href="https://scholar.google.ca/citations?user=feQAvxoAAAAJ&hl=en">
				<i class="bi bi-mortarboard-fill" style="font-size: 2rem;"></i>
			</a>
			&nbsp;
			<a href="https://www.linkedin.com/in/khalilbibi/">
				<i class="bi bi-linkedin" style="font-size: 2rem;"></i>
			</a>
		</h6>
		</center>
	  </div>
	</div>
	<div class="card_perso column_perso justify-content-center" id="volunteer_card">
	  <img src="/images/dav.jpg" alt="Khalil Bibi" class="img_card_perso">
	  <div class="container_perso" >
		<center>
		<h6>
			<b>David Alfonso-Hermelo</b>
			<br>
			Huawei Noah's Ark Lab
			<br>
			<a href="https://scholar.google.ca/citations?user=g6GccGAAAAAJ&hl=en&oi=ao">
				<i class="da da-mortarboard-fill" style="font-size: 2rem;"></i>
			</a>
			&nbsp;
			<a href="https://www.linkedin.com/in/david-alfonso-hermelo-6646a1b1/">
				<i class="bi bi-linkedin" style="font-size: 2rem;"></i>
			</a>
		</h6>
		</center>
	  </div>
	</div>
</div> -->


<br> 

<!-- Technical Committee -->
<h2 class="blackpar_title" id="technical_committee">Technical Committee</h2>
<p>
{% include technical_committee.html %}
</p>
<br>

<!-- <h2 class="blackpar_title">Diamond Sponsors</h2> -->
<!-- <center>
	<img src="/images/logos.png">	
	<img src="/images/BASF_logo.png">	
</center> -->



<!-- <h2 class="blackpar_title">Sponsors</h2>
<p>
We are currently welcoming sponsorship opportunities. If your organization is interested in supporting our conference, please contact us (neurips.ENLSP.2024@gmail.com) for more information on sponsorship packages and benefits. 
</p> -->

<h2 class="blackpar_title" id="sponsors"> Diamond Sponsors</h2>
<br>
<div class="row">
	<div class="col">
		<center>
			<img src="/images/Diamond.png" width="800px">
		</center>
	</div>
	<!-- <div class="col">
		<center>
			<img src="/images/BASF_logo.png" width="350px">
		</center>
	</div>	 -->
</div>
<br>
<h2 class="blackpar_title">Platinum Sponsor</h2>
<div class="row">
	<div class="col">
		<center>
			<img src="/images/Apple-Logo.jpg" width="250px">
		</center>
	</div>
</div>
<h2 class="blackpar_title">Gold Sponsors</h2>
<br>
<div class="row">
	<div class="col">
		<center>
			<img src="/images/shanghai_ai_lab1.png" width="200px">
		</center>
	</div>
	<div class="col">
		<center>
			<img src="/images/Logo-Sanofi.png" width="150px">
		</center>
	</div>
</div>
<br>
<div class="row">
	<div class="col">
		<center>
			<img src="/images/netmind_logo.png" width="400px">
		</center>
	</div>
	<div class="col">
		<center>
			<img src="/images/CUHK.png" width="400px">
	</center>
</div>
<br>


