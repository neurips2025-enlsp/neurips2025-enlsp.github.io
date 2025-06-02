---
#
# By default, content added below the "---" mark will appear in the home page
# between the top bar and the list of recent posts.
# To change the home page layout, edit the _layouts/home.html file.
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
#
layout: home
---
<img src="/images/banner2025_v3.png" style="pointer-events: none; user-select: none;">

<!--
<div style="position: relative; width: 100%;">
  <img src="/images/banner2025_v2.png" style="width: 100%; display: block; pointer-events: none; user-select: none;">
  
  <div style="
    position: absolute;
    top: 10%;
    left: 50%;
    transform: translateX(-50%);
    text-align: center;
    color: white;
    text-shadow: 2px 2px 8px #000;
    width: 80%;
    max-width: 1600px;
  ">
    <div style="font-size: 1.6vw; font-weight: 500; white-space: nowrap;">
      The Fifth Workshop on Efficient Natural Language and Speech Processing
    </div>
    <br>
    <div style="font-size: 3vw; font-weight: bold; margin-top: 0.5vw;">
      The Art of Smart Thinking:
    </div>
    <div style="font-size: 3vw; font-weight: bold; white-space: nowrap;">
      Efficient Reasoning & Test-Time Compute
    </div>
  </div>
</div>
-->




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
	ENLSP 2025 Mentorship Session I
Tuesday, July 15 · 7:00 – 8:00am
Time zone: America/Toronto
Google Meet joining info
Video call link: https://meet.google.com/pmz-ozpk-woj
Calendar Invite (add to your calendar): https://calendar.app.google/mTFsmwKUgPmXLaFYA  (July 29th- https://calendar.app.google/oJtevifEwr9ZXDBD8) (Aug. 12, https://calendar.app.google/TMCELHHkTNC5aVCu6)
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
Building on the momentum of previous editions, this year’s workshop continues to advance the efficiency of language and speech models, while introducing several timely and impactful directions to address emerging challenges and broaden community engagement: <strong>(1) Efficient Architectures:</strong> We spotlight architectures that balance performance with computational and memory efficiency. While Transformers dominate, their quadratic attention and KV cache overhead limit scalability. New directions include sub-quadratic models (e.g., <a href="https://arxiv.org/pdf/2312.00752">Mamba</a>, <a href="https://arxiv.org/abs/2402.04347">Hedgehog</a>, <a href="https://arxiv.org/abs/2405.04517">xLSTM</a>, and <a href="https://arxiv.org/abs/2305.13048">RVKW</a>), hybrid designs (e.g., <a href="https://arxiv.org/abs/2403.19887">Jamba</a>, <a href="https://arxiv.org/abs/2504.03624">Nemotron-H</a>), and diffusion-based language models <a href="https://arxiv.org/abs/2502.09992">LLaDA</a>, <a href="https://arxiv.org/abs/2505.16839">Lavida</a>. <strong>(2) Custom Model Composition:</strong> As applications diversify, we need modular, efficient frameworks that adapt pre-trained models without full retraining. This enables customization across hardware and user needs, beyond what compression or NAS alone can offer <a href="https://arxiv.org/abs/2407.14679">[Link1]</a>, <a href="https://arxiv.org/abs/2411.19146">[Link2]</a>, <a href="https://arxiv.org/abs/2505.00949">[Link3]</a>. <strong>(3) Efficient Training and Knowledge Transfer:</strong> Efficient models often rely on expensive training (e.g., <a href="https://arxiv.org/abs/2406.07522">Samba</a> and <a href="https://arxiv.org/abs/2411.13676">Hymba</a>, which involve pre-training) or suffer from poor transfer (e.g., <a href="https://arxiv.org/abs/2408.15237">MambaInLLaMA</a>). We invite solutions for effective post-training adaptation, and KV compression that preserve model quality at lower cost. <strong>(4) Efficient Reasoning Models:</strong> Models like OpenAI’s O1 and DeepSeek show promise in reasoning, but their pipelines (training regimes, data curation, and design decisions) remain opaque. We encourage work that improves transparency and efficiency of reasoning models across training, data, inference, and deployment. <strong>(5) Adaptive Test-Time Compute:</strong> Reasoning often benefits from increased inference compute. We seek methods like speculative decoding (e.g., multi-branch CoT) and adaptive compute allocation to extend “thinking time” based on task complexity without incurring quadratic costs. <strong>(6) Evaluation and Benchmarking:</strong> Robust, compute-aware evaluation is essential. We seek evaluation methodologies that incorporate compute-aware metrics—such as latency-quality trade-offs, and energy efficiency—to assess models under practical deployment conditions, including edge and low-power scenarios.
</p>


<!-- Call for Papers -->
<h2 class="blackpar_title" id="call_for_papers">Call for Papers</h2>
<p>
Advancing foundation models requires rethinking efficiency—especially in reasoning and test-time compute. We warmly welcome <strong>all</strong> submissions and contributions, including papers and demos, that address these challenges across the full model lifecycle: architecture, training, inference, and evaluation. Topics include, but are not limited to:
<br><br>
<b>Developing and Deploying Efficient Reasoning Language and Speech Models</b> 
<ul>
	<li>Training strategies for cost-efficient reasoning (RL vs. supervised fine-tuning trade-offs)</li>
	<li>Developing new reasoning language and speech models using knowledge transfer</li>
	<li>Curriculum learning and data-efficient reasoning</li>
	<li>Enhanced speech-text multimodal reasoning, aligning speech and text reasoning chains</li>
	<li>Architecture design for reasoning efficiency of language and speech modalities</li>
</ul>
<b>Efficiency in Test-time Compute</b> 
<ul>
	<li>Designing lightweight and adaptive test-time inference strategies for language and speech models</li>
	<li>Combining test-time compute and inference optimizations (e.g., quantization & speculative decoding)</li>
	<li>Test-time compute in heterogeneous and resource-constrained environments (e.g., edge devices)</li>
	<li>Test-time scaling law vs. training scaling law</li>
	<li>Efficient test-time compute solutions for language and speech reasoning models</li>
</ul>
<b>Designing Efficient Models and Architectures</b> 
<ul>
	<li>Non-autoregressive models such as diffusion models to improve generation speed and scalability</li>
	<li>Designing hybrid models and their efficient training</li>
	<li>Proposing new architectures with a focus on scalability, efficiency, and performance</li>
	<li>Long-sequence modeling through memory-efficient mechanisms and architectural innovations</li>
	<li>Dense vs. sparse architectures (MoEs)</li>
</ul>
<b>Efficient Training and Training Efficient Models</b> 
<ul>
	<li>Trade-offs between pre-training, post-training, and model upcycling to reduce overall training cost</li>
	<li>Efficient pre-training solutions</li>
	<li>Parameter-efficient fine-tuning (PEFT) tailored for hybrid models and emerging architectures</li>
	<li>Enhancing instruction tuning, prompt engineering, and in-context learning</li>
	<li>Data efficiency through data selection, filtering, and training strategies tailored for new models</li>
</ul>
<b>Efficient Inference Solutions</b> 
<ul>
	<li>Developing inference-optimized methods for new and non-transformer architectures</li>
	<li>Accelerating speculative decoding (SD), batched SD, and extensions to hybrid and emerging models</li>
	<li>Neural model compression techniques such as quantization, pruning, and knowledge distillation</li>
	<li>Efficient inference solutions for new architectures</li>
	<li>Many-in-one and dynamic solutions to serve diverse deployment targets efficiently</li>
</ul>
<b>Evaluation and Benchmarking</b> 
<ul>
	<li>Comprehensive evaluation of models using both efficiency metrics and performance indicators</li>
	<li>Benchmarking new architectures, reasoning models, and test-time compute techniques</li>
	<li>Datasets, benchmarks, and leaderboards for evaluating efficient reasoning language and speech models</li>
</ul>
 <span class="news-item-icon">📢</span> <b>Special Demo Track: Efficient Reasoning and Test-time Compute</b> This newly introduced track invites demo submissions that showcase working systems, tools, visualizations, or APIs addressing any of the topics above related to efficient reasoning and test-time compute in NLP & Speech.
</p>

<h2 class="blackpar_title">Mentorship Sessions</h2>
<p>
To support new researchers and underrepresented groups, we will host virtual mentorship sessions prior to the submission deadline. These sessions will provide guidance on
preparing high-quality submissions, addressing questions, and engaging with the research community.
</p>

<div style="display: flex; gap: 20px; flex-wrap: wrap; font-family: Arial, sans-serif; max-width: 100%;">
  <!-- Session I -->
  <div style="flex: 1; min-width: 300px; border: 1px solid #ccc; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
    <h3 style="color: #003366; margin-top: 0;">🕖 Session A – Global East (Best for participants in Asia, Europe, Africa, Australia)</h3>
    <p><strong>Time:</strong> 7:00 – 8:00 AM Eastern Time (Toronto)</p>
    <p><a href="https://www.timeanddate.com/worldclock/fixedtime.html?msg=ENLSP+Mentorship+Session+A&iso=20250715T07&p1=250" style="color: #1a73e8; text-decoration: none;"> <strong>➤ Convert to your local time</strong></a></p>
    

    <h3 style="margin-bottom: 5px;">Google Meet Joining Info</h3>
    <p>
      <a href="https://meet.google.com/pmz-ozpk-woj" style="color: #1a73e8; text-decoration: none;">
        🔗 Join the Video Call
      </a>
    </p>

    <h3 style="margin-bottom: 5px;">📅 Add to Your Calendar</h3>
    <ul style="padding-left: 20px; margin-top: 0;">
      <li><a href="https://calendar.app.google/mTFsmwKUgPmXLaFYA" style="color: #1a73e8;">Mentorship Day 1: July 15, 2025</a></li>
      <li><a href="https://calendar.app.google/oJtevifEwr9ZXDBD8" style="color: #1a73e8;">Mentorship Day 2: July 29, 2025</a></li>
      <li><a href="https://calendar.app.google/TMCELHHkTNC5aVCu6" style="color: #1a73e8;">Mentorship Day 3: August 12, 2025</a></li>
    </ul>
  </div>

  <!-- Session II -->
  <div style="flex: 1; min-width: 300px; border: 1px solid #ccc; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
    <h3 style="color: #003366; margin-top: 0;">🕛 Session B – Americas (Best for participants in North & South America)</h3>
    <p><strong>Time:</strong> 12:00 – 1:00 PM Eastern Time (Toronto)</p>
    <p><a href="https://www.timeanddate.com/worldclock/fixedtime.html?msg=ENLSP+Mentorship+Session+B&iso=20250715T12&p1=250" style="color: #1a73e8; text-decoration: none;"> <strong>➤ Convert to your local time</strong></a></p>

    <h3 style="margin-bottom: 5px;">Google Meet Joining Info</h3>
    <p>
      <a href="https://meet.google.com/cma-wywf-yip" style="color: #1a73e8; text-decoration: none;">
        🔗 Join the Video Call
      </a>
    </p>

    <h3 style="margin-bottom: 5px;">📅 Add to Your Calendar</h3>
    <ul style="padding-left: 20px; margin-top: 0;">
      <li><a href="https://calendar.app.google/xbu8nMd1nvPAi5Qi8" style="color: #1a73e8;">Mentorship Day 1: July 15, 2025</a></li>
      <li><a href="https://calendar.app.google/p6B8avKYmjvBZX8D6" style="color: #1a73e8;">Mentorship Day 2: July 29, 2025</a></li>
      <li><a href="https://calendar.app.google/hUTV96JJ7wMRgjC28" style="color: #1a73e8;">Mentorship Day 3: August 12, 2025</a></li>
    </ul>
  </div>
</div>

<br /> 
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
<h2 class="blackpar_title" id="speakers">Confirmed Keynote Speakers</h2>
<p>
{% include speakers.html %}
</p>

<h2 class="blackpar_title" id="speakers">Confirmed Panelists</h2>
<p>
{% include panelists.html %}
</p>

<!-- Schedule -->
<h2 class="blackpar_title" id="schedule">Tentative Schedule</h2>
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
<h2 class="blackpar_title" id="technical_committee">Confirmed Technical Committee</h2>
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

<h2 class="blackpar_title" id="sponsors"> Sponsorship Prospects</h2>
<br>
<div class="row">
	<div class="col">
		<center>
			<img src="/images/Huawei2.png" width="150px">
		</center>
	</div>
    <div class="col">
		<center>
			<img src="/images/Apple-Logo.jpg" width="250px">
		</center>
	</div>
	<div class="col">
		<center>
			<img src="/images/Logo-Sanofi.png" width="200px">
		</center>
	</div>
	<div class="col">
		<center>
			<img src="/images/AMD.png" width="200px">
		</center>
	</div>
</div>
<br>


