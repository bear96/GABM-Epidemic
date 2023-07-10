# GABM-SIR
A new modeling technique using generative AI applied to an epidemic to incorporate human reasoning and decision making.

## Installation
You will need an OpenAI API key to run this program. Please create an OpenAI API key before following the steps below.
If you do not have an API key, here is a link to create an API key: https://platform.openai.com/account/api-keys

Step 1: Clone the repository using `git clone https://github.com/bear96/GABM.git`. <br>
Step 2: Install the required packages using `pip install -r requirements.txt` <br>
Step 3: In main.py, replace the placeholder text with your OpenAI API key. Now you can replicate our results by running `python main.py --name GABM`. You can check all the available hyperparameters that you can change in detail by running `python main.py --help`. <br>
Currently, the default values of the hyperparameters are: <br>
* name: GABM
* contact_rate: 5
* infection_rate: 0.1
* no_init_healthy: 98
* no_init_infect: 2
* no_days: 50
* time_to_heal: 6
* no_of_runs: 1
* offset: 0
* load_from_run: 0

## Model Overview
We present an innovative approach to incorporate human behavior into epidemic models by combining generative artificial intelligence (AI) with epidemic modeling. Our approach involves the development of a generative agent-based model (GABM) that utilizes GPT-3.5 to create agents with realistic personas. These agents possess the ability to reason, make decisions, and adapt their behavior in response to the evolving epidemic, taking into account individual characteristics, virus information, perceived health, and infection risks. Through extensive simulation experiments, we demonstrate that the GABM accurately replicates real-world conditions, generating patterns that closely resemble observed pandemic waves and endemic periods. By integrating generative AI into epidemic modeling, our approach enables a comprehensive representation of complex human behavior dynamics, leading to improved accuracy in projections and more informed policy decisions. <br>
<br>
Link to arXiv paper:
<br>
Please create an issue if something isn't working for you. We'll be happy to help.
