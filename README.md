# GABM-SIR
This study offers a new paradigm of individual-level modeling to address the grand challenge of incorporating human behavior in epidemic models. By utilizing generative artificial intelligence in an agent-based epidemic model, each agent is empowered to make its own reasonings and decisions via connecting to a large language model like ChatGPT. Through various simulation experiments, we present compelling evidence that generative agents mimic real-world behaviors such as quarantining when sick, and self-isolation when cases rise. Collectively, the agents demonstrate patterns akin to multiple waves observed in recent pandemics followed by a period of endemic. Moreover, the agents successfully flatten the epidemic curve. This study opens up a potential to improve dynamic system modeling by offering a way to represent human brain, reasoning, and decision making.

## Installation
You will need an OpenAI API key to run this program. Please create an OpenAI API key before following the steps below.

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

Please create an issue if something isn't working for you. We'll be happy to help.
