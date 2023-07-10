import time
import mesa
from utils import get_completion_from_messages, probability_threshold
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class Citizen(mesa.Agent):
    '''
    Define who a citizen is:
    unique_id: assigns ID to agent
    name: name of the agent
    age: age of the agent
    traits: big 5 traits of the agent
    location: flag for staying at home or going on grid
    pos: position (x,y) tuple for grid position
    health_condition: flag to say if Susceptible or Infected or Recovered
    day_infected: agent attribute to count the number of days agent spends infected
    width, height: dimensions of world
    '''

    def __init__(self,model,unique_id, name, age, traits,location,pos,health_condition, day_infected, width, height):
                
        super().__init__(unique_id,model) #Inherit mesa.Agent class attributes (model is mesa.Model)
        #Persona
        self.name = name
        self.age = age
        self.location=location
        self.pos=pos
        self.traits=traits

        #Health Initialization of Agent
        self.health_condition=health_condition
        self.day_infected=day_infected

        #Contact Rate
        self.agent_interaction=[]

        #Reasoning tracking
        self.mems = {"name":name,"age":age,"traits":traits}

    #########################################
    #             Health Feedback           #
    #########################################  
    def get_health_string(self):
        Health_strings=[f"{self.name} feels normal.",
                        f"{self.name} has a light cough.",
                        f"{self.name} has a fever and a cough.",
                        ]

        if self.health_condition=="Susceptible" or self.health_condition=="Recovered" or self.health_condition=="To_Be_Infected" or self.day_infected<=2:
            return Health_strings[0]
        
        if self.day_infected==3:
            return Health_strings[1]
        
        if self.day_infected==4 or self.day_infected==5:
            return Health_strings[2]

        if self.day_infected==6:
            return Health_strings[1]
        
        
    ########################################
    #      Decision-helper functions       #
    ########################################

    def ask_agent_stay_at_home(self):
        '''
        Used in self.decide_location. Returns True or False depending on whether agent wants to
        stay at home.
        '''
        reasoning, response = self.get_response_and_reasoning()
        self.mems[self.model.schedule.steps] = {"health condition":self.health_condition,"reasoning": reasoning,
        "response": response,"health string": self.get_health_string(),"location":self.location}
        response = response.lower()
        if reasoning is None:
            reasoning = f"{self.name} did not give a reason."
            logger.warning("Reasoning was none-type.")

        if "no" in response:
            return False
        elif "yes" in response:
            return True
        else:
            logger.warning(f"Response was something unexpected. Defaulting with assuming agent decided to not stay at home.\nResponse was '{response}'")
            return False
        del response
        del reasoning


        

    def get_response_and_reasoning(self):
        '''
        GAI of model. Uses ChatGPT to provide response and reasoning to the prompt provided.
        '''
        question_prompt = f"""
        You are {self.name}. You are {self.age} years old. 
       
        Your traits are given below:
        {self.traits}
        
        Your basic bio is below:
        {self.name} lives in the town of Dewberry Hollow. {self.name} likes the town and has friends who also live there. {self.name} has a job and goes to the office for work everyday.
        
        I will provide {self.name}'s relevant memories here:
        {self.get_health_string()}
        {self.name} knows about the Catasat virus spreading across the country. It is an infectious disease that spreads from human to human contact via an airborne virus. The deadliness of the virus is unknown. Scientists are warning about a potential epidemic.
        {self.name} checks the newspaper and finds that {(self.model.day_infected_is_4[self.model.schedule.steps]*100)/self.model.population: .1f}% of Dewberry Hollow's population caught new infections of the Catasat virus yesterday.
        {self.name} goes to work to earn money to support {self.name}'s self.
       
        Based on the provided memories, should {self.name} stay at home for the entire day? Please provide your reasoning.

        If the answer is "Yes," please state your reasoning as "Reasoning: [explanation]." 
        If the answer is "No," please state your reasoning as "Reasoning: [explanation]."
        
        The format should be as follow:
        Reasoning:
        Response:

        Example response format:

        Reasoning: {self.name} is tired.
        Response: Yes

        It is important to provide Response in a single word.
        """
       
        messages =  [{'role':'system', 'content':question_prompt}]
        try:
            output = get_completion_from_messages(messages, temperature=0)
        except Exception as e:
            logger.warning(f"{e}\nProgram paused. Retrying after 60s...")
            time.sleep(60)
            output = get_completion_from_messages(messages, temperature=0)
        reasoning = ""
        response = ""
        try:
            intermediate  = output.split("Reasoning:",1)[1]
            reasoning, response = intermediate.split("Response:")
            response = response.strip().split(".",1)[0]
            reasoning = reasoning.strip()
            # print(reasoning, response)
        except:
            logger.warning("Reasoning or response were not parsed correctly.")
            response = "No"
            reasoning = None
        del question_prompt
        return reasoning, response
        

    ########################################
    #    Location-decision functions       #
    ########################################

    def decide_location(self):
        '''
        Agents decide whether they want to go outside on the grid or stay home.
        According to their decision, their location is updated.
        '''
        response=self.ask_agent_stay_at_home()
        
        #If agent wants to stay home
        if response is True:
            self.location="home"
        
            #Removing from agents_on_grid list in the world
            if self in self.model.agents_on_grid:
                self.model.agents_on_grid.remove(self)

        #If agent wants to go out on grid
        else:
            self.location="grid"
        
            #Adding to list of agents_on_grid list in the world
            if self not in self.model.agents_on_grid:
                self.model.agents_on_grid.append(self)

        del response


    ################################################################################
    #                       Meet_interact_infect functions                         #
    ################################################################################ 
    def add_agent_interaction(self, agent):
        '''
        Called in self.model.decide_agent_interactions()
        '''
        contact_rate = self.model.max_potential_interactions ## Do not add more than contact rate
        if len(self.agent_interaction) >= contact_rate or len(agent.agent_interaction) >= contact_rate:
            return

        self.agent_interaction.append(agent)
        agent.agent_interaction.append(self)
    ########################################
    #                 Interact             #
    ########################################

    def interact(self):
        '''
        Step 1. Run infection for each agent_interaction
        Step 2. Reset agent_interaction for next day
        Used in self.step()
        '''

        for agent in self.agent_interaction:
            self.infect(agent)
        #Reset Agent Interaction list
        self.agent_interaction=[]

    ########################################
    #               Infect                 #
    ########################################
    def infect(self, other):
        '''
        Step 1. See health status of both members.
        Step 2. If one is infected, roll for threshold

        Used in self.interact()
        '''
        
        #sets infection threshold
        infection_rate = self.model.infection_rate


        #if self is sick and other is not
        if self.health_condition=="Infected":

            #See if there is a chance they get infected
            if probability_threshold(infection_rate) and other.health_condition=="Susceptible":

                #Other is infected
                other.health_condition="To_Be_Infected"

        #if other is sick and self is not
        elif other.health_condition=="Infected":

            #See if there is a chance they get infected
            if probability_threshold(infection_rate) and self.health_condition == "Susceptible":

                #Self is infected
                self.health_condition="To_Be_Infected"

    ################################################################################
    #                              step functions                                  #
    ################################################################################
    def prepare_step(self):
        '''
        Make all agents decide on their location before the step functions
        '''
        self.decide_location()
  

    def step(self):
        '''
        Step function for agent
        '''
        self.interact()
