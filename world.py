import mesa
from citizen import Citizen
from tqdm import tqdm
from datetime import datetime, timedelta
from utils import generate_names,generate_big5_traits, factorize, update_day, clear_cache
import random
import pickle


# functions for mesa.DataCollector in World class
def compute_num_susceptible(model):
    '''
    Computers number of Susceptible agents for data frame
    '''
    return sum([1 for a in model.schedule.agents if a.health_condition == "Susceptible"])


def compute_num_infected(model):
    '''
    Computers number of Infected agents for data frame
    '''
    return sum([1 for a in model.schedule.agents if a.health_condition == "Infected"])


def compute_num_recovered(model):
    '''
    Computers number of Recovered agents for data frame
    '''
    return sum([1 for a in model.schedule.agents if a.health_condition == "Recovered"])


def compute_num_on_grid(model):
    '''
    Computers number of agents on the grid
    '''
    return sum([1 for a in model.schedule.agents if a.location == "grid"])


def compute_num_at_home(model):
    '''
    Computers number of agents at home
    '''
    return sum([1 for a in model.schedule.agents if a.location == "home"])


class World(mesa.Model):
    '''
    The world where Citizens roam
    '''
    def __init__(self, args, initial_healthy=2, initial_infected=1, contact_rate=5):
        
        ########################################
        #     Intialization of the world       #
        ########################################
    
        #Agent Initialization
        self.initial_healthy=initial_healthy
        self.initial_infected=initial_infected
        self.population=initial_healthy+initial_infected
        self.step_count = args.no_days
        self.offset = 0 #Offset for checkpoint load
        self.name = args.name

        #World Creation Initialization
        world_dimensions=factorize(self.population)
        self.height=world_dimensions[0]
        self.width=world_dimensions[1]
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self.current_date = datetime(2015, 3, 3)

        #Important infection variables
        self.total_contact_rates = 0
        self.track_contact_rate = [0]
        self.day_infected_is_4 = [0]
        self.list_new_cases = [0] 
        self.daily_new_cases = initial_infected
        self.infected = initial_infected
        self.contact_rate= args.contact_rate
        self.infection_rate = args.infection_rate
        self.agents_on_grid=[]
        self.max_potential_interactions=0
        #Initialize Schedule
        self.schedule = mesa.time.RandomActivation(self)


        
        #Initiate data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Susceptible": compute_num_susceptible,
                            "Infected": compute_num_infected,
                            "Recovered": compute_num_recovered,
                            "# Home": compute_num_at_home,
                            "# Grid":compute_num_on_grid
                            })
        

        ########################################
        #Assigning properties to all 100 agents#
        ########################################

        #IDs for agents
        agent_id = 0 

        #generates list of random names out of the 200 most common names in the US
        names = generate_names(self.population, self.population*2)
        traits = generate_big5_traits(self.population)

        #for loop to initialize each agents
        for i in range(self.population):

            #Assigns all agents to grid first
            location="grid"

            #Creates healthy agents
            if i+1<=(self.initial_healthy):
                health_condition="Susceptible"
                day_infected= None

            #Creates infected, unhealthy agent(s)
            else:
                health_condition = "Infected"
                day_infected = 1
        
            #create instances of the Citizen class
    
            citizen = Citizen(model=self,
                            unique_id=agent_id,name=names[i],age=random.randrange(18,65),
                            traits=traits[i],
                            location=location ,pos=None,
                            health_condition=health_condition, day_infected=day_infected,
                            width=self.width, height=self.height
                            )
            # add agents to the scheduler
            self.schedule.add(citizen)
            # Updates to new agent ID
            agent_id += 1 

        self.distribute_agents() #distributes agents in the grid world

    def distribute_agents(self):
        grid_size = (self.width,self.height)

        for idx, agent in enumerate(self.schedule.agents):
            # convert idx to 2D coordinates
            row = idx // grid_size[1]
            col = idx % grid_size[1]
            self.grid.place_agent(agent,(row,col)) #places each agent into the world

    def decide_agent_interactions(self):
        '''
        Decides interaction partners for each agent
        '''
        self.max_potential_interactions = min(self.contact_rate, len(self.agents_on_grid) - 1)
        random.shuffle(self.agents_on_grid)
        for agent in self.agents_on_grid:
            potential_interactions = [a for a in self.agents_on_grid if a is not agent and a not in agent.agent_interaction]
        
        #Not all agents will have 5 contacts as it is slightly random the order.
            while len(agent.agent_interaction) < self.max_potential_interactions and potential_interactions:
                other_agent = random.choice(potential_interactions)
                agent.add_agent_interaction(other_agent)
                potential_interactions.remove(other_agent)

    def step(self):
        '''
        Model Time step
        '''
        
        for agent in self.schedule.agents: #Cycle through each agent's substep
            agent.prepare_step()
        self.decide_agent_interactions()
       
        for agent in self.schedule.agents: #track global contact rate
            self.total_contact_rates += len(agent.agent_interaction)
        self.track_contact_rate.append(self.total_contact_rates)
        self.total_contact_rates = 0

        # call the step function of every agent
        self.schedule.step()

        #Update day of each agent
        for agent in self.schedule.agents:
            update_day(agent)
        
        #track how many agents have been infected for 4 days for case feedback
        self.day_infected_is_4.append(sum([True if agent.day_infected==4 else False for agent in self.schedule.agents]))          

    #Function to actually run the model
    def run_model(self, checkpoint_path, offset=0):
        self.offset = offset
        end_program=0
        for i in tqdm(range(self.offset,self.step_count)):
            #collect model level data
            self.datacollector.collect(self)

            #Model steps
            self.step()

            #collect all new cases from one day
            self.list_new_cases.append(self.daily_new_cases)
            #set daily new case to 0 again
            self.daily_new_cases = 0

            #Print statements
            print(f"At the end of {self.current_date.date()}")
            print(f"Total Pop: {self.population}\tNew Cases: {self.list_new_cases}")
            print (f"Currently Infected: {self.infected}")
            print(f"Agent Perspective New cases: {self.day_infected_is_4}")

            """
            early stopping condition: if there are no more infected agents left, 
            run for three more time steps, save the model and then end program
            """
            if self.infected==0:
                end_program+=1
            if end_program == 2:
                path = checkpoint_path + f"/{self.name}-final_early.pkl"
                self.save_checkpoint(file_path = path)
                break

            self.current_date += timedelta(days=1)
            path = checkpoint_path+f"/{self.name}-{i+1}.pkl"
            self.save_checkpoint(file_path = path)
            clear_cache()


    #saves checkpoint to specified file path
    def save_checkpoint(self, file_path):
        with open(file_path,"wb") as file:
            pickle.dump(self, file)
    
    '''
    A static method in Python is a function within a class that is bound to the class rather than an instance, 
    does not receive any implicit arguments, and can be called on the class itself without creating an instance
    '''
    @staticmethod
    def load_checkpoint(file_path):
        with open(file_path,"rb") as file:
            return pickle.load(file)
