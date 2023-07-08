from names_dataset import NameDataset
import numpy as np
import random
import openai
import time
import os
import shutil


def probability_threshold(threshold):
    '''
    Used in self.infect_interaction()
    '''
    #Generates random number from 0 to 1
    
    return (np.random.rand()<threshold)

def generate_names(n: int, s: int, country_alpha2='US'):
    '''
    Returns random names as names for agents from top names in the USA
    Used in World.init to initialize agents
    '''

    # This function will randomly selct n names (n/2 male and n/2 female) without
    # replacement from the s most popular names in the country defined by country_alpha2
    if n % 2 == 1:
        n += 1
    if s % 2 == 1:
        s += 1

    nd = NameDataset()
    male_names = nd.get_top_names(s//2, 'Male', country_alpha2)[country_alpha2]['M']
    female_names = nd.get_top_names(s//2, 'Female', country_alpha2)[country_alpha2]['F']
    if s < n:
        raise ValueError(f"Cannot generate {n} unique names from a list of {s} names.")
    # generate names without repetition
    names = random.sample(male_names, k=n//2) + random.sample(female_names, k=n//2)
    del male_names
    del female_names
    random.shuffle(names)
    return names


def generate_big5_traits(n: int):
    '''
    Return big 5 traits for each agent
    Used in World.init to initialize agents
    '''

    #Trait generation
    agreeableness_pos=['Cooperation','Amiability','Empathy','Leniency','Courtesy','Generosity','Flexibility',
                        'Modesty','Morality','Warmth','Earthiness','Naturalness']
    agreeableness_neg=['Belligerence','Overcriticalness','Bossiness','Rudeness','Cruelty','Pomposity','Irritability',
                        'Conceit','Stubbornness','Distrust','Selfishness','Callousness']
    #Did not use Surliness, Cunning, Predjudice,Unfriendliness,Volatility, Stinginess

    conscientiousness_pos=['Organization','Efficiency','Dependability','Precision','Persistence','Caution','Punctuality',
                            'Punctuality','Decisiveness','Dignity']
    #Did not use Predictability, Thrift, Conventionality, Logic
    conscientiousness_neg=['Disorganization','Negligence','Inconsistency','Forgetfulness','Recklessness','Aimlessness',
                            'Sloth','Indecisiveness','Frivolity','Nonconformity']

    surgency_pos=['Spirit','Gregariousness','Playfulness','Expressiveness','Spontaneity','Optimism','Candor'] 
    #Did not use Humor, Self-esteem, Courage, Animation, Assertion, Talkativeness, Energy level, Unrestraint
    surgency_neg=['Pessimism','Lethargy','Passivity','Unaggressiveness','Inhibition','Reserve','Aloofness'] 
    #Did not use Shyness, Silenece

    emotional_stability_pos=['Placidity','Independence']
    emotional_stability_neg=['Insecurity','Emotionality'] 
    #Did not use Fear, Instability, Envy, Gullibility, Intrusiveness
    
    intellect_pos=['Intellectuality','Depth','Insight','Intelligence'] 
    #Did not use Creativity, Curiousity, Sophistication
    intellect_neg=['Shallowness','Unimaginativeness','Imperceptiveness','Stupidity']


    #Combine each trait
    agreeableness_tot = agreeableness_pos + agreeableness_neg
    conscientiousness_tot = conscientiousness_pos + conscientiousness_neg
    surgency_tot = surgency_pos + surgency_neg
    emotional_stability_tot = emotional_stability_pos + emotional_stability_neg
    intellect_tot = intellect_pos + intellect_neg

    #create traits list to be returned
    traits_list = []

    for _ in range(n):
        agreeableness_rand = random.choice(agreeableness_tot)
        conscientiousness_rand = random.choice(conscientiousness_tot)
        surgency_rand = random.choice(surgency_tot)
        emotional_stability_rand = random.choice(emotional_stability_tot)
        intellect_rand = random.choice(intellect_tot)

        selected_traits=[agreeableness_rand,conscientiousness_rand,surgency_rand,
                                emotional_stability_rand,intellect_rand]

        traits_chosen = (', '.join(selected_traits))
        traits_list.append(traits_chosen)
    del agreeableness_rand
    del conscientiousness_rand
    del surgency_rand
    del emotional_stability_rand
    del intellect_rand
    del selected_traits
    del traits_chosen
    return traits_list


def update_day(agent):
    '''
    Update day funtion to update day_sick
    Used in World.step()
    '''
    # Number of days to heal
    Time_to_heal = 6
    # print("Agent ID: {} Day infected: {}".format(agent.unique_id,agent.day_infected))


    #if person is healthy, no reason to update health status
    if agent.health_condition=="Susceptible" or agent.health_condition=="Recovered":
        return
    
    if agent.health_condition=="To_Be_Infected":
        agent.health_condition="Infected"
        agent.day_infected = 0 
        agent.model.daily_new_cases +=1 #every time new infection occurs in a day, counter is updated
        agent.model.infected += 1 #Update amount infected at any given time
    
    #Update Infected status day_infected counter
    agent.day_infected+=1

    #The agent healed after 6 days,
    if agent.day_infected>Time_to_heal:
        agent.day_infected= None
        agent.health_condition="Recovered"
        agent.model.infected += -1 #Update amount infected at any given time


def factorize(n):
    '''
    Factorize number for ideal grid dimensions for # of agents
    Used in World.init
    '''
    for i in range(int(n**0.5), 1, -1):
        if n % i == 0:
            return (i, n // i)
    return (n, 1)

def get_completion_from_messages(messages, model="gpt-3.5-turbo-0301", temperature=0):
    success = False
    retry = 0
    max_retries = 30
    while retry< max_retries and not success:
      try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
            )
        success = True
      except Exception as e:
        print(f"Error: {e}\nRetrying...")
        retry+=1
        time.sleep(0.5)

    return response.choices[0].message["content"]

def clear_cache():
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")