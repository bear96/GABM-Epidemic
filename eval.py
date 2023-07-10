from world import World
from names_dataset import NameDataset
import ast
import numpy as np
import pandas as pd

#Load any checkpoint
model=World.load_checkpoint(r"checkpoint\run-1\GABM_1000_R03_individual_data-completed.pkl")

mems = []
#Get the responses and other relevant attributes of the agents over time
for agent in model.schedule.agents:
    mems.append(agent.mems)
responses_over_time = pd.DataFrame(mems)
#save responses over time as a .csv file
responses_over_time.to_csv("responses_over_time.csv")

#Get statistical data from the datacollector of the world
data = model.datacollector.get_model_vars_dataframe()
df = pd.DataFrame(data)
new_infections_newspaper=model.list_new_cases[:-1]
new_infections_newspaper[0]=model.list_new_cases[0]+model.initial_infected
new_infections_newspaper[1]=model.list_new_cases[1]-model.initial_infected
df['New Infections']=new_infections_newspaper
df['Cumulative Infections'] = df['New Infections'].cumsum()
df['Total Contact'] = model.track_contact_rate[:len(df)]
df["Daily New Cases Day 4"] = model.day_infected_is_4[:len(df)]
#Save statistical data as a .csv file
df.to_csv("stats_for_agents.csv")

nd = NameDataset()
df_full = pd.DataFrame()

file_indiv = "responses_over_time.csv" #response file dir
file_run = "stats_for_agents.csv" #stats file dir

# You can now use these file paths to read the .csv files:
data_run = pd.read_csv(file_run)
data_indiv = pd.read_csv(file_indiv)

#####################################################
#                     Dynamic Data                  #
#####################################################
data_timestep=data_indiv.drop(['name','age','traits', 'Unnamed: 0'], axis=1)

# Initialize your dataframes here
health_condition_df = pd.DataFrame()
reasoning_df = pd.DataFrame()
response_df = pd.DataFrame()
health_string_df = pd.DataFrame()
location_df = pd.DataFrame()

for col in data_timestep.columns:

  data_timestep[col] = data_timestep[col].apply(ast.literal_eval)
  data_df = pd.json_normalize(data_timestep[col])

  # Create separate dataframes

  health_condition_df[col] = pd.DataFrame(data_df['health condition'])
  reasoning_df[col] = pd.DataFrame(data_df['reasoning'])
  response_df[col] = pd.DataFrame(data_df['response'])
  health_string_df[col] = pd.DataFrame(data_df['health string'])
  location_df[col] = pd.DataFrame(data_df['location'])

df_response = response_df.replace({'Yes': 1, 'No': 0})
df_response = df_response.where(df_response.isin([0, 1]), 0)

#Find # of agents
num_agents=df_response.shape[0]

#####################################################
#                        Traits                     #
#####################################################
traits_pos = {
'agreeableness':['Cooperation','Amiability','Empathy','Leniency','Courtesy','Generosity','Flexibility',
                    'Modesty','Morality','Warmth','Earthiness','Naturalness'],
'conscientiousness':['Organization','Efficiency','Dependability','Precision','Persistence','Caution','Punctuality',
                    'Punctuality','Decisiveness','Dignity'],
'surgency':['Spirit','Gregariousness','Playfulness','Expressiveness','Spontaneity','Optimism','Candor'],
'emotional_stability': ['Placidity','Independence'],
'intellect': ['Intellectuality','Depth','Insight','Intelligence']
}

traits_neg = {
'agreeableness':['Belligerence','Overcriticalness','Bossiness','Rudeness','Cruelty','Pomposity','Irritability',
                    'Conceit','Stubbornness','Distrust','Selfishness','Callousness'],
'conscientiousness':['Disorganization','Negligence','Inconsistency','Forgetfulness','Recklessness','Aimlessness',
                        'Sloth','Indecisiveness','Frivolity','Nonconformity'],
'surgency':['Pessimism','Lethargy','Passivity','Unaggressiveness','Inhibition','Reserve','Aloofness'],
'emotional_stability':['Insecurity','Emotionality'],
'intellect':['Shallowness','Unimaginativeness','Imperceptiveness','Stupidity']
}

data_traits = data_indiv['traits'].str.split(', ', expand=True)
data_traits.columns=['agreeableness', 'conscientiousness', 'surgency', 'emotional_stability', 'intellect']
for column in ['agreeableness', 'conscientiousness', 'surgency', 'emotional_stability', 'intellect']:
    data_traits[column + '_score'] = data_traits[column].map(lambda x: 1 if x in traits_pos[column] else (0 if x in traits_neg[column] else x))

df_traits = data_traits[['agreeableness_score','conscientiousness_score','surgency_score','emotional_stability_score','intellect_score']]

#####################################################
#                        Age                        #
#####################################################

df_age = pd.DataFrame(data_indiv['age'], columns=['age'])

#####################################################
#                        Name                       #
#####################################################

data_name=data_indiv['name']

s=2000 #Change for # of sampled names in run
country_alpha2='US'
if s % 2 == 1:
    s += 1
male_names = nd.get_top_names(s//2, 'Male', country_alpha2)[country_alpha2]['M']
female_names = nd.get_top_names(s//2, 'Female', country_alpha2)[country_alpha2]['F']


# Store male and female names into dictionaries with their rank
male_name_rank = {name: rank+1 for rank, name in enumerate(male_names)}
female_name_rank = {name: rank+1 for rank, name in enumerate(female_names)}

# Create new lists for gender and rank
gender = []
rank = []
for name in data_name:
    if name in male_name_rank:
        rank.append(male_name_rank[name])
        gender.append(1)
    elif name in female_name_rank:
        rank.append(female_name_rank[name])
        gender.append(0)
    else:
        rank.append(None)
        gender.append(None)

# Convert lists into series
gender = pd.Series(gender, name='gender')
rank = pd.Series(rank, name='Name Rank')

#Normalize rank
rank = rank.divide((s/2))
rank = 1+1/(s/2)-rank
df_name = pd.concat([data_name.rename('name'), gender, rank], axis=1)

#####################################################
#                   Formatting Data                 #
#####################################################
df_response_array = df_response.values.ravel(order='F')
df_response_matrix=pd.DataFrame(df_response_array, columns=['Response'])



df_health_string_array=health_string_df.values.ravel(order='F')

health_condition_strings = ["feels normal", "has a light cough", "has a fever and a cough"]

def assign_condition(row):
    for condition in health_condition_strings:
        if condition in row['Statement']:
            return condition
    return None

df_health_string_matrix = pd.DataFrame(df_health_string_array, columns=['Statement'])

df_health_string_matrix['Condition'] = df_health_string_matrix.apply(assign_condition, axis=1)

def assign_name(row):
    for condition in health_condition_strings:
        if condition in row['Statement']:
            name = row['Statement'].replace(condition, '')
            return name.strip()  # To remove extra whitespaces at the start or end

df_health_string_matrix['Name'] = df_health_string_matrix.apply(assign_name, axis=1)

df_health_string_matrix.drop(columns=['Statement'], inplace=True)
df_health_string_matrix.drop(columns=['Name'],inplace=True)

df_health_string_matrix=pd.get_dummies(df_health_string_matrix, columns=['Condition'])

static_bio_info=pd.concat([df_name,df_age,df_traits],axis=1)


repeat_factor=df_health_string_matrix.shape[0]//static_bio_info.shape[0]

#Extending static bio info
static_bio_info_matrix = pd.concat([static_bio_info] * repeat_factor, ignore_index=True)

#Appending cases
df_new_case_alert=pd.DataFrame(data_run, columns=['Daily New Cases Day 4'])

#Time Step
df_time_step=pd.DataFrame(data_run, columns=['Step'])
df_time_step=df_time_step.rename(columns={'Step':'Time Step'})

repeat_factor_new_cases=df_response.shape[0]

# Repeat each row
repeated_df_new_case_alert = np.repeat(df_new_case_alert.values, num_agents, axis=0)

time_step_range=list(range(0, repeat_factor))
repeated_df_time_step = np.repeat(time_step_range,num_agents, axis=0)

# Convert the repeated data back to a DataFrame
df_new_case = pd.DataFrame(repeated_df_new_case_alert, columns=df_new_case_alert.columns)
df_new_case = df_new_case.divide(num_agents)

df_time_step=pd.DataFrame(repeated_df_time_step, columns=df_time_step.columns)

df_logistic_regression=pd.concat([static_bio_info_matrix,df_health_string_matrix,df_new_case,df_time_step,df_response_matrix],axis=1)

df_full = pd.concat([df_full, df_logistic_regression])

#Save processed outputs containing both responses and statistics over time for all agents in a .csv file.
df_full_logistic_regression.to_csv("R03_n1000_Indiv_Data_for_logistic_regression.csv")
