import openai
from world import World
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GABM", help = "Name of the run to save outputs.")
    parser.add_argument("--contact_rate", default=5, type=int, help="Contact Rate")
    parser.add_argument("--infection_rate", default=0.1, type=float, 
                        help="Infection Rate")
    parser.add_argument("--no_init_healthy", default=98, type=int, 
                        help="Number of initial healthy people in the world.")
    parser.add_argument("--no_init_infect", default= 2, type=int,
                        help="Number of initial infected people in the world.")
    parser.add_argument("--no_days", default=50, type=int,
                        help="Total number of days the world should run.")
    parser.add_argument("--time_to_heal", default=6,type=int, help="Time taken to heal from infection.")
    parser.add_argument("--no_of_runs", default = 1, type = int, help = "Total number of times you want to run this code.")
    parser.add_argument("--offset", default=0,type=int, help="offset is equal to number of days if you need to load a checkpoint")
    parser.add_argument("--load_from_run", default=0,type=int, help="equal to run # - 1 if you need to load a checkpoint (e.g. if you want to load run 2 checkpoint 8, then offset = 8, load_from_run = 1)")

    args = parser.parse_args()
    print(f"Parameters: {args}")
    if os.path.exists("output") is not True:
        os.mkdir("output")
    if os.path.exists("checkpoint") is not True:
        os.mkdir("checkpoint")
    openai.api_key = "use your OpenAI API key here"
    for i in range(args.load_from_run, args.no_of_runs):
        print(f"--------Run - {i+1}---------")
        checkpoint_path = f"checkpoint/run-{i+1}"
        output_path = f"output/run-{i+1}"
        if os.path.exists(checkpoint_path) is not True:
            os.mkdir(checkpoint_path)
        if os.path.exists(output_path) is not True:
            os.mkdir(output_path)

        if args.load_from_run != 0:  # Load specific checkpoint only from the specified run
            checkpoint_file = f"checkpoint/run-{args.load_from_run+1}/{args.name}-{args.offset}.pkl"
            if os.path.exists(checkpoint_file):
                model = World.load_checkpoint(checkpoint_file)
            else:
                print(f"Warning! Checkpoint not found. Initializing new world for run {args.load_from_checkpoint+1}. This is normal if you want to continue from run {args.load_from_checkpoint+1} from scratch")
                model = World(args, initial_healthy=args.no_init_healthy, initial_infected=args.no_init_infect,contact_rate=args.contact_rate)
        else:
            model = World(args, initial_healthy=args.no_init_healthy, initial_infected=args.no_init_infect,contact_rate=args.contact_rate)

        model.run_model(checkpoint_path, args.offset)
        data = model.datacollector.get_model_vars_dataframe() #collect data from the successful run of the model

        df = pd.DataFrame(data)
        new_infections_newspaper=model.list_new_cases[:-1]
        new_infections_newspaper[0]=model.list_new_cases[0]+model.initial_infected
        new_infections_newspaper[1]=model.list_new_cases[1]-model.initial_infected
        df['New Infections']=new_infections_newspaper
        df['Cumulative Infections'] = df['New Infections'].cumsum()
        df['Total Contact'] = model.track_contact_rate[:len(df)]
        df["Daily New Cases Day 4"] = model.day_infected_is_4[:len(df)]

        #Insert a step column function
        df.insert(0, 'Step',range(0,len(df)))

        #save data
        df.to_csv(output_path+f"/{args.name}-data.csv")

        #plot and save required figures for each run
        plt.figure(figsize=(10,6))
        plt.plot(df['Step'], df['Susceptible'], label="Susceptible")
        plt.plot(df['Step'], df['Infected'], label="Infected")
        plt.plot(df['Step'] ,df['Recovered'], label="Recovered")
        plt.xlabel('Step')
        plt.ylabel('# of People')
        plt.title('SIR')
        plt.legend()
        plt.savefig(output_path+f'/{args.name}-SIR.png', bbox_inches='tight')


        plt.figure(figsize=(10,6))
        plt.plot(df['Step'], df['# Grid'], label="Citizens outside")
        plt.plot(df['Step'], df['# Home'], label="Citizens at Home")
        plt.xlabel('Step')
        plt.ylabel('# of People')
        plt.title('SIR')
        plt.legend()
        plt.savefig(output_path + f'/{args.name}-NumHome.png', bbox_inches='tight')

        #save final checkpoint after successful run
        model.save_checkpoint(file_path = checkpoint_path + f"/{args.name}-completed.pkl")
