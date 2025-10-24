from environment.agent import UserInputAgent, ConstantAgent, run_real_time_match

my_agent = UserInputAgent()
opponent = ConstantAgent()
run_real_time_match(my_agent,
                    agent_2=opponent,
                    max_timesteps=30*300)