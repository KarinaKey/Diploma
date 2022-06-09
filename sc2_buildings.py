from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.data import Difficulty, Race, Result  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
import numpy as np
import os.path
from os import path
import pandas as pd

#
actions = [
	"no_action",
	"create_drone",
	#"create_overlord",
	#"create_queen",
	#"create_larva",
	"build_hatchery",
	"build_gas"
	#"distribute_workers"
]

build_reward = 0.5
mineral_reward = 0.02
destroy_reward = - 0.5


class Qlearning:
    def __init__(self, actions, filepath, alpha=0.1, gamma=0.9, epsilon=0.9):
        self.actions = actions
        self.alpha = alpha # learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = self.load_table(filepath) if path.exists(filepath) else pd.DataFrame(columns=self.actions, dtype=np.float64)


    def load_table(self, filepath):
        return pd.read_csv(filepath, index_col=0)

    def save_qtable(self, filepath):
        self.qtable.to_csv(filepath)

    def choose_action(self, state):
        self.check_state(state)

        if np.random.uniform() < self.epsilon:
            actions_by_state = self.qtable.loc[state, :]
            actions_by_state = actions_by_state.reindex(np.random.permutation(actions_by_state.index))
            action = actions_by_state.idxmax()

        else:
            action = np.random.choice(self.actions)
        return action

    def check_state(self, state):
        if state not in self.qtable.index:
            self.qtable = self.qtable.append(pd.Series([0]*len(self.actions), index=self.qtable.columns, name=state))

    def learn(self, pr_state, pr_action, reward, cur_state):
        self.check_state(cur_state)
        self.check_state(pr_state)
        try:
            q_pr = self.qtable.loc[pr_state, pr_action]
        except:
            q_pr = self.qtable.loc[pr_state, str(pr_action)]


        q_cur = reward + self.gamma * self.qtable.loc[cur_state, :].max()
        print(q_cur)
        try:
            self.qtable.loc[pr_state, pr_action] += self.alpha * (q_cur - q_pr)
        except:
            self.qtable.loc[pr_state, str(pr_action)] += self.alpha * (q_cur - q_pr)



class MyBot(BotAI):
    def __init__(self):
        self.path = 'D:\\MMF\\DIPLOMA\\Qtable.csv'
        self.qlearn = Qlearning(actions=list(range(len(actions))), filepath=self.path)
        self.pr_action = None
        self.pr_state = None
        self.pr_minerals = 0
        self.pr_buildings = 0
        self.pr_gas = 0

    async def on_end(self, game_result: Result):
        self.qlearn.save_qtable(self.path)


    async def on_step(self, iteration: int):

        print('buildings:', self.structures(UnitTypeId.HATCHERY).amount + self.structures(UnitTypeId.EXTRACTOR).amount)
        #print('estractors:', self.structures(UnitTypeId.EXTRACTOR).amount)
        #print('overlords:', self.units(UnitTypeId.OVERLORD).amount)

        await self.distribute_workers()

        drones = self.units(UnitTypeId.DRONE).amount
        #overlords = self.units(UnitTypeId.OVERLORD).amount
        #queens = self.units(UnitTypeId.QUEEN).amount
        hatches = self.structures(UnitTypeId.HATCHERY).amount
        extractors = self.structures(UnitTypeId.EXTRACTOR).amount

        if hatches + extractors == 20:
            print(iteration)
            f = open('D:\\MMF\\DIPLOMA\\res_test.txt', 'a')
            f.write('\n'+str(iteration))
            f.close()
            await self._client.leave()
        if iteration == 0:
            await self.chat_send("(glhf)")

        cur_state = [drones,hatches,extractors]


        if self.pr_state is not None:
            reward = 0

            minerals_prev = self.pr_minerals
            buildings_prev = self.pr_buildings

            self.pr_minerals = self.minerals
            self.pr_buildings = self.structures(UnitTypeId.HATCHERY).amount + self.structures(UnitTypeId.EXTRACTOR).amount

            if self.pr_minerals > minerals_prev + drones - 3*extractors:
                reward += mineral_reward
            if self.pr_buildings > buildings_prev:
                reward += build_reward
            elif self.pr_buildings < buildings_prev:
            	reward += destroy_reward


            self.qlearn.learn(str(self.pr_state), self.pr_action, reward, str(cur_state))

        rl_action = self.qlearn.choose_action(str(cur_state))
        action = actions[int(rl_action)]

        self.pr_state = cur_state
        self.pr_action = rl_action

        if action == "no_action":
            pass
        elif action == "create_drone":
            await self.create_drone()
        #elif action == "create_overlord":
        #    await self.create_overlord()
        #elif action == "create_queen":
        #    await self.create_queen()
        #elif action == "create_larva":
        #	await self.create_larva()
        elif action == "build_hatchery":
            await self.build_hatchery()
        elif action == "build_gas":
            await self.build_gas()

    async def create_drone(self):
        if self.can_afford(UnitTypeId.DRONE):
            self.train(UnitTypeId.DRONE)
        else:
        	self.create_overlord()


    async def create_overlord(self):
        if self.can_afford(UnitTypeId.OVERLORD):
            self.train(UnitTypeId.OVERLORD)

    async def build_hatchery(self):
        if self.can_afford(UnitTypeId.HATCHERY):
            await self.expand_now()

    async def build_gas(self):
        if self.can_afford(UnitTypeId.EXTRACTOR) and self.workers:
            # build from the closest pylon towards the enemy
            worker = self.workers.random
            gas = self.vespene_geyser.closest_to(worker)
            worker.build_gas(gas)
                    
if __name__ == '__main__':
    run_game(
        maps.get("TritonLE"),
        [Bot(Race.Zerg, MyBot()), Computer(Race.Zerg, Difficulty.Easy)],
        realtime=False,
    )
