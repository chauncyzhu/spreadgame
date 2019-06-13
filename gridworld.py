import numpy as np
import math
import time
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
import collections
import matplotlib
from matplotlib import cm
import itertools

def restrict(val, minval, maxval):
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val

class Entity(object):
    def __init__(self):
        self.position = np.array([0, 0])


class World(object):
    def __init__(self,
                 height,
                 width,
                 stack,
                 n_agents,
                 n_landmarks,
                 t_max,
                 seed,
                 evaluation_seed,
                 n_evaluation_levels,

                 observation_type='vector',
                 distance_metric='manhattan',
                 reward_mode='sparse',
                 world_geometry='flat'
                 ):

        assert observation_type in {'grid', 'vector'}
        assert distance_metric in {'euclidean', 'manhattan'}
        assert reward_mode in {'dense', 'sparse'}
        assert world_geometry in {'flat', 'toroidal'}

        self.n_agents = n_agents
        self.n_landmarks = n_landmarks

        self.height = height
        self.width = width
        self.timestep_stack = stack

        self.reward_mode = reward_mode

        self.max_t = t_max



        self.distance_metric = 'manhattan'

        self.observation_type = observation_type

        self.render = None

        self.n_entities = self.n_agents + self.n_landmarks

        self.observation_grid = None
        self.observation_vector = None
        self.observation_grid_stack = None

        self.obs_images = None

        self.agent_colorset = [np.asarray([255, 0, 0]), np.asarray([0, 255, 0]), np.asarray([0, 0, 255]),
                               np.asarray([255, 255, 0]), np.asarray([0, 255, 255]), np.asarray([255, 0, 255]),
                               np.asarray([255, 128, 0]), np.asarray([255, 0, 128]), np.asarray([128, 255, 0])] # TODO

        self.agent_colors = self.agent_colorset[0:self.n_agents]
        self.landmark_colors = self.n_landmarks * [np.asarray([150, 150, 150])]

        #print(self.agent_colors)
        #print(self.landmark_colors)
        #print(self.agent_colors + self.landmark_colors)

        self.t = None

        self.agents = [Entity() for _ in range(self.n_agents)]
        self.landmarks = [Entity() for _ in range(self.n_landmarks)]

        self.max_distance = self.get_max_distance()
        self.min_score = -self.max_distance * self.n_landmarks

        self.regions = self.generate_regions(self.height, self.width)
        self.curriculum_regions = self.generate_curriculum_regions(self.height, self.width)

        state = np.random.get_state()
        self.evaluation_levels = self.generate_evaluation_levels(n_evaluation_levels, evaluation_seed)
        np.random.set_state(state)

        self.evaluation_levels_scores = self.determine_evaluation_level_scores()

        for a, b in itertools.combinations(self.evaluation_levels, 2):
            if np.array_equal(a, b):
                print('SAME!')

       # print(collections.Counter(self.evaluation_levels))

        # print(self.evaluation_levels_scores)

        np.random.seed(seed)

    def get_distance(self, position_1, position_2):
        if self.distance_metric == 'euclidean':
             return np.sqrt(np.sum(np.square(position_1 - position_2)))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.absolute(position_1 - position_2))

    def get_max_distance(self):
        return self.get_distance(np.array([0, 0]), np.array([self.height, self.width]))

    def generate_evaluation_levels(self, n_levels, seed=1):
        np.random.seed(seed)
        return [self.generate_level() for _ in range(n_levels)]

    def determine_evaluation_level_scores(self):
        scores = []
        for i, level in enumerate(self.evaluation_levels):
            obs = self.reset(evaluation_level_id=i)
            score = 0.0
            while True:
                actions = [self.best_action(obs[i], self.n_agents, self.n_landmarks) for i in range(self.n_agents)]
                obs, reward, done = self.step(actions)
                score += reward
                if done:
                    break
            scores.append(score)
        return scores




    # obs, max_score = env.reset()
    # done = False

    # while not done:
    # print(obs[0])
    # print(obs[1])
    # print(obs[2])
    # actions = [obs_to_positions(obs[i]) for i in range(3)]
    # obs, reward, done = env.step(actions)
    # print(reward)

    def generate_regions(self, height, width):

        regions = [[], [], [], [], [], []]

        # Center
        for h in range(round(height * 0.3), round(height * 0.7)):
            for w in range(round(width * 0.3), round(width * 0.7)):
                regions[0].append(h * height + w)

        # Corners
        for h in range(0, round(height * 0.3)):
            for w in range(0, round(width * 0.3)):
                regions[1].append(h * height + w)
            for w in range(round(width * 0.7), width):
                regions[1].append(h * height + w)

        for h in range(round(height * 0.7), height):
            for w in range(0, round(width * 0.3)):
                regions[1].append(h * height + w)
            for w in range(round(width * 0.7), width):
                regions[1].append(h * height + w)

        # Right
        for h in range(0, height):
            for w in range(0, round(width * 0.5)):
                regions[2].append(h * height + w)

        # Left
        for h in range(0, height):
            for w in range(round(width * 0.5), width):
                regions[3].append(h * height + w)

        # Up
        for h in range(0, round(height * 0.5)):
            for w in range(0, width):
                regions[4].append(h * height + w)

        # Down
        for h in range(round(height * 0.5), height):
            for w in range(0, width):
                regions[5].append(h * height + w)

        return regions

    def generate_curriculum_regions(self, height, width, n_stages=5):
        regions = []
        percentage_base = 1.0/n_stages

        for stage in range(n_stages):
            regions.append([])

            percentage = percentage_base*(stage + 1)
            percentage_start = (1.0 - percentage)/2
            percentage_end = percentage_start + percentage

            for h in range(round(height * percentage_start), round(height * percentage_end)):
                for w in range(round(width * percentage_start), round(width * percentage_end)):
                    regions[-1].append(h * height + w)

        return regions


    def generate_level(self, level_structure=0):
        if level_structure == 0:
             return np.random.choice(self.height * self.width, self.n_entities, replace=False)
        elif level_structure == 1:  # Agents RIGHT, Landmarks LEFT
            agents = np.random.choice(self.regions[2], self.n_agents, replace=False)
            landmarks = np.random.choice(self.regions[3], self.n_landmarks, replace=False)
            return list(agents) + list(landmarks)
        elif level_structure == 2:  # Agents LEFT, Landmarks RIGHT
            agents = np.random.choice(self.regions[3], self.n_agents, replace=False)
            landmarks = np.random.choice(self.regions[2], self.n_landmarks, replace=False)
            return list(agents) + list(landmarks)
        elif level_structure == 3:  # Agents UP, Landmarks DOWN
            agents = np.random.choice(self.regions[4], self.n_agents, replace=False)
            landmarks = np.random.choice(self.regions[5], self.n_landmarks, replace=False)
            return list(agents) + list(landmarks)
        elif level_structure == 4:  # Agents DOWN, Landmarks UP
            agents = np.random.choice(self.regions[5], self.n_agents, replace=False)
            landmarks = np.random.choice(self.regions[4], self.n_landmarks, replace=False)
            return list(agents) + list(landmarks)

    def generate_level_byc(self, curriculum_level=None):
        if curriculum_level is None:
            return np.random.choice(self.height * self.width, self.n_entities, replace=False)
        else:
            return np.random.choice(self.curriculum_regions[curriculum_level], self.n_entities, replace=False)


    def get_visualization(self):
        return self.obs_images, self.render_traces()

    def reset(self, evaluation_level_id=None, curriculum_level=None, level_structure=0, render=False):

        self.render = render

        if self.render:
            self.obs_images = []
            self.obs_trace = np.zeros((self.height, self.width, self.n_agents + self.n_landmarks),
                                      dtype=np.float32)
        else:
            self.obs_images = None
            self.obs_trace = None

        if evaluation_level_id is None:
            # positions = self.generate_level_byc(curriculum_level=curriculum_level)
            positions = self.generate_level(level_structure)
        else:
            positions = self.evaluation_levels[evaluation_level_id]

        # Distribute positions over grid
        for position, entity in zip(positions, self.agents + self.landmarks):
            entity.position = np.array([math.floor(position / self.height), position % self.width])

        #max_score = self.best_possible_episode_reward()

        self.t = 0

        self.observation_grid_stack = [collections.deque(maxlen=self.timestep_stack) for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            for _ in range(self.timestep_stack):
                self.observation_grid_stack[i].append(np.zeros((self.height, self.width, 3), dtype=np.uint8))

        self.update_observation()

        if self.observation_type == 'grid':
            return [np.dstack(observation_grid_stack) for observation_grid_stack in self.observation_grid_stack]
        if self.observation_type == 'vector':
            return self.observation_vector

    def best_possible_episode_reward(self):

        distances = np.zeros((self.n_agents, self.n_landmarks), dtype=np.int32)

        for i, agent in enumerate(self.agents):
            for j, landmark in enumerate(self.landmarks):
                distances[i, j] = self.get_distance(agent.position, landmark.position)

        agent_p = list(itertools.permutations(range(self.n_agents)))
        landmark_p = list(itertools.permutations(range(self.n_landmarks)))

        pairings_list = []
        for i in range(len(agent_p)):
            for j in range(len(landmark_p)):
                pairings = []
                for k in range(min(self.n_agents, self.n_landmarks)):
                    pairings.append((agent_p[i][k], landmark_p[j][k]))
                pairings_list.append(sorted(pairings))

        pairings_set = set(tuple(i) for i in pairings_list)

        min_cost = np.inf
        min_cost_pairing = None
        for pairing in pairings_set:
            #print(pairing)
            cost = 0.0
            for pair in pairing:
                #print(pair, distances[pair[0], pair[1]])
                cost += distances[pair[0], pair[1]]
            #print(cost)

            if cost < min_cost:
                min_cost = cost
                min_cost_pairing = pairing

        #print(min_cost_pairing, min_cost)

        max_score = 0.0
        for pair in min_cost_pairing:
            max_score += self.max_t - distances[pair[0], pair[1]]

        #print(max_score)

        max_score /= self.n_agents

        return max_score

    def simulate_episode(self):
        pass

    def render_traces(self):
        obs_trace_image = np.zeros((self.height, self.width, 3), dtype=np.float32)

        #print(self.landmark_colors)

        for h in range(self.height):
            for w in range(self.width):
                color = np.zeros(3, dtype=np.float32)
                for k, entity_color in enumerate(self.agent_colors + self.landmark_colors):
                    color += (self.obs_trace[h, w, k] * entity_color)
                    #print(k, color)

                obs_trace_image[h, w, :] = color

        obs_trace_image = np.clip(obs_trace_image, 0.0, 255.0)
        obs_trace_image = cv2.resize(obs_trace_image, (500, 500), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return obs_trace_image

    def render(self):  # TODO: To be implemented.
        pass

    def observation(self):  # TODO: To be implemented.
        pass

    def update_observation(self):

        self.observation_grid = []
        self.observation_vector = []

        for i, agent_i in enumerate(self.agents):

            self.observation_grid.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            observation_vector_ind = []

            for j, agent_j in enumerate(self.agents):
                if j == i:  # Agent itself
                    self.observation_grid[i][agent_j.position[0], agent_j.position[1], 0] = 1
                else:  # Other agents
                    self.observation_grid[i][agent_j.position[0], agent_j.position[1], 1] = 1
                    observation_vector_ind += [(agent_j.position[0] - agent_i.position[0])/float(self.height),
                                                (agent_j.position[1] - agent_i.position[1])/float(self.width)]

            for landmark in self.landmarks:
                self.observation_grid[i][landmark.position[0], landmark.position[1], 2] = 1
                observation_vector_ind += [(landmark.position[0] - agent_i.position[0])/float(self.height),
                                            (landmark.position[1] - agent_i.position[1])/float(self.width)]

            # [2.0*(1.0 - self.t/float(self.max_t))-1.0]

            self.observation_vector.append(np.array(observation_vector_ind + [2.0*(1.0 - self.t/float(self.max_t))-1.0],
                                                    dtype=np.float32))

            #self.observation_vector.append(np.array(observation_vector_ind, dtype=np.float32))

            self.observation_grid_stack[i].append(np.copy(self.observation_grid[i]))

        #self.observation_grid_image = None

        if self.render:
            obs_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            for i, agent in enumerate(self.agents):
                obs_image[agent.position[0], agent.position[1], :] = self.agent_colors[i]

            for i, landmark in enumerate(self.landmarks):
                obs_image[landmark.position[0], landmark.position[1], :] = self.landmark_colors[i]
                # check if any agent occupies it
                for k, agent in enumerate(self.agents):
                    if self.get_distance(landmark.position, agent.position) == 0:
                        obs_image[landmark.position[0], landmark.position[1], :] = [255, 255, 255]
                        break

            obs_image = cv2.resize(obs_image, (500, 500), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            self.obs_images.append(obs_image)

            # Update trace
            self.obs_trace *= 0.95
            for i, entity in enumerate(self.agents + self.landmarks):
                self.obs_trace[entity.position[0], entity.position[1], i] = 1.0

    def update_state(self, actions):
        agent_positions_initial = []
        agent_positions = []

        for i, agent in enumerate(self.agents):
            agent_positions_initial.append(np.array(agent.position))

            if actions[i] == 1:
                agent.position[0] += 1
            elif actions[i] == 2:
                agent.position[0] -= 1
            elif actions[i] == 3:
                agent.position[1] += 1
            elif actions[i] == 4:
                agent.position[1] -= 1

            #if agent.position[0] == -1:
            #    agent.position[0] = self.height - 1
            #elif agent.position[0] == self.height:
            #    agent.position[0] = 0

            #if agent.position[1] == -1:
            #    agent.position[1] = self.width - 1
            #elif agent.position[1] == self.width:
            #    agent.position[1] = 0

            # check for edges
            agent.position[0] = restrict(agent.position[0], 0, self.height - 1)
            agent.position[1] = restrict(agent.position[1], 0, self.width - 1)

            agent_positions.append(agent.position)

        n_collisions = 0

        agent_collided = [False] * self.n_agents
        for i in range(self.n_agents):
            for j in range((i + 1), self.n_agents):
                if np.array_equal(agent_positions[i], agent_positions[j]):  # collision happened
                    agent_collided[i] = True
                    agent_collided[j] = True
                    n_collisions += 1

        # for i, agent in enumerate(self.agents):
        #    if agent_collided[i]:
        #        agent.position = np.array(agent_positions_initial[i])

        self.t += 1
        done = self.t >= self.max_t

        if self.render:
            self.render_traces()

        return done

    def determine_reward(self):
        reward = 0.0

        if self.reward_mode == 'dense':
            for i, landmark in enumerate(self.landmarks):
                distances = [self.get_distance(landmark.position, agent.position) for agent in self.agents]
                reward -= min(distances)

            # Normalize to [0, 1]
            reward = 1.0 + (reward / abs(self.min_score))



        elif self.reward_mode == 'sparse':
            for i, landmark in enumerate(self.landmarks):
                distances = [self.get_distance(landmark.position, agent.position) for agent in self.agents]
                if min(distances) == 0:
                    reward += 1.0

            # Normalize to [0, 1]
            reward /= float(self.n_agents)


        return reward

    def best_action(self, obs, n_agents, n_landmarks):

        agents = [Entity() for _ in range(n_agents)]
        landmarks = [Entity() for _ in range(n_landmarks)]

        for i, agent in enumerate(agents):
            if i == 0:
                agent.position = np.array([0, 0])
            else:
                start = (i - 1) * 2
                end = start + 2
                agent.position = np.array(obs[start:end])

        for i, landmark in enumerate(landmarks):
            start = (n_agents - 1) * 2 + (i) * 2
            end = start + 2
            landmark.position = np.array(obs[start:end])

        distances = np.zeros((n_agents, n_landmarks), dtype=np.float32)

        for i, agent in enumerate(agents):
            for j, landmark in enumerate(landmarks):
                distances[i, j] = self.get_distance(agent.position, landmark.position)

        agent_p = list(itertools.permutations(range(n_agents)))
        landmark_p = list(itertools.permutations(range(n_landmarks)))

        pairings_list = []
        for i in range(len(agent_p)):
            for j in range(len(landmark_p)):
                pairings = []
                for k in range(min(n_agents, n_landmarks)):
                    pairings.append((agent_p[i][k], landmark_p[j][k]))
                pairings_list.append(sorted(pairings))

        pairings_set = set(tuple(i) for i in pairings_list)

        min_cost = np.inf
        min_cost_pairing = None
        for pairing in pairings_set:
            cost = 0.0
            for pair in pairing:
                cost += distances[pair[0], pair[1]]

            if cost < min_cost:
                min_cost = cost
                min_cost_pairing = pairing

        target_landmark = min_cost_pairing[0][1]

        action = 0

        if landmarks[target_landmark].position[0] > 0:
            action = 1
        elif landmarks[target_landmark].position[0] < 0:
            action = 2

        if landmarks[target_landmark].position[1] > 0:
            action = 3
        elif landmarks[target_landmark].position[1] < 0:
            action = 4

        return action


    def step(self, actions):

        done = self.update_state(actions)
        reward = self.determine_reward()

        self.update_observation()

        if self.observation_type == 'grid':
            return [np.dstack(observation_grid_stack) for observation_grid_stack in self.observation_grid_stack], \
               reward, done
        if self.observation_type == 'vector':
            # print(self.observation_vector)
            return self.observation_vector, reward, done

# ==================
