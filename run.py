from screen import *
import numpy as np
from game import Game
from QLearn import DQNAgent
import torch.optim as optim
import torch
import datetime
import time
import win32com.client as wincl

DEVICE = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def initialize_game(player, game, food, agent, batch_size):
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = True
    score, mean, stdev = run(params)
    print("score:"+str(score)+"\nmean:"+str(mean)+"\nstdev:"+str(stdev))
    return score, mean, stdev


def run(params):
    """
    Run the Q learn algorithm, based on the parameters previously set.
    """
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Initialize classes
        gameA = Game(440, 440)
        player1 = gameA.player
        food1 = gameA.food

        # Perform first move
        initialize_game(player1, gameA, food1, agent, params['batch_size'])
        if params['display']:
            display(player1, food1, gameA, record)
        # start = time.time()
        while not gameA.crash:
            if not params['train']:
                agent.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(gameA, player1, food1)

            from random import uniform
            from random import randint

            # perform random actions based on agent.epsilon, or choose the action
            if uniform(0, 1) < agent.epsilon:
                final_move = np.eye(3)[randint(0, 2)]
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, gameA, food1, agent)
            state_new = agent.get_state(gameA, player1, food1)

            # set reward for the new state
            reward = agent.set_reward(player1, gameA.crash)

            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, gameA.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, gameA.crash)

            record = get_record(gameA.score, record)
            if params['display']:
                display(player1, food1, gameA, record)
                pygame.time.wait(params['speed'])
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        total_score += gameA.score
        print(f'Game {counter_games}      Score: {gameA.score}')
        score_plot.append(gameA.score)
        counter_plot.append(counter_games)
    mean, stdev = get_mean_stdev(score_plot)
    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])
    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])
        print("score:" + str(total_score) + "\nmean:" + str(mean) + "\nstdev:" + str(stdev))
    return total_score, mean, stdev


def init(params):
    #
    # Neural Network
    params['epsilon_decay_linear'] = 1 / 90
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200  # neurons in the first layer
    params['second_layer_size'] = 20  # neurons in the second layer
    params['third_layer_size'] = 50  # neurons in the third layer
    params['episodes'] = 150
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['load_weights'] = False
    params['train'] = True
    params["test"] = True
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + '.txt'

    params['display'] = True
    params['speed'] = 50
    return params


pygame.font.init()
params = dict()
params = init(params)
# if args.bayesianopt:
#     bayesOpt = BayesianOptimizer(params)
#     bayesOpt  .optimize_RL()
if params['train']:
    start = time.time()
    a = run(params)
    print(time.time()-start)
if params['test']:
    b = test(params)
    print(time.time() - start)
print("train")
print(a)
print("test")
print(b)
#  alarm sound
speaker = wincl.Dispatch("SAPI.SpVoice")
speaker.Speak("Training has finished, Uriell i want to get a hickeey from hadar. will you help me?")
